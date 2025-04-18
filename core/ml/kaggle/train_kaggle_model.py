import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import os
import pickle
import tqdm
import warnings

# Suppress potential warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")


# --- Data Handling ---

def robust_transform(scaler, X):
    """Apply scaling and clip extreme values."""
    X_scaled = scaler.transform(X)
    # Cap extreme values at +/- 5 standard deviations
    return np.clip(X_scaled, -5, 5)

# Define the collate_fn globally to make it picklable
def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences, padding, and weights.
    """
    try:
        # Unpack sequences, labels, and weights
        sequences, labels, weights = zip(*batch)
    except ValueError:
        print("Warning: Received empty batch or batch items are not tuples of 3.")
        return None, None, None, None # Indicate an issue

    # Filter out None items if any sequence failed in __getitem__
    valid_batch = [(s, l, w) for s, l, w in zip(sequences, labels, weights) if s is not None]
    if not valid_batch:
        print("Warning: Batch contains only None items after filtering.")
        return None, None, None, None
    
    sequences, labels, weights = zip(*valid_batch)

    # Get lengths of each sequence
    lengths = [len(seq) for seq in sequences]
    if not lengths or max(lengths) == 0:
         print(f"Warning: Batch has sequences of zero length. Lengths: {lengths}")
         return None, None, None, None # Cannot proceed with zero-length sequences

    max_len = max(lengths)
    feature_dim = sequences[0].size(-1) # Get feature dimension from the first sequence

    # Pad sequences
    padded_seqs = torch.zeros(len(sequences), max_len, feature_dim, dtype=torch.float32)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long) # Labels are class indices
    padded_weights = torch.zeros(len(weights), max_len, dtype=torch.float32)

    for i, (seq, label, weight) in enumerate(zip(sequences, labels, weights)):
        end = lengths[i]
        if end > 0: # Ensure sequence is not empty
            padded_seqs[i, :end] = seq
            padded_labels[i, :end] = label
            padded_weights[i, :end] = weight

    return padded_seqs, padded_labels, padded_weights, torch.LongTensor(lengths)


class JazzPhraseDataset(Dataset):
    """
    Dataset for jazz phrase segmentation, creating sequences from melodies.
    """
    def __init__(self, features, labels, sequence_length=128, stride=32):
        self.features = features # Expects features with melid in the first column
        self.labels = labels
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = []

        if self.features is None or len(self.features) == 0:
             print("Warning: Features array is None or empty.")
             return
        if self.labels is None or len(self.labels) != len(self.features):
             print("Warning: Labels array is None or size mismatch with features.")
             return

        # Calculate Global Weight Factor
        global_class_0 = np.sum(self.labels == 0)
        global_class_1 = np.sum(self.labels == 1)
        global_ratio = global_class_0 / max(global_class_1, 1)
        weight_factor = min(global_ratio, 10.0) # Cap weight factor
        print(f"Global class ratio (0/1): {global_ratio:.2f}, using weight factor: {weight_factor:.2f}")

        # Group by Melody ID
        try:
             df = pd.DataFrame(self.features, columns=['melid'] + [f'feat_{i}' for i in range(1, self.features.shape[1])])
             df['label'] = self.labels
             df['original_index'] = np.arange(len(df)) # Keep track of original indices
             grouped = df.groupby('melid')
        except Exception as e:
            print(f"Error grouping features by melid: {e}")
            grouped = {}


        # Create Sequences per Melody
        context_window = 2 # For weighting context around phrase starts

        for melid, group in grouped:
            # Sort by original index to maintain time order
            group = group.sort_values(by='original_index')
            mel_features = group.iloc[:, 1:-2].values.astype(np.float32) # Features excluding melid, label, index
            mel_labels = group['label'].values.astype(np.int64)

            if len(mel_features) == 0:
                continue

            # Generate Overlapping Sequences
            for start_idx in range(0, len(mel_features), self.stride):
                end_idx = min(start_idx + self.sequence_length, len(mel_features))

                # Skip sequences that are too short
                if end_idx - start_idx < 10:
                    continue

                seq_features = torch.from_numpy(mel_features[start_idx:end_idx])
                seq_labels = torch.from_numpy(mel_labels[start_idx:end_idx])

                # Calculate Weights for the Sequence
                seq_weights = torch.ones(seq_labels.size(0), dtype=torch.float32)
                phrase_start_indices = torch.nonzero(seq_labels == 1).squeeze(1)

                if len(phrase_start_indices) > 0:
                    # Assign base weight to phrase starts
                    seq_weights[phrase_start_indices] = weight_factor

                    # Apply gradient weights around phrase starts
                    for idx in phrase_start_indices:
                        start_context = max(0, idx - context_window)
                        for j in range(start_context, idx):
                            weight = weight_factor * (1 - (idx - j) / (context_window * 2)) # Adjusted gradient
                            seq_weights[j] = max(seq_weights[j], weight)

                        end_context = min(len(seq_labels), idx + context_window + 1)
                        for j in range(idx + 1, end_context):
                            weight = weight_factor * (1 - (j - idx) / (context_window * 2)) # Adjusted gradient
                            seq_weights[j] = max(seq_weights[j], weight)

                self.sequences.append((seq_features, seq_labels, seq_weights))

        print(f"Created {len(self.sequences)} sequences from {len(grouped)} melodies.")
        if not self.sequences:
            print("Warning: No sequences were created. Check data and parameters.")

        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
         if idx < 0 or idx >= len(self.sequences):
              print(f"Warning: Index {idx} out of bounds for {len(self.sequences)} sequences.")
              return None, None, None # Return None for invalid index
         return self.sequences[idx]


# --- Model Definition ---

class BiLSTM_Attention(nn.Module):
    """
    BiLSTM learning-jazz-improvisation-app with Attention mechanism for sequence classification.
    Replicates the structure of the local phrase_segmentation_model
    with separate processing for feature groups.
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, num_lstm_layers=2, dropout=0.3):
        super(BiLSTM_Attention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        # --- Feature group dimensions (must match feature engineering) ---
        self.temporal_dim = 5 
        self.melodic_dim = 2
        self.metric_dim = 2
        
        # Calculate basic_dim based on the actual input_dim received
        self.basic_dim = input_dim - self.temporal_dim - self.melodic_dim - self.metric_dim
        print(f"Initializing BiLSTM_Attention: input_dim={input_dim}")
        print(f"  Feature dims: temporal={self.temporal_dim}, melodic={self.melodic_dim}, metric={self.metric_dim}, basic={self.basic_dim}")
        
        if (self.temporal_dim + self.melodic_dim + self.metric_dim + self.basic_dim) != input_dim:
             raise ValueError(f"Sum of feature dimensions ({self.temporal_dim + self.melodic_dim + self.metric_dim + self.basic_dim}) does not match input_dim ({input_dim})")
        if self.basic_dim < 1:
            raise ValueError(f"Input dimension {input_dim} is too small for feature slicing. basic_dim must be >= 1, but got {self.basic_dim}.")
            
        fc_output_dim = hidden_dim // 4 # Output dim for each feature group Linear layer
            
        self.temporal_layer = nn.Linear(self.temporal_dim, fc_output_dim)
        self.melodic_layer = nn.Linear(self.melodic_dim, fc_output_dim)
        self.metric_layer = nn.Linear(self.metric_dim, fc_output_dim)
        self.basic_layer = nn.Linear(self.basic_dim, fc_output_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim) # Norm applied to concatenated features

        self.lstm = nn.LSTM(
            input_size=hidden_dim, # Input size is the concatenated features
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )

        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim) # Input from attended LSTM output
        self.fc2 = nn.Linear(hidden_dim, output_dim) # output_dim is 2 for (B, I) classes

        self._init_weights()
        
        
    def _init_weights(self):
        """Initialize weights for LSTM and Linear layers."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Initialize forget gate bias to 1 for LSTM
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    param.data[start:end].fill_(1.)
            # Initialize Linear layers
            elif any(layer_name in name for layer_name in ['fc', 'attention_layer', 'temporal_layer', 'melodic_layer', 'metric_layer', 'basic_layer']): 
                if 'weight' in name:
                    if param.data.dim() > 1: 
                        nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                     if param.data.dim() > 0: 
                          param.data.fill_(0)

    def forward(self, x, lengths=None):
        """
        Forward pass replicating local learning-jazz-improvisation-app: Slice -> Linear -> Concat -> Norm -> BiLSTM -> Attention -> FC.
        """
        batch_size, seq_len, current_input_dim = x.size()
        
        # Check if the actual input dimension matches initialization
        if current_input_dim != (self.temporal_dim + self.melodic_dim + self.metric_dim + self.basic_dim):
             raise RuntimeError(f"Model forward pass received input with dim {current_input_dim}, but learning-jazz-improvisation-app was initialized expecting {self.temporal_dim + self.melodic_dim + self.metric_dim + self.basic_dim}")

        # --- Feature Slicing --- 
        idx_end_temporal = self.temporal_dim
        idx_end_melodic = idx_end_temporal + self.melodic_dim
        idx_end_metric = idx_end_melodic + self.metric_dim
        
        temporal_features = x[:, :, :idx_end_temporal]
        melodic_features = x[:, :, idx_end_temporal:idx_end_melodic]
        metric_features = x[:, :, idx_end_melodic:idx_end_metric]
        basic_features = x[:, :, idx_end_metric:] # Takes the rest
        
        # --- Add shape validation before applying layers --- 
        if temporal_features.shape[-1] != self.temporal_layer.in_features:
            raise RuntimeError(f"Shape mismatch: temporal_features has {temporal_features.shape[-1]} features (expected {self.temporal_layer.in_features})")
        if melodic_features.shape[-1] != self.melodic_layer.in_features:
            raise RuntimeError(f"Shape mismatch: melodic_features has {melodic_features.shape[-1]} features (expected {self.melodic_layer.in_features})")
        if metric_features.shape[-1] != self.metric_layer.in_features:
            raise RuntimeError(f"Shape mismatch: metric_features has {metric_features.shape[-1]} features (expected {self.metric_layer.in_features})")
        if basic_features.shape[-1] != self.basic_layer.in_features:
            raise RuntimeError(f"Shape mismatch: basic_features has {basic_features.shape[-1]} features (expected {self.basic_layer.in_features})")
        # -------------------------------------------------
        
        # Process each group
        temporal_out = F.relu(self.temporal_layer(temporal_features))
        melodic_out = F.relu(self.melodic_layer(melodic_features))
        metric_out = F.relu(self.metric_layer(metric_features))
        basic_out = F.relu(self.basic_layer(basic_features))
        
        # Concatenate processed features and apply LayerNorm
        combined = torch.cat([temporal_out, melodic_out, metric_out, basic_out], dim=2)
        combined_norm = self.layer_norm(combined) # Shape: (batch, seq_len, hidden_dim)
        
        # --- LSTM Layer ---        
        lstm_input = combined_norm
        if lengths is not None:
             # Ensure lengths are on CPU and clamped
             lengths_cpu = lengths.cpu() 
             lengths_cpu = torch.clamp(lengths_cpu, min=1, max=seq_len) # Ensure lengths are at least 1
             packed_input = nn.utils.rnn.pack_padded_sequence(lstm_input, lengths_cpu, batch_first=True, enforce_sorted=False)
             packed_output, _ = self.lstm(packed_input)
             lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
        else:
             warnings.warn("Lengths are not provided to LSTM layer. Padding might affect results.")
             lstm_out, _ = self.lstm(lstm_input) # shape: (batch_size, seq_len, hidden_dim * 2)


        # --- Attention Mechanism --- 
        attention_scores = self.attention_layer(lstm_out) # shape: (batch_size, seq_len, 1)
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < lengths.to(x.device)[:, None]
            mask = mask.unsqueeze(-1) 
            fill_value = -torch.finfo(attention_scores.dtype).max
            # Ensure mask and scores are compatible before masking
            if mask.shape == attention_scores.shape:
                attention_scores = attention_scores.masked_fill(mask == 0, fill_value)
            else:
                 warnings.warn(f"Attention mask shape {mask.shape} mismatch with scores shape {attention_scores.shape}. Skipping masking.")
        attention_weights = F.softmax(attention_scores, dim=1) # shape: (batch_size, seq_len, 1)
        attended_lstm_out = lstm_out * attention_weights 


        # --- Output Layers ---
        dropped_out = self.dropout(attended_lstm_out)
        fc1_out = F.relu(self.fc1(dropped_out))
        fc1_dropped = self.dropout(fc1_out)
        output = self.fc2(fc1_dropped) # shape: (batch_size, seq_len, output_dim)
        
        # Return logits and optionally attention weights
        return output, attention_weights
    

    def loss(self, logits, target, weights, lengths):
        """
        Calculates the weighted Focal Loss for sequence labeling.
        """
        batch_size, seq_len, output_dim = logits.shape
        device = logits.device
        gamma = 2.0 # Focal loss gamma parameter

        logits_flat = logits.view(-1, output_dim)
        target_flat = target.view(-1)
        weights_flat = weights.view(-1)

        ce_loss = F.cross_entropy(logits_flat, target_flat, reduction='none')

        # Focal Loss component
        pt = torch.exp(-ce_loss) # Probability of the true class
        focal_loss = (1 - pt) ** gamma * ce_loss
        
        # Create mask based on sequence lengths to ignore padding
        mask = torch.arange(seq_len, device=device)[None, :].expand(batch_size, -1) < lengths.to(device)[:, None]
        mask_flat = mask.view(-1)

        # Apply weights and mask
        weighted_loss = focal_loss * weights_flat * mask_flat

        # Calculate mean loss only over non-padded elements
        total_valid_elements = mask_flat.sum().float().clamp(min=1e-6)
        final_loss = weighted_loss.sum() / total_valid_elements

        return final_loss

    def decode(self, logits, lengths=None):
        """
        Decodes learning-jazz-improvisation-app output logits into class predictions with smoothing.
        """
        probabilities = F.softmax(logits, dim=-1)
        raw_preds = torch.argmax(probabilities, dim=-1)
        raw_preds_np = raw_preds.cpu().numpy()

        if lengths is None:
             lengths = [logits.shape[1]] * logits.shape[0]
        else:
             lengths = lengths.cpu().numpy()


        final_predictions = []
        for i in range(logits.shape[0]):
            seq_len = lengths[i]
            seq_preds = raw_preds_np[i, :seq_len]

            # Apply smoothing using class 1 probabilities
            smoothed_preds = self._smooth_predictions(seq_preds, probabilities[i, :seq_len, 1].cpu().numpy())
            final_predictions.append(smoothed_preds)

        return final_predictions


    def _smooth_predictions(self, raw_preds, class1_probs, min_phrase_length=4):
        """
        Applies heuristic smoothing to phrase start predictions.
        (Adapted from phrase_segmentation_model)
        """
        n = len(raw_preds)
        if n == 0: return np.array([], dtype=int)

        smoothed_preds = np.zeros(n, dtype=int)
        phrase_starts = np.where(raw_preds == 1)[0]

        # If no starts detected by argmax, find peaks in probability
        if len(phrase_starts) == 0 and n > 0 :
             n_peaks = min(max(1, n // 10), 5) # Adaptive number of peaks
             peak_indices = np.argsort(-class1_probs)[:n_peaks]
             peak_indices = sorted(peak_indices)

             # Filter peaks too close to each other
             filtered_peaks = []
             last_peak = -min_phrase_length
             for idx in peak_indices:
                 if idx - last_peak >= min_phrase_length:
                     filtered_peaks.append(idx)
                     last_peak = idx
             phrase_starts = filtered_peaks


        # Ensure minimum distance between identified phrase starts
        valid_starts = []
        last_valid = -min_phrase_length
        for start_idx in phrase_starts:
            if start_idx - last_valid >= min_phrase_length:
                valid_starts.append(start_idx)
                last_valid = start_idx

        if valid_starts:
             smoothed_preds[np.array(valid_starts)] = 1

        return smoothed_preds


# --- Evaluation ---

def evaluate_epoch(model, dataloader, device, use_amp):
    """Helper function to evaluate learning-jazz-improvisation-app performance on a dataloader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_lengths = []
    valid_batches = 0

    with torch.no_grad():
        for batch_data in dataloader:
             if batch_data[0] is None: # Skip incomplete batches
                 continue

             data, target, weights, lengths = batch_data
             data = data.to(device, non_blocking=True)
             target = target.to(device, non_blocking=True)
             weights = weights.to(device, non_blocking=True)

             if use_amp:
                 with torch.autocast(device_type=device.type):
                     logits, _ = model(data, lengths)
                     loss = model.loss(logits, target, weights, lengths)
             else:
                 logits, _ = model(data, lengths)
                 loss = model.loss(logits, target, weights, lengths)

             if not torch.isnan(loss):
                 total_loss += loss.item()
                 valid_batches += 1
             else:
                print("Warning: NaN loss encountered during evaluation.")
                continue # Skip batch if loss is NaN

             preds = model.decode(logits, lengths) # Returns list of numpy arrays
             all_preds.extend(preds)
             # Store targets and lengths correctly
             for i in range(target.size(0)):
                 seq_len = lengths[i].item()
                 all_targets.append(target[i, :seq_len].cpu().numpy())
             all_lengths.extend(lengths.cpu().numpy())


    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')

    # Flatten predictions and targets, considering sequence lengths
    flat_preds = np.concatenate([p for p, l in zip(all_preds, all_lengths) if l > 0])
    flat_targets = np.concatenate([t for t, l in zip(all_targets, all_lengths) if l > 0])


    # Calculate metrics (Precision, Recall, F1 for class 1)
    true_positives = np.sum((flat_preds == 1) & (flat_targets == 1))
    false_positives = np.sum((flat_preds == 1) & (flat_targets == 0))
    false_negatives = np.sum((flat_preds == 0) & (flat_targets == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_combined_metric(val_metrics):
     """ Calculates a combined metric, prioritizing F1 score. """
     f1 = val_metrics['f1']
     # Optional: Add bonus for balanced precision/recall if F1 is low
     if f1 < 0.5:
         balance_factor = 1.0 - abs(val_metrics['precision'] - val_metrics['recall']) / 2.0
         f1 = f1 * (1.0 + 0.05 * balance_factor)
     return f1


# --- Training ---
def train_kaggle_model(
    X_train, y_train, X_val, y_val, # Expect pre-scaled data
    model_save_path="kaggle_model.pt",
    num_epochs=100, 
    batch_size=64,
    hidden_dim=256, 
    num_lstm_layers=2, 
    dropout=0.3, 
    learning_rate=0.0003, 
    weight_decay=0.01,
    patience=15,
    min_delta=0.001,
    sequence_length=128,
    stride=32
    ):
    """
    Trains the BiLSTM_Attention learning-jazz-improvisation-app using preprocessed data.
    
    Args:
        X_train, y_train: Pre-scaled training features (with melid) and labels.
        X_val, y_val: Pre-scaled validation features (with melid) and labels.
        model_save_path: Path to save the best learning-jazz-improvisation-app.
        ... other hyperparameters ...
        
    Returns:
        learning-jazz-improvisation-app: Trained learning-jazz-improvisation-app (best state).
        history: Dictionary containing training/validation metrics per epoch.
    """

    print("--- Starting Model Training ---")
    
    # Use the input data directly as it is preprocessed
    X_train_final = X_train
    X_val_final = X_val

    # --- Datasets and DataLoaders ---
    print("Creating Datasets...")
    train_dataset = JazzPhraseDataset(X_train_final, y_train, sequence_length, stride)
    val_dataset = JazzPhraseDataset(X_val_final, y_val, sequence_length, stride)


    if len(train_dataset) == 0:
         raise ValueError("Train dataset creation resulted in zero sequences.")
    if len(val_dataset) == 0:
         print("Warning: Validation dataset creation resulted in zero sequences. Evaluation might be skipped or inaccurate.")


    print("Creating DataLoaders...")
    num_workers = 2 # Adjust based on Kaggle environment capabilities
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False, drop_last=False
    )
    print(f"Train DataLoader length: {len(train_loader)} batches")
    if len(val_dataset) > 0:
        print(f"Validation DataLoader length: {len(val_loader)} batches")
        if len(val_loader) == 0:
            print("Warning: Validation DataLoader is empty despite non-empty dataset. Check batch size and drop_last=False.")
    else:
        print("Validation DataLoader not created as validation dataset is empty.")

    if len(train_loader) == 0:
        raise ValueError("Train DataLoader is empty. Check train dataset and batch size.")


    # --- Model Initialization ---
    # Calculate input_dim from the pre-scaled features (excluding melid)
    input_dim = X_train_final.shape[1] - 1 # Subtract 1 for the melid column
    output_dim = 2 # B (0), I (1) -> Background/Inside, Phrase Start
    print(f"Initializing BiLSTM_Attention learning-jazz-improvisation-app (Local Arch) with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, lstm_layers={num_lstm_layers}")
    model = BiLSTM_Attention(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim, 
        num_lstm_layers=num_lstm_layers, 
        dropout=dropout
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")


    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8
    )

    # Cyclical learning rate scheduler with warmup and cosine decay
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * 5 # Warmup for 5 epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress))) # Cosine decay to 0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    

    # --- Training Loop ---
    use_amp = torch.cuda.is_available() # Enable Automatic Mixed Precision if CUDA is available
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Automatic Mixed Precision (AMP) enabled: {use_amp}")

    best_metric = 0.0 # Using F1-based metric
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'lr': []}

    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0
        valid_train_batches = 0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data[0] is None: # Skip incomplete batches
                continue
                
            data, target, weights, lengths = batch_data
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                # Model expects input_dim = features - 1 (melid excluded)
                logits, _ = model(data, lengths)
                loss = model.loss(logits, target, weights, lengths)
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected during training batch {batch_idx}. Skipping update.")
                    continue # Skip optimizer step if loss is NaN

            # Scale loss, backward pass, unscale, clip, step optimizer
            amp_scaler.scale(loss).backward()
            
            # Optional: Gradient norm debugging
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            if batch_idx % 50 == 0: # Print every 50 batches
                 progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0], grad_norm=total_norm)
            else:
                 progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
            # Unscale gradients before clipping
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            amp_scaler.step(optimizer)
            amp_scaler.update()

            scheduler.step() # Update learning rate

            epoch_train_loss += loss.item()
            valid_train_batches += 1


        avg_train_loss = epoch_train_loss / valid_train_batches if valid_train_batches > 0 else float('inf')
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(scheduler.get_last_lr()[0])


        # --- Validation Phase ---
        if len(val_dataset) > 0 and len(val_loader) > 0:
            val_metrics = evaluate_epoch(model, val_loader, device, use_amp)
            history['val_loss'].append(val_metrics['loss'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])
            current_metric = calculate_combined_metric(val_metrics) # Use F1-based metric

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                  f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # --- Early Stopping and Model Saving ---
            if current_metric > best_metric + min_delta:
                best_metric = current_metric
                best_model_state = model.state_dict().copy() # Deep copy state dict
                patience_counter = 0
                print(f"  -> New best learning-jazz-improvisation-app found with metric: {best_metric:.4f}. Saving learning-jazz-improvisation-app...")
                torch.save(best_model_state, model_save_path)
                print(f"     Model saved to {model_save_path}")

            else:
                patience_counter += 1
                print(f"  -> No improvement for {patience_counter} epochs. Best metric: {best_metric:.4f}")
                # --- Early stopping break is disabled for Kaggle runs --- 
                # if patience_counter >= patience:
                #     print(f"EARLY STOPPING triggered after {patience} epochs without improvement.")
                #     break
        else: 
            # Handle case with no validation
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Validation skipped | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            patience_counter += 1 # Increment patience even if validation is skipped
            

    print("--- Training Finished ---")

    # Load the best learning-jazz-improvisation-app state if it was saved
    if best_model_state is not None:
        print(f"Loading best learning-jazz-improvisation-app state with validation metric: {best_metric:.4f}")
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best learning-jazz-improvisation-app state was saved. Returning the final learning-jazz-improvisation-app state.")


    # Return learning-jazz-improvisation-app and history
    return model, history 


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Running Kaggle Training Script...")

    # --- Configuration --- 
    # Define Kaggle paths
    KAGGLE_INPUT_DIR = "/kaggle/input/d/temsviridov/jazz-solo-dataset" 
    KAGGLE_WORKING_DIR = "/kaggle/working/"

    # Paths for cached data (input)
    DATASET_CACHE_PATH = os.path.join(KAGGLE_INPUT_DIR, "dataset_cache.npz")
    SCALER_CACHE_PATH = os.path.join(KAGGLE_INPUT_DIR, "scaler_cache.pkl")

    # Output paths in Kaggle's working directory
    MODEL_SAVE_PATH = os.path.join(KAGGLE_WORKING_DIR, "best_jazz_phrase_model.pt")
    HISTORY_SAVE_PATH = os.path.join(KAGGLE_WORKING_DIR, "training_history.pkl")

    # Hyperparameters
    NUM_EPOCHS = 200
    BATCH_SIZE = 64
    HIDDEN_DIM = 256 
    NUM_LSTM_LAYERS = 2 
    DROPOUT = 0.35 
    LEARNING_RATE = 0.00001 
    WEIGHT_DECAY = 0.01
    PATIENCE = 15 
    SEQUENCE_LENGTH = 128
    STRIDE = 32


    # --- Load Preprocessed Data --- 
    print(f"Loading preprocessed data from {DATASET_CACHE_PATH}...")
    try:
        data_cache = np.load(DATASET_CACHE_PATH)
        X_train = data_cache['X_train']
        y_train = data_cache['y_train']
        X_val = data_cache['X_val']
        y_val = data_cache['y_val']
        print(f"Loaded data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

        # Basic validation
        if X_train.shape[0] != y_train.shape[0] or X_val.shape[0] != y_val.shape[0]:
            raise ValueError("Loaded train/validation features and labels have mismatched first dimensions.")
        if X_train.shape[1] < 2:
             raise ValueError("Features array must have at least 2 columns (melid + 1 feature).")
             
    except FileNotFoundError:
        print(f"Error: Preprocessed data file not found at {DATASET_CACHE_PATH}.")
        print("Ensure prepare_kaggle_data.py was run and the dataset uploaded to Kaggle includes dataset_cache.npz.")
        exit() 
    except KeyError as e:
         print(f"Error: Missing key in {DATASET_CACHE_PATH}: {e}. Ensure the file was created correctly.")
         exit()
    except ValueError as e:
         print(f"Error loading or validating preprocessed data: {e}")
         exit()
    except Exception as e:
         print(f"An unexpected error occurred loading preprocessed data: {e}")
         exit()

    # --- Load Scaler ---
    print(f"Loading scaler from {SCALER_CACHE_PATH}...")
    try:
        with open(SCALER_CACHE_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {SCALER_CACHE_PATH}.")
        print("Ensure prepare_kaggle_data.py was run and the dataset uploaded to Kaggle includes scaler_cache.pkl.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the scaler: {e}")
        exit()
        

    # --- Train Model --- 
    try:
        # Pass the loaded, preprocessed data directly
        model, history = train_kaggle_model(
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_val,     
            y_val=y_val,     
            model_save_path=MODEL_SAVE_PATH,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_lstm_layers=NUM_LSTM_LAYERS, 
            dropout=DROPOUT,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE
        )

        # --- Save History --- 
        print(f"Saving training history to {HISTORY_SAVE_PATH}...")
        with open(HISTORY_SAVE_PATH, 'wb') as f:
            pickle.dump(history, f)

        print("Training script finished successfully.")

    except ValueError as e: # Catch specific errors like empty datasets
        print(f"\n--- A ValueError occurred during training setup ---")
        print(f"Error message: {e}")
        print("Please check dataset creation and data loading steps.")
        print("Training script finished with errors.")
    except Exception as e:
         print(f"\n--- An error occurred during training ---")
         import traceback
         print(traceback.format_exc())
         print(f"Error message: {e}")
         print("Training script finished with errors.")
          