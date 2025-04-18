import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import tqdm
import warnings

# Import necessary components from the training script
from core.ml.kaggle.train_kaggle_model import BiLSTM_Attention, JazzPhraseDataset, collate_fn

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning) # Added for potential future issues

# --- Paths ---
# Assumes the necessary files are downloaded from Kaggle into this directory
DATA_DIR = "cache" 
DATASET_CACHE_PATH = os.path.join(DATA_DIR, "dataset_cache.npz")
SCALER_CACHE_PATH = os.path.join(DATA_DIR, "scaler_cache.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "best_jazz_phrase_model.pt")

# --- Hyperparameters (Must match the trained learning-jazz-improvisation-app!) ---
HIDDEN_DIM = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.35 # Use the value from the last training run
# sequence_length and stride are needed for JazzPhraseDataset
SEQUENCE_LENGTH = 128
STRIDE = 32
BATCH_SIZE = 64 # Can be adjusted for evaluation

# --- Data Handling (Copied from train_kaggle_model.py) ---

# Define the collate_fn globally to make it picklable
def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences, padding, and weights.
    """
    try:
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

    lengths = [len(seq) for seq in sequences]
    if not lengths or max(lengths) == 0:
         print(f"Warning: Batch has sequences of zero length. Lengths: {lengths}")
         return None, None, None, None # Cannot proceed

    max_len = max(lengths)
    # Get feature dimension from the first valid sequence
    if sequences and sequences[0] is not None and sequences[0].dim() > 1:
        feature_dim = sequences[0].size(-1)
    else:
        print("Warning: Could not determine feature dimension from the first sequence.")
        return None, None, None, None 

    # Pad sequences
    padded_seqs = torch.zeros(len(sequences), max_len, feature_dim, dtype=torch.float32)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long) 
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
    (Adapted from train_kaggle_model.py)
    """
    def __init__(self, features, labels, sequence_length=128, stride=32):
        self.features_with_melid = features # Expects features with melid in the first column
        self.labels = labels
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = []

        if self.features_with_melid is None or len(self.features_with_melid) == 0:
             print("Warning: Features array is None or empty.")
             return
        if self.labels is None or len(self.labels) != len(self.features_with_melid):
             print("Warning: Labels array is None or size mismatch with features.")
             return

        # Calculate Global Weight Factor (using the same logic as training for consistency)
        global_class_0 = np.sum(self.labels == 0)
        global_class_1 = np.sum(self.labels == 1)
        global_ratio = global_class_0 / max(global_class_1, 1)
        weight_factor = min(global_ratio, 10.0) # Use the same cap as in training
        print(f"Dataset Class Ratio (0/1): {global_ratio:.2f}, using weight factor for sequence weights: {weight_factor:.2f}")

        # Group by Melody ID
        try:
             num_features = self.features_with_melid.shape[1] - 1 # Exclude melid
             feature_cols = [f'feat_{i}' for i in range(num_features)]
             df = pd.DataFrame(self.features_with_melid, columns=['melid'] + feature_cols)
             df['label'] = self.labels
             df['original_index'] = np.arange(len(df)) # Keep track of original indices
             grouped = df.groupby('melid')
        except Exception as e:
            print(f"Error grouping features by melid: {e}")
            grouped = {}


        # Create Sequences per Melody
        context_window = 2 # For weighting context around phrase starts

        for melid, group in grouped:
            group = group.sort_values(by='original_index')
            # Extract features EXCLUDING melid 
            mel_features_np = group.iloc[:, 1:-2].values.astype(np.float32) # Features excluding melid, label, index
            mel_labels_np = group['label'].values.astype(np.int64)

            if len(mel_features_np) == 0:
                continue

            # Generate Overlapping Sequences
            for start_idx in range(0, len(mel_features_np), self.stride):
                end_idx = min(start_idx + self.sequence_length, len(mel_features_np))

                if end_idx - start_idx < 10: # Skip sequences that are too short (consistent with training)
                    continue

                seq_features = torch.from_numpy(mel_features_np[start_idx:end_idx])
                seq_labels = torch.from_numpy(mel_labels_np[start_idx:end_idx])

                # Calculate Weights for the Sequence (consistent with training)
                seq_weights = torch.ones(seq_labels.size(0), dtype=torch.float32)
                phrase_start_indices = torch.nonzero(seq_labels == 1).squeeze(1)

                if len(phrase_start_indices) > 0:
                    seq_weights[phrase_start_indices] = weight_factor
                    for idx in phrase_start_indices:
                        start_context = max(0, idx - context_window)
                        for j in range(start_context, idx):
                            weight = weight_factor * (1 - (idx - j) / (context_window * 2))
                            seq_weights[j] = max(seq_weights[j], weight)
                        end_context = min(len(seq_labels), idx + context_window + 1)
                        for j in range(idx + 1, end_context):
                            weight = weight_factor * (1 - (j - idx) / (context_window * 2))
                            seq_weights[j] = max(seq_weights[j], weight)

                # Add the processed sequence (features *without* melid, labels, weights)
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
         # Return features (without melid), labels, and weights
         return self.sequences[idx]

# --- Model Definition (Copied from train_kaggle_model.py) ---

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

        # Define feature group dimensions explicitly
        self.temporal_dim = 5
        self.melodic_dim = 2
        self.metric_dim = 2
        self.basic_dim = input_dim - self.temporal_dim - self.melodic_dim - self.metric_dim
        
        print(f"Initializing BiLSTM_Attention: input_dim={input_dim}") 
        print(f"  Calculated feature dims: temporal={self.temporal_dim}, melodic={self.melodic_dim}, metric={self.metric_dim}, basic={self.basic_dim}")

        if (self.temporal_dim + self.melodic_dim + self.metric_dim + self.basic_dim) != input_dim:
             raise ValueError(f"Sum of feature dimensions ({self.temporal_dim + self.melodic_dim + self.metric_dim + self.basic_dim}) does not match input_dim ({input_dim})")
        if self.basic_dim < 1:
            raise ValueError(f"Input dimension {input_dim} is too small for feature slicing. basic_dim must be >= 1, but got {self.basic_dim}.")

        fc_output_dim = hidden_dim // 4

        self.temporal_layer = nn.Linear(self.temporal_dim, fc_output_dim)
        self.melodic_layer = nn.Linear(self.melodic_dim, fc_output_dim)
        self.metric_layer = nn.Linear(self.metric_dim, fc_output_dim)
        self.basic_layer = nn.Linear(self.basic_dim, fc_output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # No need to initialize weights here, they will be loaded

    def forward(self, x, lengths=None):
        batch_size, seq_len, current_input_dim = x.size()

        if current_input_dim != (self.temporal_dim + self.melodic_dim + self.metric_dim + self.basic_dim):
             raise RuntimeError(f"Model forward pass received input with dim {current_input_dim}, but model was initialized expecting {self.temporal_dim + self.melodic_dim + self.metric_dim + self.basic_dim}")

        idx_end_temporal = self.temporal_dim
        idx_end_melodic = idx_end_temporal + self.melodic_dim
        idx_end_metric = idx_end_melodic + self.metric_dim

        temporal_features = x[:, :, :idx_end_temporal]
        melodic_features = x[:, :, idx_end_temporal:idx_end_melodic]
        metric_features = x[:, :, idx_end_melodic:idx_end_metric]
        basic_features = x[:, :, idx_end_metric:]

        # Shape validation (important sanity check)
        if temporal_features.shape[-1] != self.temporal_layer.in_features: raise RuntimeError(f"Shape mismatch: temporal_features has {temporal_features.shape[-1]} features (expected {self.temporal_layer.in_features})")
        if melodic_features.shape[-1] != self.melodic_layer.in_features: raise RuntimeError(f"Shape mismatch: melodic_features has {melodic_features.shape[-1]} features (expected {self.melodic_layer.in_features})")
        if metric_features.shape[-1] != self.metric_layer.in_features: raise RuntimeError(f"Shape mismatch: metric_features has {metric_features.shape[-1]} features (expected {self.metric_layer.in_features})")
        if basic_features.shape[-1] != self.basic_layer.in_features: raise RuntimeError(f"Shape mismatch: basic_features has {basic_features.shape[-1]} features (expected {self.basic_layer.in_features})")

        temporal_out = F.relu(self.temporal_layer(temporal_features))
        melodic_out = F.relu(self.melodic_layer(melodic_features))
        metric_out = F.relu(self.metric_layer(metric_features))
        basic_out = F.relu(self.basic_layer(basic_features))

        combined = torch.cat([temporal_out, melodic_out, metric_out, basic_out], dim=2)
        combined_norm = self.layer_norm(combined)

        lstm_input = combined_norm
        if lengths is not None:
             lengths_cpu = lengths.cpu()
             lengths_cpu = torch.clamp(lengths_cpu, min=1, max=seq_len)
             try:
                 packed_input = nn.utils.rnn.pack_padded_sequence(lstm_input, lengths_cpu, batch_first=True, enforce_sorted=False)
                 packed_output, _ = self.lstm(packed_input)
                 lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
             except RuntimeError as e:
                 print(f"Error during packing/unpacking: {e}")
                 # Fallback to non-packed if error occurs
                 lstm_out, _ = self.lstm(lstm_input) 
                 warnings.warn("LSTM fallback to non-packed sequence due to packing error.")
        else:
             warnings.warn("Lengths are not provided to LSTM layer. Padding might affect results.")
             lstm_out, _ = self.lstm(lstm_input)

        attention_scores = self.attention_layer(lstm_out)
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < lengths.to(x.device)[:, None]
            mask = mask.unsqueeze(-1)
            fill_value = -torch.finfo(attention_scores.dtype).max
            if mask.shape == attention_scores.shape:
                attention_scores = attention_scores.masked_fill(mask == 0, fill_value)
            else:
                 warnings.warn(f"Attention mask shape {mask.shape} mismatch with scores shape {attention_scores.shape}. Skipping masking.")
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_lstm_out = lstm_out * attention_weights

        dropped_out = self.dropout(attended_lstm_out)
        fc1_out = F.relu(self.fc1(dropped_out))
        fc1_dropped = self.dropout(fc1_out)
        output = self.fc2(fc1_dropped)

        return output, attention_weights

    # Loss function might not be strictly needed for evaluation script, but useful for comparison
    def loss(self, logits, target, weights, lengths):
        """Calculates the weighted Focal Loss."""
        batch_size, seq_len, output_dim = logits.shape
        device = logits.device
        gamma = 2.0 # Focal loss gamma parameter

        logits_flat = logits.view(-1, output_dim)
        target_flat = target.view(-1)
        weights_flat = weights.view(-1)

        ce_loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss

        mask = torch.arange(seq_len, device=device)[None, :].expand(batch_size, -1) < lengths.to(device)[:, None]
        mask_flat = mask.view(-1)

        weighted_loss = focal_loss * weights_flat * mask_flat
        total_valid_elements = mask_flat.sum().float().clamp(min=1e-6)
        final_loss = weighted_loss.sum() / total_valid_elements
        return final_loss

    def decode(self, logits, lengths=None):
        """Decodes model output logits into class predictions with smoothing."""
        probabilities = F.softmax(logits, dim=-1)
        raw_preds = torch.argmax(probabilities, dim=-1)
        raw_preds_np = raw_preds.cpu().numpy()

        if lengths is None:
             lengths = [logits.shape[1]] * logits.shape[0]
        else:
             if isinstance(lengths, torch.Tensor):
                 lengths = lengths.cpu().numpy()

        final_predictions = []
        for i in range(logits.shape[0]):
            seq_len = lengths[i]
            seq_preds = raw_preds_np[i, :seq_len]
            
            if probabilities.shape[2] > 1: # Ensure class 1 probabilities exist
                 class1_probs = probabilities[i, :seq_len, 1].cpu().numpy()
            else: 
                 class1_probs = np.zeros_like(seq_preds, dtype=float) 
                 
            smoothed_preds = self._smooth_predictions(seq_preds, class1_probs)
            final_predictions.append(smoothed_preds)
        return final_predictions

    def _smooth_predictions(self, raw_preds, class1_probs, min_phrase_length=4, prob_threshold=0.5):
        """
        Applies heuristic smoothing to phrase start predictions.
        Includes a probability threshold to filter low-confidence predictions.
        Args:
            prob_threshold: Minimum probability for a note to be considered a potential start
        """
        n = len(raw_preds)
        if n == 0: return np.array([], dtype=int)

        smoothed_preds = np.zeros(n, dtype=int)

        # 1. Find potential starts based on raw predictions and probability threshold
        potential_starts_raw = np.where((raw_preds == 1) & (class1_probs >= prob_threshold))[0]

        # 2. Find potential starts based on probability peaks (if few raw starts above threshold)
        potential_starts_peak = []
        if len(potential_starts_raw) < max(1, n // 50): 
            n_peaks = min(max(1, n // 10), 5) # Adaptive number of peaks
            candidate_peak_indices = np.argsort(-class1_probs)
            # Consider only peaks with probability above threshold
            peak_indices_above_threshold = [idx for idx in candidate_peak_indices if class1_probs[idx] >= prob_threshold]
            peak_indices = sorted(peak_indices_above_threshold[:n_peaks])

            # Filter peaks too close to each other
            last_peak = -min_phrase_length
            for idx in peak_indices:
                 if idx - last_peak >= min_phrase_length:
                     potential_starts_peak.append(idx)
                     last_peak = idx

        # 3. Combine and filter potential starts
        combined_starts = sorted(list(set(potential_starts_raw) | set(potential_starts_peak)))

        # 4. Ensure minimum distance between final identified phrase starts
        valid_starts = []
        last_valid = -min_phrase_length
        for start_idx in combined_starts: 
            # Final check ensures probability threshold and minimum distance
            if class1_probs[start_idx] >= prob_threshold and start_idx - last_valid >= min_phrase_length:
                valid_starts.append(start_idx)
                last_valid = start_idx

        if valid_starts:
             smoothed_preds[np.array(valid_starts)] = 1
        return smoothed_preds

def evaluate_model(model, dataloader, device):
    """Evaluates model performance on a dataloader."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    all_preds = []
    all_targets = []
    valid_batches = 0
    use_amp = torch.cuda.is_available() # Use AMP if available

    with torch.no_grad(): # Disable gradient calculations
        progress_bar = tqdm.tqdm(dataloader, desc="Evaluating", leave=False)
        for batch_data in progress_bar:
             if batch_data[0] is None: continue # Skip incomplete batches

             data, target, weights, lengths = batch_data
             data = data.to(device, non_blocking=True)
             target = target.to(device, non_blocking=True)
             weights = weights.to(device, non_blocking=True) # Needed for loss calc

             # Forward pass
             if use_amp:
                 with torch.autocast(device_type=device.type):
                     logits, _ = model(data, lengths) 
                     # Calculate loss using the method from the imported model class
                     loss = model.loss(logits, target, weights, lengths) 
             else:
                 logits, _ = model(data, lengths)
                 loss = model.loss(logits, target, weights, lengths)

             if not torch.isnan(loss):
                 total_loss += loss.item()
                 valid_batches += 1
             else:
                print("Warning: NaN loss encountered during evaluation.")
                continue

             # Decode predictions using the method from the imported model class
             preds = model.decode(logits, lengths) # Returns list of numpy arrays
             all_preds.extend(preds)

             # Store targets correctly based on lengths
             for i in range(target.size(0)):
                 seq_len = lengths[i].item()
                 all_targets.append(target[i, :seq_len].cpu().numpy())

             progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')

    # Flatten predictions and targets
    if not all_preds or not all_targets:
        print("Warning: No predictions or targets collected during evaluation.")
        return {'loss': avg_loss, 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
        
    flat_preds = np.concatenate(all_preds)
    flat_targets = np.concatenate(all_targets)
    
    # Ensure same length after concatenation (handle potential edge cases)
    if len(flat_preds) != len(flat_targets):
        print(f"Warning: Mismatch between flattened predictions ({len(flat_preds)}) and targets ({len(flat_targets)}). Metrics might be inaccurate.")
        min_len = min(len(flat_preds), len(flat_targets))
        flat_preds = flat_preds[:min_len]
        flat_targets = flat_targets[:min_len]


    # Calculate metrics (Precision, Recall, F1 for class 1)
    true_positives = np.sum((flat_preds == 1) & (flat_targets == 1))
    false_positives = np.sum((flat_preds == 1) & (flat_targets == 0))
    false_negatives = np.sum((flat_preds == 0) & (flat_targets == 1))
    true_negatives = np.sum((flat_preds == 0) & (flat_targets == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(flat_targets) if len(flat_targets) > 0 else 0

    # --- Print Detailed Metrics ---
    print("\n--- Evaluation Metrics ---")
    print(f" Test Loss: {avg_loss:.4f}")
    print(f" Accuracy:  {accuracy:.4f}")
    print(" --- Metrics for Class 1 (Phrase Start) ---")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1-Score:  {f1:.4f}")
    print(f" True Positives:  {true_positives}")
    print(f" False Positives: {false_positives}")
    print(f" False Negatives: {false_negatives}")
    print(f" True Negatives:  {true_negatives}")
    print("--------------------------")


    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# --- Function for Detailed Melody Evaluation ---
def evaluate_melody(model, melody_features, melody_labels, device):
    """Evaluates performance for a single melody."""
    model.eval()
    n_notes = len(melody_labels)
    if n_notes == 0:
        return 0, 0, 0, 0, 0 # TP, FP, FN, Total Predicted, Total Actual

    # Features should already exclude melid
    data = torch.from_numpy(melody_features).unsqueeze(0).float().to(device)
    lengths = torch.LongTensor([n_notes]).to('cpu') 

    with torch.no_grad():
        use_amp = torch.cuda.is_available()
        if use_amp:
            with torch.autocast(device_type=device.type):
                 logits, _ = model(data, lengths)
        else:
            logits, _ = model(data, lengths)

        # Decode using the model's smoothing logic directly
        # The decode method itself calls _smooth_predictions internally
        # We expect decode to return a list containing one numpy array for the single melody
        predicted_starts_list = model.decode(logits, lengths) 
        
        if not predicted_starts_list: # Handle cases where decode might return empty
             print(f"Warning: Model decode returned empty list for melody.")
             predicted_starts = np.zeros(n_notes, dtype=int)
        else:
             predicted_starts = predicted_starts_list[0]


    # Calculate metrics for this melody
    actual_starts = melody_labels
    # predicted_starts is already calculated above using model.decode

    tp = np.sum((predicted_starts == 1) & (actual_starts == 1))
    fp = np.sum((predicted_starts == 1) & (actual_starts == 0))
    fn = np.sum((predicted_starts == 0) & (actual_starts == 1))
    
    total_predicted = np.sum(predicted_starts == 1)
    total_actual = np.sum(actual_starts == 1)

    return tp, fp, fn, total_predicted, total_actual


# --- Main Evaluation Script ---
if __name__ == "__main__":
    print("--- Starting Model Evaluation ---")

    # --- Load Data ---
    print(f"Loading test data from {DATASET_CACHE_PATH}...")
    try:
        data_cache = np.load(DATASET_CACHE_PATH)
        X_test = data_cache['X_test']
        y_test = data_cache['y_test']
        print(f"Loaded test data shapes: X_test={X_test.shape}, y_test={y_test.shape}")

        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("Test features and labels have mismatched first dimensions.")
        if X_test.shape[1] < 2:
             raise ValueError("Features array must have at least 2 columns (melid + 1 feature).")

    except FileNotFoundError:
        print(f"Error: Test data file not found at {DATASET_CACHE_PATH}.")
        exit()
    except KeyError as e:
         print(f"Error: Missing 'X_test' or 'y_test' key in {DATASET_CACHE_PATH}: {e}.")
         exit()
    except ValueError as e:
         print(f"Error loading or validating test data: {e}")
         exit()
    except Exception as e:
         print(f"An unexpected error occurred loading test data: {e}")
         exit()
         
    # --- Prepare Dataset and DataLoader (Using imported classes) ---
    print("Creating Test Dataset...")
    # Use the imported JazzPhraseDataset
    test_dataset = JazzPhraseDataset(X_test, y_test, SEQUENCE_LENGTH, STRIDE) 

    if len(test_dataset) == 0:
        print("Error: Test dataset creation resulted in zero sequences. Cannot evaluate.")
        exit()

    print("Creating Test DataLoader...")
    # Use the imported collate_fn
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, 
        num_workers=2, pin_memory=torch.cuda.is_available(), drop_last=False 
    )
    print(f"Test DataLoader length: {len(test_loader)} batches")
    if len(test_loader) == 0:
        print("Error: Test DataLoader is empty. Check dataset and batch size.")
        exit()

    # --- Load Model (Using imported class) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate input_dim from the loaded test data (features without melid)
    input_dim = X_test.shape[1] - 1 
    output_dim = 2 # B, I

    print(f"Loading model architecture with input_dim={input_dim}, hidden_dim={HIDDEN_DIM}...")
    # Instantiate the imported BiLSTM_Attention class
    model = BiLSTM_Attention(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT # Use the consistent dropout value
    )

    print(f"Loading model weights from {MODEL_PATH}...")
    try:
        # Load state dict; ensure map_location handles CPU/GPU moves
        state_dict = torch.load(MODEL_PATH, map_location=device) 
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the model state_dict: {e}")
        # This could be due to architecture mismatch if hyperparameters are wrong
        print("Check if hyperparameters (HIDDEN_DIM, NUM_LSTM_LAYERS, DROPOUT etc.) match the saved model.")
        exit()

    model.to(device)

    # --- Detailed Melody Evaluation (First 10 Melodies) ---
    print("\n--- Detailed Evaluation for First 10 Test Melodies ---")
    unique_test_melids = np.unique(X_test[:, 0])
    melids_to_evaluate = unique_test_melids[:10]

    for melid in melids_to_evaluate:
        print(f"\nEvaluating Melody ID: {int(melid)}")
        melody_mask = (X_test[:, 0] == melid)
        X_melody = X_test[melody_mask, 1:] # Features without melid
        y_melody = y_test[melody_mask]    
        
        if len(y_melody) < 10: # Skip very short melodies
             print(f"  Skipping melody (too short: {len(y_melody)} notes)")
             continue
             
        # evaluate_melody now uses the imported model's methods
        tp, fp, fn, total_pred, total_act = evaluate_melody(model, X_melody, y_melody, device)
        
        print(f"  Actual Phrase Starts: {total_act}")
        print(f"  Predicted Phrase Starts: {total_pred}")
        print(f"    Correctly Identified (TP): {tp}")
        print(f"    Incorrectly Identified (FP): {fp}")
        print(f"    Missed Phrases (FN): {fn}")
        
        mel_precision = tp / total_pred if total_pred > 0 else 0
        mel_recall = tp / total_act if total_act > 0 else 0
        print(f"    Melody Precision: {mel_precision:.3f}")
        print(f"    Melody Recall: {mel_recall:.3f}")

    # --- Overall Evaluation (on entire test set) ---
    print("\n--- Overall Test Set Evaluation --- ")
    print("Starting evaluation...")
    test_metrics = evaluate_model(model, test_loader, device)

    print("\n--- Final Test Results --- ")
    for key, value in test_metrics.items():
        print(f" {key.capitalize()}: {value:.4f}")
    print("--------------------------")
    print("Evaluation finished.")
