import numpy as np
import pandas as pd
import sqlite3
import os
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress scikit-learn version warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

def extract_features_from_db(db_path, cache_dir="cache"):
    """
    Extracts features and labels from the database, creating or using a cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    features_cache_path = os.path.join(cache_dir, "features_cache.npz")
    
    if os.path.exists(features_cache_path):
        print(f"Loading cached features from {features_cache_path}")
        cache = np.load(features_cache_path)
        return cache['features'], cache['labels']
    
    print(f"No cache found. Extracting features from database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    notes_query = """
    SELECT 
        m1.eventid, m1.melid, m1.onset, m1.pitch, m1.duration, 
        m1.period, m1.bar, m1.beat, m1.loud_max,
        m1.f0_freq_hz,
        LAG(m1.onset) OVER (PARTITION BY m1.melid ORDER BY m1.onset) as prev_onset,
        LAG(m1.duration) OVER (PARTITION BY m1.melid ORDER BY m1.onset) as prev_duration,
        LAG(m1.pitch) OVER (PARTITION BY m1.melid ORDER BY m1.onset) as prev_pitch,
        LEAD(m1.onset) OVER (PARTITION BY m1.melid ORDER BY m1.onset) as next_onset,
        LEAD(m1.duration) OVER (PARTITION BY m1.melid ORDER BY m1.onset) as next_duration,
        CAST((m1.beat % 1.0) * 4 AS INTEGER) as beat_position,
        COUNT(*) OVER (PARTITION BY m1.melid, m1.bar) as notes_in_bar
    FROM melody m1
    ORDER BY m1.melid, m1.onset
    """
    notes_df = pd.read_sql_query(notes_query, conn)
    
    phrases_query = """
    SELECT melid, start, end, value
    FROM sections
    WHERE type = 'PHRASE'
    ORDER BY melid, start
    """
    phrases_df = pd.read_sql_query(phrases_query, conn)
    
    all_features = []
    all_labels = []
    
    for melid in notes_df['melid'].unique():
        mel_notes = notes_df[notes_df['melid'] == melid].sort_values(by='onset').reset_index(drop=True)
        mel_phrases = phrases_df[phrases_df['melid'] == melid].sort_values(by='start')
        
        if len(mel_notes) == 0 or len(mel_phrases) == 0:
            continue
        
        first_eventid = mel_notes.iloc[0]['eventid']
        first_onset = mel_notes.iloc[0]['onset']
        
        note_features = []
        for idx in range(len(mel_notes)):
            note = mel_notes.iloc[idx]
            prev_note = mel_notes.iloc[idx-1] if idx > 0 else note
            next_note = mel_notes.iloc[idx+1] if idx < len(mel_notes)-1 else note
            
            rest_before = note['onset'] - (prev_note['onset'] + prev_note['duration']) if idx > 0 else 0
            ioi_from_prev = note['onset'] - prev_note['onset'] if idx > 0 else 0
            
            rest_weight = np.tanh(rest_before * 2)  
            ioi_weight = np.tanh(ioi_from_prev * 1.5) 
            
            tempo_change = (ioi_from_prev / prev_note['duration']) if idx > 0 and prev_note['duration'] > 0 else 1.0
            rest_ratio = rest_before / ioi_from_prev if ioi_from_prev > 0 else 0
            
            melodic_interval = abs(note['pitch'] - prev_note['pitch']) if idx > 0 else 0
            melodic_direction = np.sign(note['pitch'] - prev_note['pitch']) if idx > 0 else 0
            
            rest_and_interval = rest_weight * np.log1p(1 + melodic_interval)
            
            beat_position = note['beat_position']
            is_strong_beat = 1.0 if beat_position == 0 else (0.5 if beat_position == 2 else 0.25)
            metric_weight = is_strong_beat * (1 + rest_weight) 
            
            rhythm_pattern = [
                1 if note['duration'] > prev_note['duration'] else 0,
                1 if note['duration'] > next_note['duration'] else 0,
                1 if ioi_from_prev > note['duration'] else 0
            ]
            rhythm_complexity = sum(rhythm_pattern) / 3
            
            relative_duration = note['duration'] / note['period'] if pd.notna(note['period']) and note['period'] > 0 else 0
            duration_ratio = np.log1p(relative_duration)  
            
            features = [
                note['melid'],
                note['pitch'] / 127.0, 
                duration_ratio, 
                (note['onset'] - first_onset) / 100.0, 
                note['loud_max'] if pd.notna(note['loud_max']) else 0,
                rest_weight * 2,
                ioi_weight * 2, 
                rest_and_interval * 1.5, 
                tempo_change,
                rest_ratio,
                melodic_interval / 12.0, 
                melodic_direction,
                metric_weight,
                rhythm_complexity,
                relative_duration
            ]
            note_features.append(features)
        
        labels = np.zeros(len(mel_notes), dtype=int)
        local_eventids = mel_notes['eventid'] - first_eventid
        
        for _, phrase in mel_phrases.iterrows():
            start_idx = mel_notes[local_eventids == phrase['start']].index.min()
            if pd.notna(start_idx):
                labels[start_idx] = 1
        
        all_features.extend(note_features)
        all_labels.extend(labels)
    
    conn.close()
    
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    print(f"Saving features to cache at {features_cache_path}")
    np.savez_compressed(features_cache_path, features=features_array, labels=labels_array)
    
    return features_array, labels_array

def prepare_datasets(db_path, cache_dir="cache", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prepares and scales train, validation, and test datasets from the database,
    ensuring stratified splitting of melodies based on phrase count.
    Uses caching for datasets and scaler.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dataset_cache_path = os.path.join(cache_dir, "dataset_cache.npz")
    scaler_cache_path = os.path.join(cache_dir, "scaler_cache.pkl")
    
    if os.path.exists(dataset_cache_path) and os.path.exists(scaler_cache_path):
        print(f"Loading cached datasets from {dataset_cache_path} and scaler from {scaler_cache_path}")
        datasets_cache = np.load(dataset_cache_path)
        X_train = datasets_cache['X_train']
        X_val = datasets_cache['X_val']
        X_test = datasets_cache['X_test']
        y_train = datasets_cache['y_train']
        y_val = datasets_cache['y_val']
        y_test = datasets_cache['y_test']
        
        with open(scaler_cache_path, 'rb') as f:
            scaler = pickle.load(f)
            
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler
    
    print("No cache found. Preparing datasets from database...")
    
    features, labels = extract_features_from_db(db_path, cache_dir)
    
    class_0_count = np.sum(labels == 0)
    class_1_count = np.sum(labels == 1)
    total_count = len(labels)
    print(f"Overall class distribution: Class 0={class_0_count} ({class_0_count/total_count:.4f}), Class 1={class_1_count} ({class_1_count/total_count:.4f})")
    
    unique_melids = np.unique(features[:, 0])
    
    # Stratify split based on phrase count per melody
    melid_to_phrases = {}
    phrase_count_bins = [0, 5, 10, 20, 50, 100, float('inf')] # Bins for grouping by phrase count
    melid_to_bin = {}
    
    for melid in unique_melids:
        mel_mask = features[:, 0] == melid
        mel_labels = labels[mel_mask]
        phrase_count = np.sum(mel_labels == 1)
        melid_to_phrases[melid] = phrase_count
        
        for i in range(len(phrase_count_bins) - 1):
            if phrase_count_bins[i] <= phrase_count < phrase_count_bins[i+1]:
                melid_to_bin[melid] = i
                break
    
    bin_to_melids = {i: [] for i in range(len(phrase_count_bins) - 1)}
    for melid, bin_idx in melid_to_bin.items():
         if bin_idx in bin_to_melids: # Ensure bin_idx is valid
             bin_to_melids[bin_idx].append(melid)
    
    # Split melodies within each bin to preserve proportions
    train_melids = []
    val_melids = []
    test_melids = []
    
    np.random.seed(42) # Fix seed for reproducibility
    for bin_idx, melids_in_bin in bin_to_melids.items():
        if not melids_in_bin:
            continue
            
        np.random.shuffle(melids_in_bin)
        
        n_bin = len(melids_in_bin)
        n_train = max(1, int(n_bin * train_ratio)) if n_bin > 2 else (1 if n_bin > 1 else n_bin)
        n_val = max(1, int(n_bin * val_ratio)) if n_bin - n_train > 1 else (1 if n_bin - n_train > 0 else 0)

        # Ensure at least one sample for test if possible and sizes allow
        n_test = n_bin - n_train - n_val
        if n_test == 0 and n_val > 1: # Borrow from validation if possible
             n_val -= 1
             n_test = 1
        elif n_test == 0 and n_train > 1: # Borrow from train if validation is already 1 or 0
             n_train -= 1
             n_test = 1

        train_melids.extend(melids_in_bin[:n_train])
        val_melids.extend(melids_in_bin[n_train:n_train+n_val])
        test_melids.extend(melids_in_bin[n_train+n_val:])
    
    # Shuffle the final lists
    np.random.shuffle(train_melids)
    np.random.shuffle(val_melids)
    np.random.shuffle(test_melids)
    
    print(f"Data split: {len(train_melids)} train, {len(val_melids)} validation, {len(test_melids)} test melodies")
    
    train_phrases = sum(melid_to_phrases.get(melid, 0) for melid in train_melids)
    val_phrases = sum(melid_to_phrases.get(melid, 0) for melid in val_melids)
    test_phrases = sum(melid_to_phrases.get(melid, 0) for melid in test_melids)
    print(f"Phrase starts distribution: {train_phrases} train, {val_phrases} validation, {test_phrases} test phrases")
    
    train_mask = np.isin(features[:, 0], train_melids)
    val_mask = np.isin(features[:, 0], val_melids)
    test_mask = np.isin(features[:, 0], test_melids)
    
    # Handle potential NaNs/Infs introduced during feature calculation
    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0) # Use clip values
    
    # Scale features (excluding the first column 'melid')
    scaler = StandardScaler()
    scaler.fit(features[train_mask, 1:])
    
    with open(scaler_cache_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    # Apply scaling and clipping
    X_train_scaled = np.clip(scaler.transform(features[train_mask, 1:]), -5, 5)
    X_val_scaled = np.clip(scaler.transform(features[val_mask, 1:]), -5, 5)
    X_test_scaled = np.clip(scaler.transform(features[test_mask, 1:]), -5, 5)
    
    # Re-attach melid column
    X_train = np.column_stack([features[train_mask, 0], X_train_scaled])
    X_val = np.column_stack([features[val_mask, 0], X_val_scaled])
    X_test = np.column_stack([features[test_mask, 0], X_test_scaled])
    
    y_train = labels[train_mask]
    y_val = labels[val_mask]
    y_test = labels[test_mask]
    
    print(f"Saving datasets to cache at {dataset_cache_path}")
    np.savez_compressed(dataset_cache_path, 
                        X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def main():
    """Main function to prepare data for Kaggle training."""
    db_path = r"C:\polytech\Diploma\wjazzd.db" 
    cache_dir = "cache" 
    
    os.makedirs(cache_dir, exist_ok=True)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_datasets(
        db_path=db_path,
        cache_dir=cache_dir
    )
    
    print(f"Data preparation completed successfully.")
    print(f"Training set: {X_train.shape[0]} samples, {np.sum(y_train == 1)} phrase starts")
    print(f"Validation set: {X_val.shape[0]} samples, {np.sum(y_val == 1)} phrase starts")
    print(f"Test set: {X_test.shape[0]} samples, {np.sum(y_test == 1)} phrase starts")
    
    # Save class weight info based on training set for potential use during training
    class_0_train = np.sum(y_train == 0)
    class_1_train = np.sum(y_train == 1)
    # Example weight calculation (adjust logic as needed for actual training script)
    # This weight factor aims to give more importance to the minority class (phrase starts)
    class_weight_factor = min((class_0_train / class_1_train) if class_1_train > 0 else 1.0, 10.0) 
    
    weight_info = {'class_0_count': class_0_train, 
                   'class_1_count': class_1_train, 
                   'weight_factor_for_class1': class_weight_factor}
                   
    weight_info_path = os.path.join(cache_dir, "class_weight_info.pkl")
    with open(weight_info_path, 'wb') as f:
        pickle.dump(weight_info, f)
    
    print(f"Class weight info saved to {weight_info_path}: {weight_info}")
    print(f"All data prepared and cached in {cache_dir}")

if __name__ == "__main__":
    main()
    