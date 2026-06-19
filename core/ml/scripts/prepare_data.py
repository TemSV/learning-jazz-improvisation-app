import numpy as np
import pandas as pd
import sqlite3
import os
import pickle
from sklearn.preprocessing import StandardScaler
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")


def extract_features_from_db(db_path, cache_dir="cache"):
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
        m1.period, m1.loud_max,
        LAG(m1.pitch) OVER (PARTITION BY m1.melid ORDER BY m1.onset) as prev_pitch,
        LAG(m1.duration) OVER (PARTITION BY m1.melid ORDER BY m1.onset) as prev_duration,
        LEAD(m1.duration) OVER (PARTITION BY m1.melid ORDER BY m1.onset) as next_duration
    FROM melody m1
    ORDER BY m1.melid, m1.onset
    """
    notes_df = pd.read_sql_query(notes_query, conn)

    phrases_query = """
    SELECT melid, start, end
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
            prev_note = mel_notes.iloc[idx - 1] if idx > 0 else note
            next_note = mel_notes.iloc[idx + 1] if idx < len(mel_notes) - 1 else note

            melodic_interval = abs(note['pitch'] - prev_note['pitch']) if idx > 0 else 0
            melodic_direction = np.sign(note['pitch'] - prev_note['pitch']) if idx > 0 else 0

            rhythm_pattern = [
                1 if note['duration'] > prev_note['duration'] else 0,
                1 if note['duration'] > next_note['duration'] else 0
            ]
            rhythm_complexity = sum(rhythm_pattern) / 2.0

            relative_duration = note['duration'] / note['period'] if pd.notna(note['period']) and note[
                'period'] > 0 else 0
            duration_ratio = np.log1p(relative_duration)

            features = [
                note['melid'],
                int(note['pitch']),

                duration_ratio,
                (note['onset'] - first_onset) / 100.0,
                note['loud_max'] if pd.notna(note['loud_max']) else 0,
                melodic_interval / 12.0,
                melodic_direction,
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

    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    print(f"Saving features to cache at {features_cache_path}")
    np.savez_compressed(features_cache_path, features=features_array, labels=labels_array)

    return features_array, labels_array


def prepare_datasets(db_path, cache_dir="cache", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    os.makedirs(cache_dir, exist_ok=True)
    dataset_cache_path = os.path.join(cache_dir, "dataset_cache.npz")
    scaler_cache_path = os.path.join(cache_dir, "scaler_cache.pkl")

    if os.path.exists(dataset_cache_path) and os.path.exists(scaler_cache_path):
        print(f"Loading cached datasets from {dataset_cache_path} and scaler from {scaler_cache_path}")
        datasets_cache = np.load(dataset_cache_path)
        with open(scaler_cache_path, 'rb') as f:
            scaler = pickle.load(f)
        return datasets_cache['X_train'], datasets_cache['X_val'], datasets_cache['X_test'], \
            datasets_cache['y_train'], datasets_cache['y_val'], datasets_cache['y_test'], scaler

    print("No cache found. Preparing datasets from database...")
    features, labels = extract_features_from_db(db_path, cache_dir)

    unique_melids = np.unique(features[:, 0])
    melid_to_phrases = {melid: np.sum(labels[features[:, 0] == melid] == 1) for melid in unique_melids}

    phrase_count_bins = [0, 5, 10, 20, 50, 100, float('inf')]
    bin_to_melids = {i: [] for i in range(len(phrase_count_bins) - 1)}

    for melid, count in melid_to_phrases.items():
        for i in range(len(phrase_count_bins) - 1):
            if phrase_count_bins[i] <= count < phrase_count_bins[i + 1]:
                bin_to_melids[i].append(melid)
                break

    train_melids, val_melids, test_melids = [], [], []
    np.random.seed(42)

    for bin_idx, melids_in_bin in bin_to_melids.items():
        if not melids_in_bin: continue
        np.random.shuffle(melids_in_bin)
        n_bin = len(melids_in_bin)
        n_train = max(1, int(n_bin * train_ratio)) if n_bin > 2 else (1 if n_bin > 1 else n_bin)
        n_val = max(1, int(n_bin * val_ratio)) if n_bin - n_train > 1 else (1 if n_bin - n_train > 0 else 0)

        train_melids.extend(melids_in_bin[:n_train])
        val_melids.extend(melids_in_bin[n_train:n_train + n_val])
        test_melids.extend(melids_in_bin[n_train + n_val:])

    train_mask = np.isin(features[:, 0], train_melids)
    val_mask = np.isin(features[:, 0], val_melids)
    test_mask = np.isin(features[:, 0], test_melids)

    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)

    continuous_cols = list(range(2, features.shape[1]))

    scaler = StandardScaler()
    scaler.fit(features[train_mask][:, continuous_cols])

    with open(scaler_cache_path, 'wb') as f:
        pickle.dump(scaler, f)

    def scale_subset(mask):
        subset = features[mask].copy()
        scaled_cont = np.clip(scaler.transform(subset[:, continuous_cols]), -5, 5)
        subset[:, continuous_cols] = scaled_cont
        return subset

    X_train = scale_subset(train_mask)
    X_val = scale_subset(val_mask)
    X_test = scale_subset(test_mask)

    y_train = labels[train_mask]
    y_val = labels[val_mask]
    y_test = labels[test_mask]

    print(f"Saving datasets to cache at {dataset_cache_path}")
    np.savez_compressed(dataset_cache_path,
                        X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def main():
    db_path = r"C:\polytech\Diploma\wjazzd.db"
    cache_dir = "../../../data/cache"

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_datasets(
        db_path=db_path,
        cache_dir=cache_dir
    )

    print(f"Data preparation completed successfully.")
    print(f"Training set: {X_train.shape[0]} samples, {np.sum(y_train == 1)} phrase starts")
    print(f"Validation set: {X_val.shape[0]} samples, {np.sum(y_val == 1)} phrase starts")
    print(f"Test set: {X_test.shape[0]} samples, {np.sum(y_test == 1)} phrase starts")


if __name__ == "__main__":
    main()