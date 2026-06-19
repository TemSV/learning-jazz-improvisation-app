Читать на русском языке: [README_ru.md](https://github.com/TemSV/learning-jazz-improvisation-app/blob/master/README_ru.md)

# Jazz Phrase Segmentation AI

A mobile application for learning jazz improvisation: detects harmony patterns and recommends contextually appropriate phrases from solos by renowned jazz musicians. The core idea is to use machine learning to segment note sequences into musical phrases, enabling automatic expansion of the recommendations database.

This repository contains the ML and backend components of the project. [Mobile app code](https://github.com/TemSV/learning-jazz-improvization-mobile)

## Machine Learning Model
- Task: binary classification per note (phrase start vs continuation)
- Data: Weimar Jazz Database (456 solos, 11,082 phrases)
- Architecture: Modular Transformer Encoder with Rotary Positional Embeddings (RoPE) and Feature Fusion.
- Features: Discrete (pitch embeddings) and continuous (duration, loudness, metric weight, intervals, rests) combined via a dedicated fusion layer.
- Training: Masked Focal Loss, AdamW, padding masks for variable-length sequences.

## Data Preparation
- Dataset based on [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html)
- Cleaning and anomaly handling: filling missing values, replacing infinite and NaN values, normalizations and log-transforms for distribution smoothing.
- Feature engineering: relative onsets and durations, normalized intervals and direction of movement, combined pause and inter-note interval features, rhythmic complexity metric and metric weight.
- Scaling: standard feature scaling (StandardScaler) before training. Discrete features (pitch) are isolated for embedding layers.
- Data split: stratified train/val/test = 70%/15%/15% by complexity (number of phrases in melody) to balance styles and solo lengths.

## Neural Network Training
- Loss function: Masked Focal Loss to handle the severe class imbalance (99% background vs 1% phrase starts) and correctly ignore padding tokens.
- Optimization: AdamW, dynamic batching via custom collate function.
- Regularization and stability: Dropout layers in attention and feed-forward networks, strict tensor typing, and sliding window sequence generation.

## Model Results
- Loss: 0.1142
- Precision: 0.8966
- Recall: 0.8302
- F1: 0.8621
- Accuracy: 0.9859

Focus on high precision for correct phrase boundaries.

## Application Architecture

<p align="center">
  <img src="assets/C4-container.drawio.png" alt="System container diagram" width="800"><br/>
  Container diagram
</p>

<p align="center">
  <img src="assets/C4-component.drawio.png" alt="System component diagram" width="800"><br/>
  Component diagram
</p>

## API
- GET /api/songs
  Paginated list of songs with search by title.
  Parameters: `q` (optional), `limit` (default 20), `offset` (default 0).
  Response: `{ total, items: [{ id, title }] }`.

- GET /api/songs/{song_id}/chords
  Chord sequence of the song.
  Path parameters: `song_id` (required).
  Response: `{ song_id, title, bars: [{ id, number, time_signature, chords, section }] }`.

- GET /api/songs/{song_id}/patterns
  Detected harmonic patterns for the song.
  Path parameters: `song_id` (required).
  Response: `{ song_id, title, patterns: [{ type, key, bar_ids: [..], normalized_chords: [{ chord, duration }], features: [...] }] }`.

- POST /api/recommendations/phrases
  Phrase recommendations for a given pattern.
  Body: `{ features: [...] }` (numeric feature vector of the pattern).
  Response: `{ items: [{ melid, first_note_id, last_note_id, score, chords }] }`.

- GET /api/phrases/{melid}/notes
  Notes of the selected phrase for playback.
  Path parameters: `melid` (required).
  Query parameters: `first_note_id`, `last_note_id` (both required).
  Response: `{ notes: [{ pitch, onset, duration, loudness }] }`.

## Repository Structure
