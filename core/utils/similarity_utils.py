import math
from typing import Dict, Set, List, Optional

from core.pattern_analysis.models import CHORD_TYPE_TO_NUMERIC, ChordWithDuration

def calculate_cosine_similarity(features1: Dict[str, float], features2: Dict[str, float]) -> float:
    keys: Set[str] = set(features1.keys()) | set(features2.keys())
    if not keys:
        return 1.0 if not features1 and not features2 else 0.0

    vector1: List[float] = [features1.get(k, 0.0) for k in keys]
    vector2: List[float] = [features2.get(k, 0.0) for k in keys]

    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm1 = math.sqrt(sum(a * a for a in vector1))
    norm2 = math.sqrt(sum(b * b for b in vector2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0 if norm1 == 0.0 and norm2 == 0.0 else 0.0
    else:
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))


def calculate_harmonic_features(
    chords_with_duration: List[ChordWithDuration],
    chord_type_strs: List[str],
    relative_intervals: List[Optional[int]],
    interval_weight: float,
    chord_type_weight: float,
    chord_duration_weight: float
) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if not chords_with_duration:
        return features

    total_duration = sum(c.duration for c in chords_with_duration)
    if total_duration <= 0:
        return features

    num_chords = len(chords_with_duration)
    features['feat_num_chords'] = num_chords / 10.0  # Normalization
    features['feat_total_duration'] = total_duration / 20.0  # Normalization

    for i, chord_dur in enumerate(chords_with_duration):
        features[f'feat_chord_{i}_position'] = i / num_chords if num_chords > 1 else 0

        raw_duration_feature = chord_dur.duration / total_duration
        features[f'feat_chord_{i}_duration'] = raw_duration_feature * chord_duration_weight

        numeric_chord_type = CHORD_TYPE_TO_NUMERIC.get(chord_type_strs[i], CHORD_TYPE_TO_NUMERIC['unknown'])
        features[f'feat_chord_{i}_type'] = numeric_chord_type * chord_type_weight

        if i < num_chords - 1:
            interval = relative_intervals[i]
            interval_feature_normalized = 0.5
            if interval is not None:
                normalized_interval = (interval + 5.0) / 11.0
                interval_feature_normalized = max(0.0, min(1.0, normalized_interval))
            features[f'feat_interval_{i}'] = interval_feature_normalized * interval_weight
            
    return features
