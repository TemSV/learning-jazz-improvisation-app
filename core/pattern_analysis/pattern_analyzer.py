from typing import List, Tuple, Dict, Optional
import math
import re # Import regex for splitting slash chords

from .models import ChordInfo, ChordQuality, ChordPattern, ChordWithDuration
from .patterns import (HarmonicPattern, TwoFiveOnePattern, BluesPattern,
                       MinorTwoFiveOnePattern)


class PatternAnalyzer:
    def __init__(self):
        self.note_values = {
            'C': 0, 'C#': 1, 'Db': 1,
            'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4,
            'F': 5, 'F#': 6, 'Gb': 6,
            'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10,
            'B': 11
        }
        # Register available patterns
        self.patterns: List[HarmonicPattern] = [
            TwoFiveOnePattern(),
            MinorTwoFiveOnePattern(),
            BluesPattern()
        ]

    def parse_chord(self, chord: str) -> ChordInfo:
        """
        Parses a string representation of a chord into its components.
        Example: "Dm7" -> ChordInfo(root="D", quality=ChordQuality.MINOR7)
        """
        root = chord[0]
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            root = chord[:2]
            quality_part = chord[2:]
        else:
            quality_part = chord[1:]

        if 'j7' in quality_part or 'maj7' in quality_part:
            quality = ChordQuality.MAJOR7
        elif 'm7b5' in quality_part or '-7b5' in quality_part:
            quality = ChordQuality.HALF_DIMINISHED
        elif '-7' in quality_part or 'm7' in quality_part:
            quality = ChordQuality.MINOR7
        elif '7' in quality_part:
            quality = ChordQuality.DOMINANT
        elif '-' in quality_part or 'm' in quality_part:
            quality = ChordQuality.MINOR
        elif 'o7' in quality_part or 'dim' in quality_part:
            quality = ChordQuality.DIMINISHED
        else:
            quality = ChordQuality.MAJOR # Default to Major

        return ChordInfo(root=root, quality=quality)

    def get_chord_intervals(self, chord_sequence: List[str]) -> List[int]:
        """
        Calculates the intervals between consecutive chords in semitones.
        """
        intervals = []
        for i in range(len(chord_sequence) - 1):
            current = self.parse_chord(chord_sequence[i])
            next_chord = self.parse_chord(chord_sequence[i + 1])

            current_value = self.note_values[current.root]
            next_value = self.note_values[next_chord.root]

            interval = (next_value - current_value) % 12
            # Represent intervals as shortest path (+/-)
            if interval > 6:
                interval -= 12
            intervals.append(interval)

        return intervals

    def _get_root(self, chord: str) -> Optional[str]:
        """Extracts the root note from a chord string. Returns None if invalid."""
        if not chord: return None
        # Handle slash chords: only consider the part before '/' for the root
        main_chord = chord.split('/')[0].strip()
        if not main_chord: return None

        if len(main_chord) > 1 and main_chord[1] in ['#', 'b']:
            root = main_chord[:2]
            if root in self.note_values: return root
        elif main_chord[0] in self.note_values:
            return main_chord[0]
        return None # Invalid root

    def _get_chord_type(self, chord: str) -> str:
        """
        Determines the chord type string (centralized logic).
        Handles slash chords (analyzes chord before '/'), sus chords, and 6th chords.
        """
        if not chord: return 'unknown'

        # Handle slash chords: analyze only the main chord part
        main_chord_part = chord.split('/')[0].strip()
        if not main_chord_part: return 'unknown'

        # Extract root and quality part from the main chord part
        quality_part = ''
        root = self._get_root(main_chord_part) # Use main_chord_part here
        if root:
             quality_part = main_chord_part[len(root):]
        else:
             # If root is invalid, cannot determine quality reliably
             return 'unknown'

        # --- Type determination logic with improved order and new types ---

        # Most specific first
        if 'm7b5' in quality_part or '-7b5' in quality_part: return 'half_dim'
        if 'j7' in quality_part or 'maj7' in quality_part: return 'maj7'
        if 'o7' in quality_part or 'dim7' in quality_part: return 'dim7' # Fully diminished 7th
        if 'dim' in quality_part or quality_part.endswith('o'): return 'dim' # Diminished triad

        # Suspended chords (check before simpler 7ths/triads)
        # Need careful check for sus with 7th etc. "sus" often means sus4
        is_sus = 'sus' in quality_part
        is_sus4 = 'sus4' in quality_part
        is_sus2 = 'sus2' in quality_part

        if is_sus4: return 'sus4' # Prioritize explicit sus4
        if is_sus2: return 'sus2' # Prioritize explicit sus2
        if is_sus: return 'sus4' # Treat generic 'sus' as 'sus4'

        # Sevenths (after more specific types)
        if ('m7' in quality_part or '-7' in quality_part) and 'b5' not in quality_part: return 'min7'
        if '7' in quality_part: return 'dom7' # Dominant 7th

        # Sixths (check after sevenths)
        if 'm6' in quality_part or '-6' in quality_part: return 'min6'
        if 'maj6' in quality_part or '6' in quality_part: return 'maj6' # Treat '6' as Maj6

        # Basic Triads
        if ('-' in quality_part or 'm' in quality_part): return 'min' # Minor triad
        if 'aug' in quality_part or '+' in quality_part: return 'aug' # Augmented triad
        if quality_part == '': return 'maj' # Major triad if nothing else specified

        # Fallback for unrecognized quality parts
        # Don't print warning here, let caller decide if needed
        # print(f"Warning: Unknown chord quality for '{main_chord_part}', quality part '{quality_part}'")
        return 'unknown'

    def get_relative_interval(self, chord1: str, chord2: str) -> Optional[int]:
        """
        Calculates the interval between two chords' roots in semitones.
        Returns None if roots are invalid. Ignores bass note in slash chords.
        Positive values indicate upward motion (shortest path).
        """
        root1 = self._get_root(chord1) # _get_root handles slash chords
        root2 = self._get_root(chord2)

        if root1 is None or root2 is None:
            return None

        val1 = self.note_values[root1]
        val2 = self.note_values[root2]

        interval = (val2 - val1) % 12
        if interval > 6:
             return interval - 12
        else:
             return interval

    def _compute_comparison_features(self, chords: List[ChordWithDuration], interval_weight: float = 3.0) -> Dict[str, float]:
        """
        Computes key-invariant features for comparison (for patterns or phrases).
        Focuses on relative durations, types, positions, and intervals.
        """
        features = {}
        if not chords:
            return features

        total_duration = sum(c.duration for c in chords)
        if total_duration <= 0:
            return features

        num_chords = len(chords)
        features['feat_num_chords'] = num_chords / 10.0
        features['feat_total_duration'] = total_duration / 20.0

        for i, chord_dur in enumerate(chords):
            features[f'feat_chord_{i}_position'] = i / num_chords if num_chords > 1 else 0
            features[f'feat_chord_{i}_duration'] = chord_dur.duration / total_duration

            # Chord type uses the updated _get_chord_type
            chord_type = self._get_chord_type(chord_dur.chord)
            features[f'feat_chord_{i}_type'] = hash(chord_type) % 100 / 100.0

            # Interval uses the updated get_relative_interval
            if i < num_chords - 1:
                interval = self.get_relative_interval(chord_dur.chord, chords[i+1].chord)
                interval_feature_normalized = 0.5 # Default
                if interval is not None:
                     # Normalize interval from range [-5, 6] to [0, 1]
                     # Add 5 to shift range to [0, 11], then divide by 11
                     # Clamp the value just in case interval is outside [-5, 6], though it shouldn't be
                     normalized_interval = (interval + 5.0) / 11.0
                     interval_feature_normalized = max(0.0, min(1.0, normalized_interval))
                # Apply weight to the normalized interval feature
                features[f'feat_interval_{i}'] = interval_feature_normalized * interval_weight

        return features

    def find_patterns(self, chord_sequence: List[Tuple[int, List[str]]]) -> List[ChordPattern]:
        """
        Finds all registered patterns, computes comparison features, and returns ChordPattern objects.
        """
        chords_with_duration = self._process_chord_sequence(chord_sequence)
        if not chords_with_duration: return []

        found_patterns = []
        for pattern_def in self.patterns:
            window_size = pattern_def.get_window_size()
            if len(chords_with_duration) < window_size:
                continue

            for i in range(len(chords_with_duration) - window_size + 1):
                window = chords_with_duration[i:i + window_size]
                # Match uses the updated quality check methods below
                if pattern_def.match(window, self):
                    pattern_obj = pattern_def.create_pattern(window, i)
                    # Features are computed using the updated logic
                    pattern_obj.features = self._compute_comparison_features(pattern_obj.chords)
                    found_patterns.append(pattern_obj)

        return found_patterns

    def _process_chord_sequence(self, chord_sequence: List[Tuple[int, List[str]]]) -> List[ChordWithDuration]:
        # This method might need more robust duration calculation if time signatures vary significantly
        # Assuming 4/4 for now for beat calculation within bars
        raw_chords_with_duration = []
        current_chord_str = None
        current_duration = 0.0 # Use float
        beats_per_bar = 4.0 # Use float

        last_bar_num = -1
        if chord_sequence:
             last_bar_num = chord_sequence[-1][0]

        bar_map = {bar_num: bar_chords for bar_num, bar_chords in chord_sequence}
        start_bar = chord_sequence[0][0] if chord_sequence else 0

        # Propagate initial chord if needed (complex, depends on wider context)
        # Simplified: Assume sequence starts fresh or first bar defines initial state

        active_chord = None
        active_chord_start_time = 0.0
        current_time = 0.0 # Absolute time in beats

        for bar_num in range(start_bar, last_bar_num + 1):
            bar_start_time = (bar_num - start_bar) * beats_per_bar # Time at the beginning of this bar
            bar_chords = bar_map.get(bar_num, [])

            if not bar_chords:
                 # Empty bar, potentially extend previous chord duration
                 # This simple logic assumes the previous chord lasts the whole bar
                 if active_chord:
                      pass # Duration handled by the start of the *next* chord change or end of sequence
                 current_time = bar_start_time + beats_per_bar # Advance time to end of bar
            else:
                 beats_per_chord_in_bar = beats_per_bar / len(bar_chords)
                 for i, chord_str in enumerate(bar_chords):
                     chord_start_time_in_bar = i * beats_per_chord_in_bar
                     current_chord_time = bar_start_time + chord_start_time_in_bar

                     # If this is a change from the active chord, finalize the previous one
                     if active_chord and chord_str != active_chord:
                          duration = current_chord_time - active_chord_start_time
                          if duration > 1e-6: # Avoid zero-duration chords
                               raw_chords_with_duration.append(ChordWithDuration(active_chord, duration))
                          # Start the new chord
                          active_chord = chord_str
                          active_chord_start_time = current_chord_time

                     elif not active_chord:
                           # First chord encountered in the sequence
                           active_chord = chord_str
                           active_chord_start_time = current_chord_time # Should be bar_start_time if first chord in first bar

                 # Advance current_time to the end of the bar after processing all chords in it
                 current_time = bar_start_time + beats_per_bar

        # Add the very last active chord after the loop finishes
        if active_chord:
             # Calculate duration until the effective end of the sequence
             # End time is the time after the last bar processed
             end_time = (last_bar_num + 1 - start_bar) * beats_per_bar
             duration = end_time - active_chord_start_time
             if duration > 1e-6:
                   raw_chords_with_duration.append(ChordWithDuration(active_chord, duration))

        # Merge consecutive identical chords
        if not raw_chords_with_duration: return []
        merged = []
        if raw_chords_with_duration:
            current = raw_chords_with_duration[0]
            for next_chord in raw_chords_with_duration[1:]:
                 if next_chord.chord == current.chord:
                      current.duration += next_chord.duration
                 else:
                      merged.append(current)
                      current = ChordWithDuration(next_chord.chord, next_chord.duration) # Create new object
            merged.append(current)

        return merged

    def _get_quality_part(self, chord: str) -> str:
        """Extracts the quality part of the chord string (after the root)."""
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            return chord[2:]
        return chord[1:]

    # --- Chord Quality Check Methods ---

    def is_dominant_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'dom7'

    def is_minor_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'min'

    def is_major_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'maj'

    def is_major7_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'maj7'

    def is_half_diminished_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'half_dim'

    def is_minor7_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'min7'

    def is_diminished_chord(self, chord: str) -> bool:
        # Check for both triad and 7th
        chord_type = self._get_chord_type(chord)
        return chord_type == 'dim' or chord_type == 'dim7'

    # Add checks for new types if needed elsewhere (though _get_chord_type is primary)
    def is_sus_chord(self, chord: str) -> bool:
        chord_type = self._get_chord_type(chord)
        return chord_type == 'sus4' or chord_type == 'sus2'

    def is_major6_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'maj6'

    def is_minor6_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'min6'
                