from typing import List, Tuple, Dict, Optional
import math
import re

from .models import ChordInfo, ChordQuality, ChordPattern, ChordWithDuration, CHORD_TYPE_TO_NUMERIC
from .patterns import (HarmonicPattern, TwoFiveOnePattern, BluesPattern,
                       MinorTwoFiveOnePattern)
from ..utils.similarity_utils import calculate_harmonic_features


class HarmonyAnalyzer:
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
            quality = ChordQuality.MAJOR

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
            
            if interval > 6:
                interval -= 12
            intervals.append(interval)

        return intervals

    def _get_root(self, chord: str) -> Optional[str]:
        """Extracts the root note from a chord string. Returns None if invalid."""
        if not chord: return None
        
        main_chord = chord.split('/')[0].strip()
        if not main_chord: return None

        if len(main_chord) > 1 and main_chord[1] in ['#', 'b']:
            root = main_chord[:2]
            if root in self.note_values: return root
        elif main_chord[0] in self.note_values:
            return main_chord[0]
        return None

    def _get_chord_type(self, chord: str) -> str:
        """
        Determines the chord type string (centralized logic).
        Handles slash chords (analyzes chord before '/'), sus chords, and 6th chords.
        """
        if not chord: return 'unknown'

        main_chord_part = chord.split('/')[0].strip()
        if not main_chord_part: return 'unknown'

        quality_part = ''
        root = self._get_root(main_chord_part)
        if root:
             quality_part = main_chord_part[len(root):]
        else:
             return 'unknown'


        if 'm7b5' in quality_part or '-7b5' in quality_part: return 'half_dim'
        if 'j7' in quality_part or 'maj7' in quality_part: return 'maj7'
        if 'o7' in quality_part or 'dim7' in quality_part: return 'dim7' # Fully diminished 7th
        if 'dim' in quality_part or quality_part.endswith('o'): return 'dim' # Diminished triad

        is_sus = 'sus' in quality_part
        is_sus4 = 'sus4' in quality_part
        is_sus2 = 'sus2' in quality_part

        if is_sus4: return 'sus4'
        if is_sus2: return 'sus2'
        if is_sus: return 'sus4'

        if ('m7' in quality_part or '-7' in quality_part) and 'b5' not in quality_part: return 'min7'
        if '7' in quality_part: return 'dom7'

        if 'm6' in quality_part or '-6' in quality_part: return 'min6'
        if 'maj6' in quality_part or '6' in quality_part: return 'maj6'

        if ('-' in quality_part or 'm' in quality_part): return 'min'
        if 'aug' in quality_part or '+' in quality_part: return 'aug'
        if quality_part == '': return 'maj'

        return 'unknown'

    def get_relative_interval(self, chord1: str, chord2: str) -> Optional[int]:
        """
        Calculates the interval between two chords' roots in semitones.
        Returns None if roots are invalid. Ignores bass note in slash chords.
        Positive values indicate upward motion (shortest path).
        """
        root1 = self._get_root(chord1)
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

    def compute_comparison_features(self, chords_with_duration: List[ChordWithDuration],
                                    interval_weight: float = 1.5,
                                    chord_type_weight: float = 2.0,
                                    chord_duration_weight: float = 2.0) -> Dict[str, float]:
        if not chords_with_duration:
            return {}   

        # 1. Pre-calculate chord types and relative intervals
        chord_type_strs: List[str] = []
        for cd in chords_with_duration:
            chord_type_strs.append(self._get_chord_type(cd.chord))
        
        relative_intervals: List[Optional[int]] = []
        for i in range(len(chords_with_duration) - 1):
            interval = self.get_relative_interval(chords_with_duration[i].chord, chords_with_duration[i+1].chord)
            relative_intervals.append(interval)


        # 2. Call the centralized feature calculation function
        return calculate_harmonic_features(
            chords_with_duration=chords_with_duration,
            chord_type_strs=chord_type_strs,
            relative_intervals=relative_intervals,
            interval_weight=interval_weight,
            chord_type_weight=chord_type_weight,
            chord_duration_weight=chord_duration_weight
        )

    def find_patterns(self, chord_sequence: List[Tuple[int, List[str]]]) -> List[ChordPattern]:
        """
        Finds all registered patterns, computes comparison features, and returns ChordPattern objects.
        """
        chords_with_duration = self.process_chord_sequence(chord_sequence)
        if not chords_with_duration: return []

        found_patterns = []
        for pattern_def in self.patterns:
            window_size = pattern_def.get_window_size()
            if len(chords_with_duration) < window_size:
                continue

            for i in range(len(chords_with_duration) - window_size + 1):
                window = chords_with_duration[i:i + window_size]
               
                if pattern_def.match(window, self):
                    pattern_obj = pattern_def.create_pattern(window, i)
                    pattern_obj.features = self.compute_comparison_features(pattern_obj.chords)
                    found_patterns.append(pattern_obj)

        return found_patterns

    def process_chord_sequence(self, chord_sequence: List[Tuple[int, List[str]]]) -> List[ChordWithDuration]:
        raw_chords_with_duration = []
        current_chord_str = None
        current_duration = 0.0
        beats_per_bar = 4.0

        last_bar_num = -1
        if chord_sequence:
             last_bar_num = chord_sequence[-1][0]

        bar_map = {bar_num: bar_chords for bar_num, bar_chords in chord_sequence}
        start_bar = chord_sequence[0][0] if chord_sequence else 0


        active_chord = None
        active_chord_start_time = 0.0
        current_time = 0.0

        for bar_num in range(start_bar, last_bar_num + 1):
            bar_start_time = (bar_num - start_bar) * beats_per_bar
            bar_chords = bar_map.get(bar_num, [])

            if not bar_chords:
                 if active_chord:
                      pass
                 current_time = bar_start_time + beats_per_bar
            else:
                 beats_per_chord_in_bar = beats_per_bar / len(bar_chords)
                 for i, chord_str in enumerate(bar_chords):
                     chord_start_time_in_bar = i * beats_per_chord_in_bar
                     current_chord_time = bar_start_time + chord_start_time_in_bar

                     if active_chord and chord_str != active_chord:
                          duration = current_chord_time - active_chord_start_time
                          if duration > 1e-6:
                               raw_chords_with_duration.append(ChordWithDuration(active_chord, duration))
                          active_chord = chord_str
                          active_chord_start_time = current_chord_time

                     elif not active_chord:
                          active_chord = chord_str
                          active_chord_start_time = current_chord_time

                 current_time = bar_start_time + beats_per_bar

        if active_chord:
             end_time = (last_bar_num + 1 - start_bar) * beats_per_bar
             duration = end_time - active_chord_start_time
             if duration > 1e-6:
                   raw_chords_with_duration.append(ChordWithDuration(active_chord, duration))

        if not raw_chords_with_duration: return []
        merged = []
        if raw_chords_with_duration:
            current = raw_chords_with_duration[0]
            for next_chord in raw_chords_with_duration[1:]:
                 if next_chord.chord == current.chord:
                      current.duration += next_chord.duration
                 else:
                      merged.append(current)
                      current = ChordWithDuration(next_chord.chord, next_chord.duration)
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

    def is_sus_chord(self, chord: str) -> bool:
        chord_type = self._get_chord_type(chord)
        return chord_type == 'sus4' or chord_type == 'sus2'

    def is_major6_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'maj6'

    def is_minor6_chord(self, chord: str) -> bool:
        return self._get_chord_type(chord) == 'min6'
                