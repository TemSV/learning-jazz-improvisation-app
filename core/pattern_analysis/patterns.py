from abc import ABC, abstractmethod
from typing import List

from .models import ChordPattern, ChordWithDuration, ChordQuality
# Forward declaration for type hinting PatternAnalyzer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .harmony_analyzer import HarmonyAnalyzer

class HarmonicPattern(ABC):
    """Base class for all harmonic patterns"""
    
    def __init__(self):
        self.pattern_type: str = self.__class__.__name__
    
    @abstractmethod
    def match(self, chords: List[ChordWithDuration], analyzer: 'HarmonyAnalyzer') -> bool:
        """Checks if the chord sequence matches the pattern"""
        pass
    
    @abstractmethod
    def create_pattern(self, chords: List[ChordWithDuration], start_index: int) -> ChordPattern:
        """Creates a pattern object from the chord sequence"""
        pass
        
    @abstractmethod
    def get_window_size(self) -> int:
        """Returns the window size for pattern search (number of chords)"""
        pass

    def _get_root(self, chord: str) -> str:
        """Extracts the root note from a chord string."""
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            return chord[:2]
        # Handle cases like empty string or just '#'/'b'
        elif len(chord) > 0:
            return chord[0]
        else:
            return "" # Return empty string for invalid input


class TwoFiveOnePattern(HarmonicPattern):
    """II-V-I pattern (Major resolution: IIm7 - V7 - Imaj/Imaj7)"""
    
    def get_window_size(self) -> int:
        return 3
    
    def match(self, chords: List[ChordWithDuration], analyzer: 'HarmonyAnalyzer') -> bool:
        if len(chords) != 3:
            return False
            
        chord2, chord5, chord1 = chords
        
        # Check intervals (II->V P4 up, V->I P4 up)
        interval1 = analyzer.get_relative_interval(chord2.chord, chord5.chord)
        interval2 = analyzer.get_relative_interval(chord5.chord, chord1.chord)
        
        if not (interval1 == 5 and interval2 == 5):
             return False
            
        # Check chord qualities: IIm7, V7, Imaj/Imaj7 (Strict Major Tonic)
        is_minor_7_ii = analyzer.is_minor7_chord(chord2.chord)
        is_dominant_v = analyzer.is_dominant_chord(chord5.chord)
        is_major_tonic = (analyzer.is_major_chord(chord1.chord) or
                          analyzer.is_major7_chord(chord1.chord))

        return is_minor_7_ii and is_dominant_v and is_major_tonic
    
    def create_pattern(self, chords: List[ChordWithDuration], start_index: int) -> ChordPattern:
        # Tonic is Major
        chord1_str = chords[2].chord
        chord1_root = self._get_root(chord1_str)

        return ChordPattern(
            pattern_type="II-V-I", # Major II-V-I
            chords=chords,
            start_bar=start_index, # Use index in processed list
            key=chord1_root # Major Key
        )


class MinorTwoFiveOnePattern(HarmonicPattern):
    """Minor II-V-I pattern (Classic: IIm7b5 - V7 - Im/Im7)"""

    def get_window_size(self) -> int:
        return 3

    def match(self, chords: List[ChordWithDuration], analyzer: 'HarmonyAnalyzer') -> bool:
        if len(chords) != 3:
            return False

        chord2, chord5, chord1 = chords

        # Check intervals (II->V P4 up, V->I P4 up)
        interval1 = analyzer.get_relative_interval(chord2.chord, chord5.chord)
        interval2 = analyzer.get_relative_interval(chord5.chord, chord1.chord)

        if not (interval1 == 5 and interval2 == 5):
             return False

        # Check chord qualities: IIm7b5, V7, Im/Im7
        is_half_dim_ii = analyzer.is_half_diminished_chord(chord2.chord)
        is_dominant_v = analyzer.is_dominant_chord(chord5.chord)
        is_minor_tonic = (analyzer.is_minor_chord(chord1.chord) or
                          analyzer.is_minor7_chord(chord1.chord))

        return is_half_dim_ii and is_dominant_v and is_minor_tonic

    def create_pattern(self, chords: List[ChordWithDuration], start_index: int) -> ChordPattern:
        # Tonic is Minor
        chord1_str = chords[2].chord
        chord1_root = self._get_root(chord1_str)

        return ChordPattern(
            pattern_type="Minor-II-V-I",
            chords=chords,
            start_bar=start_index, # Use index in processed list
            key=f"{chord1_root}m" # Minor Key
        )


class BluesPattern(HarmonicPattern):
    """Blues pattern (Basic I-IV-V, specific qualities)"""
    # Note: This is a simplified blues pattern example.
    # Real blues can be much more complex (e.g., 12-bar structure).
    # This currently looks for Tonic(Maj/Maj7/Dom) -> Subdominant(min) -> Dominant(Dom)
    # with specific intervals.

    def get_window_size(self) -> int:
        return 3
    
    def match(self, chords: List[ChordWithDuration], analyzer: 'HarmonyAnalyzer') -> bool:
        if len(chords) != 3:
            return False
            
        chord1, chord4, chord5 = chords # Assuming I-IV-V structure for naming
        
        # Check intervals: I -> IV (P4 up = 5), IV -> V (M2 up = 2 or m3 up = 3? Let's stick to 3 for now)
        # Original comment said interval2 == 3, which is m3 up (e.g. Fm -> G7).
        interval1 = analyzer.get_relative_interval(chord1.chord, chord4.chord)
        interval2 = analyzer.get_relative_interval(chord4.chord, chord5.chord)

        # Current logic checks I->IV (P4 up) and IV->V (m3 up)
        if not (interval1 == 5 and interval2 == 3):
            return False
            
        # Check chord qualities: Tonic (Maj/Maj7/Dom), Subdominant (min), Dominant (Dom)
        is_tonic = (analyzer.is_major_chord(chord1.chord) or
                   analyzer.is_major7_chord(chord1.chord) or
                   analyzer.is_dominant_chord(chord1.chord))
        is_subdominant_minor = analyzer.is_minor_chord(chord4.chord)
        is_dominant = analyzer.is_dominant_chord(chord5.chord)

        return is_tonic and is_subdominant_minor and is_dominant
    
    def create_pattern(self, chords: List[ChordWithDuration], start_index: int) -> ChordPattern:
        # Key is determined by the first chord (Tonic)
        chord1_str = chords[0].chord
        key = self._get_root(chord1_str)

        return ChordPattern(
            pattern_type="Blues",
            chords=chords,
            start_bar=start_index, # Use index in processed list
            key=key
        )
