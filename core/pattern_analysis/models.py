from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ChordQuality(Enum):
    MAJOR = ''
    MINOR = '-'
    DOMINANT = '7'
    MAJOR7 = 'j7'
    MINOR7 = 'm7'
    HALF_DIMINISHED = 'm7b5'
    DIMINISHED = 'o7'

@dataclass
class ChordInfo:
    root: str  # Root note (C, D, E, etc.)
    quality: ChordQuality  # Chord quality (maj, min, 7, etc.)

@dataclass
class ChordWithDuration:
    chord: str
    duration: float  # duration in beats

@dataclass
class ChordPattern:
    """Class to store information about a harmonic pattern"""
    pattern_type: str  # Type of pattern (e.g., "II-V-I", "Blues", etc.)
    chords: List[ChordWithDuration]  # List of chords with their durations
    start_bar: int  # Start index in the processed chord list
    key: str  # Key of the pattern (e.g., "F" for II-V-I in F) - Keep for info, but won't use in comparison features
    # Features will be computed and assigned externally
    features: Dict[str, float] = field(default_factory=dict)
    total_duration: float = field(init=False) # Total duration of the pattern in beats

    def __post_init__(self):
        # Calculate total_duration
        # Features are no longer computed here
        self.total_duration = sum(chord.duration for chord in self.chords)

    def to_dict(self) -> Dict:
        """Converts the pattern to a dictionary for database storage"""
        return {
            'pattern_type': self.pattern_type,
            'chords': [(c.chord, c.duration) for c in self.chords],
            'start_bar': self.start_bar,
            'total_duration': self.total_duration,
            'key': self.key,
            'features': self.features # Store the computed features
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChordPattern':
        """Creates a pattern object from a dictionary"""
        chords_data = data.get('chords', [])
        chords = [ChordWithDuration(chord, duration)
                 for chord, duration in chords_data]

        # Create instance, __post_init__ calculates total_duration
        pattern = cls(
            pattern_type=data.get('pattern_type', 'Unknown'),
            chords=chords,
            start_bar=data.get('start_bar', -1),
            key=data.get('key', ''),
        )
        # Assign features if they exist in the loaded data
        pattern.features = data.get('features', {})
        return pattern

    # get_feature_vector might still be useful if needed elsewhere
    def get_feature_vector(self) -> List[float]:
        """Returns the feature vector of the pattern for comparison"""
        if not self.features:
            return []
        sorted_keys = sorted(self.features.keys())
        return [self.features[key] for key in sorted_keys]


@dataclass
class PhraseSectionInfo:
    """Basic info about a phrase entry in the sections table"""
    section_id: int # Assuming 'sections' table has a primary key 'section_id'
    melid: int
    phrase_type: str # e.g., 'phrase'
    start_eventid: int # Global eventid from melody table marking the start
    end_eventid: int   # Global eventid from melody table marking the end
    value: str         # Annotation or label for the section

@dataclass
class BeatInfo:
    """Represents a row from the 'beats' table relevant to a phrase"""
    onset: float
    bar: int
    beat: int
    signature: str
    chord: str

@dataclass
class PhraseInfo:
    """Represents a phrase definition from the database."""
    melid: int
    phrase_type: str # 'phrase' or other types from sections.type
    start_note_index: int # 0-based index within the melody's notes (ordered by onset)
    end_note_index: int   # 0-based index (exclusive)
    value: str            # The 'value' field from sections table
    # Optional fields are now correctly typed
    start_onset: Optional[float] = field(default=None, init=False)
    end_onset: Optional[float] = field(default=None, init=False)
    phrase_db_id: Optional[int] = None # Assuming sections might have a primary key later
