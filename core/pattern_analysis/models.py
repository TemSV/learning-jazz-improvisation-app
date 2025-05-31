from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional



CHORD_TYPE_TO_NUMERIC = {
    # Major family (values around 0.8-0.9)
    'maj': 0.8,
    'maj6': 0.85,
    'maj7': 0.9,

    # Minor family (values around 0.5-0.6)
    'min': 0.5,
    'min6': 0.55,
    'min7': 0.6,

    # Dominant family (values around 0.2-0.3)
    'dom7': 0.3,

    # Diminished/Altered Tension (values around 0.0-0.15)
    'half_dim': 0.1,  # m7b5
    'dim': 0.12,      # diminished triad
    'dim7': 0.15,     # fully diminished 7th
    'aug': 0.05,      # augmented (distinct tension)

    # Suspended (distinct category, e.g., around 0.7)
    'sus4': 0.7,
    'sus2': 0.72,

    'unknown': 0.4   # A neutral or less emphasized value for unknowns
}


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
    root: str
    quality: ChordQuality

@dataclass
class ChordWithDuration:
    chord: str
    duration: float

@dataclass
class ChordPattern:
    pattern_type: str
    chords: List[ChordWithDuration]
    start_bar: int
    key: str
    features: Dict[str, float] = field(default_factory=dict)
    total_duration: float = field(init=False)

    def __post_init__(self):
        self.total_duration = sum(chord.duration for chord in self.chords)

    def to_dict(self) -> Dict:
        """Converts the pattern to a dictionary for database storage"""
        return {
            'pattern_type': self.pattern_type,
            'chords': [(c.chord, c.duration) for c in self.chords],
            'start_bar': self.start_bar,
            'total_duration': self.total_duration,
            'key': self.key,
            'features': self.features
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChordPattern':
        """Creates a pattern object from a dictionary"""
        chords_data = data.get('chords', [])
        chords = [ChordWithDuration(chord, duration)
                 for chord, duration in chords_data]

        pattern = cls(
            pattern_type=data.get('pattern_type', 'Unknown'),
            chords=chords,
            start_bar=data.get('start_bar', -1),
            key=data.get('key', ''),
        )
        pattern.features = data.get('features', {})
        return pattern

    def get_feature_vector(self) -> List[float]:
        if not self.features:
            return []
        sorted_keys = sorted(self.features.keys())
        return [self.features[key] for key in sorted_keys]


@dataclass
class PhraseSectionInfo:
    """Basic info about a phrase entry in the sections table"""
    section_id: int
    melid: int
    phrase_type: str
    start_eventid: int
    end_eventid: int
    value: str

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
    phrase_type: str
    start_note_index: int
    end_note_index: int
    value: str            
    start_onset: Optional[float] = field(default=None, init=False)
    end_onset: Optional[float] = field(default=None, init=False)
    phrase_db_id: Optional[int] = None
