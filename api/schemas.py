from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional

# --- Basic Data Structures ---

class ChordDuration(BaseModel):
    chord: str
    duration: float

# Reference to a specific chord within a specific bar of the *original* song chords list
class OriginalChordRef(BaseModel):
    barid: int

# --- Song Related Schemas ---

class SongInfo(BaseModel):
    id: int
    title: str | None = None

class SongListResponse(BaseModel):
    songs: List[SongInfo]
    total: int

class SongChordEntry(BaseModel):
    barid: int
    bar: int
    signature: str | None = None
    chords: str | None = None
    form: str | None = None

class SongChordsResponse(BaseModel):
    song_id: int
    title: str | None = None
    chords: List[SongChordEntry]

# --- Pattern Related Schemas (Re-added/Updated) ---

class PatternInfo(BaseModel):
    type: str
    key: str | None = None
    original_chord_refs: List[OriginalChordRef] = Field(description="References to the original chords/bars that form this pattern")
    chords: List[ChordDuration]
    features: Dict[str, float] = Field(description="Features dict for this pattern instance")

class SongPatternsResponse(BaseModel):
    song_id: int
    title: str | None = None
    # Removed processed_song_chords
    patterns: List[PatternInfo]


# --- Recommendation Related Schemas ---

class RecommendationRequest(BaseModel):
    pattern_features: Dict[str, float]
    limit: int = Field(default=5, ge=1, le=50)

class RecommendedPhrase(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    melid: int
    start_index: int = Field(alias="start_note_index")
    end_index: int = Field(alias="end_note_index")
    similarity: float
    chords: List[ChordDuration]

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedPhrase]

class NoteData(BaseModel):
    pitch: int
    onset: float
    duration: float
    loud_med: Optional[float] = None
    f0_med_dev: Optional[float] = None

    class Config:
        from_attributes = True


class PhraseAnalysisDB(BaseModel):
    melid: int
    start_note_index: int
    end_note_index: int
    phrase_value: str
    processed_chords_json: str
    features_json: str

    class Config:
        from_attributes = True


class PatternDB(BaseModel):
    pattern_name: str
    key: str
    chords_json: str
    features_json: str

    class Config:
        from_attributes = True
