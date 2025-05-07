from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional

# --- Basic Data Structures ---

class ChordDuration(BaseModel):
    chord: str
    duration: float

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
    start_index: int # Index in the *processed* chord list of the song
    end_index: int   # Inclusive end index in the *processed* chord list
    chords: List[ChordDuration] # Chords of the pattern itself
    features: Dict[str, float] = Field(description="Features dict for this pattern instance")

class SongPatternsResponse(BaseModel):
    song_id: int
    title: str | None = None
    # Processed chords are useful context for start_index/end_index of patterns
    processed_song_chords: List[ChordDuration]
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
