import sqlite3
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Optional, Tuple

from ..schemas import (
    SongListResponse, SongChordsResponse, SongInfo, SongChordEntry,
    PatternInfo, SongPatternsResponse, ChordDuration, OriginalChordRef
)

from ..dependencies import get_db_path, get_chord_parser, get_pattern_analyzer
from core.pattern_analysis.parser import DatabaseChordParser, SongChord
from core.pattern_analysis.pattern_analyzer import PatternAnalyzer
from core.pattern_analysis.models import ChordWithDuration as ModelChordWithDuration, ChordPattern
import core.pattern_analysis.models

router = APIRouter(
    prefix="/api/songs",
    tags=["Songs"],
)

def get_beats_per_bar(signature: Optional[str]) -> int:
    if signature:
        if signature == "4/4": return 4
        if signature == "3/4": return 3
        if signature == "2/4": return 2
    return 4 

@router.get("", response_model=SongListResponse)
async def get_song_list(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: Optional[str] = Query(default=None)
):
    """Retrieves a list of available songs with titles."""
    songs = []
    total = 0
    conn = None
    current_db_path = get_db_path()

    try:
        conn = sqlite3.connect(current_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        count_query = "SELECT COUNT(songid) FROM song"
        base_query = "SELECT songid, title FROM song"
        params_list = []

        if search:
            where_clause = " WHERE title LIKE ?"
            count_query += where_clause
            base_query += where_clause
            params_list.append(f"%{search}%\u0025")

        cursor.execute(count_query, params_list)
        count_result = cursor.fetchone()
        total = count_result[0] if count_result else 0

        query = f"{base_query} ORDER BY title, songid LIMIT ? OFFSET ?"
        params_list.extend([limit, offset])
        cursor.execute(query, params_list)
        rows = cursor.fetchall()
        songs = [SongInfo(id=row["songid"], title=row["title"]) for row in rows]

    except sqlite3.Error as e:
        print(f"Database error fetching song list: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching song list")
    except Exception as e:
        print(f"Unexpected error fetching song list: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if conn: conn.close()
    return SongListResponse(songs=songs, total=total)


@router.get("/{song_id}/chords", response_model=SongChordsResponse)
async def get_song_chords(
    song_id: int
):
    """Retrieves the chord progression for a specific song directly from song_chords table."""
    print(f"Getting raw chords for song_id: {song_id}")
    chord_entries = []
    song_title: str | None = None
    conn = None
    current_db_path = get_db_path()

    try:
        conn = sqlite3.connect(current_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT title FROM song WHERE songid = ?", (song_id,))
        title_row = cursor.fetchone()
        if title_row: song_title = title_row["title"]

        cursor.execute("""
            SELECT barid, bar, signature, chords, form
            FROM song_chords WHERE songid = ? ORDER BY bar ASC
        """, (song_id,))
        rows = cursor.fetchall()

        if not rows and not song_title:
            raise HTTPException(status_code=404, detail=f"Song with id {song_id} not found.")

        chord_entries = [
            SongChordEntry(barid=row["barid"], bar=row["bar"], signature=row["signature"],
                           chords=row["chords"], form=row["form"]) for row in rows
        ]
    except sqlite3.Error as e:
        print(f"Database error fetching song chords for id {song_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching song chords")
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        print(f"Unexpected error fetching song chords for id {song_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if conn: conn.close()
    return SongChordsResponse(song_id=song_id, title=song_title, chords=chord_entries)


@router.get("/{song_id}/patterns", response_model=SongPatternsResponse)
async def get_song_patterns(
    song_id: int,
    parser: DatabaseChordParser = Depends(get_chord_parser),
    analyzer: PatternAnalyzer = Depends(get_pattern_analyzer)
):
    """Analyzes song chords and returns detected patterns."""
    print(f"Analyzing patterns for song_id: {song_id}")
    song_title: str | None = None
    conn = None
    current_db_path_for_title = get_db_path()

    # 1. Get song title
    try:
        conn = sqlite3.connect(current_db_path_for_title)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM song WHERE songid = ?", (song_id,))
        title_row = cursor.fetchone()
        if title_row:
            song_title = title_row["title"]
    except sqlite3.Error as e:
        print(f"Database error fetching song title for patterns endpoint (song_id {song_id}): {e}")
    finally:
        if conn:
            conn.close()

    # 2. Parse song chords and analyze patterns using dependencies
    try:
        raw_song_chords_data: List[SongChord] = parser.get_song_chords(song_id)
        if not raw_song_chords_data:
            if not song_title:
                 raise HTTPException(status_code=404, detail=f"Song with id {song_id} not found.")
            return SongPatternsResponse(
                song_id=song_id, title=song_title, patterns=[]
            )

        parsed_chords_for_analysis: List[Tuple[int, List[str]]] = parser.parse_song_chords(song_id)
        found_patterns_raw: List[ChordPattern] = analyzer.find_patterns(parsed_chords_for_analysis)

        patterns_response: List[PatternInfo] = []
        for p_model in found_patterns_raw:
            original_refs_for_pattern: List[OriginalChordRef] = []
            
            if p_model.start_bar < 0 or p_model.start_bar >= len(parsed_chords_for_analysis):
                print(f"Warning: Pattern start_bar index {p_model.start_bar} out of bounds for parsed_chords_for_analysis (len {len(parsed_chords_for_analysis)}) for song {song_id}, pattern {p_model.pattern_type}")
                continue

            start_original_bar_number = parsed_chords_for_analysis[p_model.start_bar][0]
            
            initial_raw_bar_idx = -1
            for i, r_bar in enumerate(raw_song_chords_data):
                if r_bar.bar == start_original_bar_number:
                    initial_raw_bar_idx = i
                    break
            
            if initial_raw_bar_idx == -1:
                print(f"Warning: Could not find starting raw bar for pattern in song {song_id}, pattern {p_model.pattern_type}, start_bar_num {start_original_bar_number}")
                continue

            current_raw_bar_idx = initial_raw_bar_idx
            beats_consumed_in_current_raw_bar = 0.0

            for pattern_norm_chord in p_model.chords:
                beats_to_assign_for_this_pattern_chord = float(pattern_norm_chord.duration)
                
                while beats_to_assign_for_this_pattern_chord > 1e-6 and current_raw_bar_idx < len(raw_song_chords_data):
                    raw_bar = raw_song_chords_data[current_raw_bar_idx]
                    actual_beats_in_raw_bar = float(get_beats_per_bar(raw_bar.signature))
                    
                    if not original_refs_for_pattern or original_refs_for_pattern[-1].barid != raw_bar.barid:
                        original_refs_for_pattern.append(OriginalChordRef(barid=raw_bar.barid))

                    beats_available_to_consume_in_raw_bar = actual_beats_in_raw_bar - beats_consumed_in_current_raw_bar
                    beats_to_consume_this_iteration = min(beats_to_assign_for_this_pattern_chord, beats_available_to_consume_in_raw_bar)

                    beats_to_assign_for_this_pattern_chord -= beats_to_consume_this_iteration
                    beats_consumed_in_current_raw_bar += beats_to_consume_this_iteration

                    if beats_consumed_in_current_raw_bar >= actual_beats_in_raw_bar - 1e-6:
                        current_raw_bar_idx += 1
                        beats_consumed_in_current_raw_bar = 0.0
            
            if original_refs_for_pattern:
                patterns_response.append(
                    PatternInfo(
                        type=p_model.pattern_type, key=p_model.key,
                        original_chord_refs=original_refs_for_pattern,
                        chords=[ChordDuration(chord=c.chord, duration=c.duration) for c in p_model.chords],
                        features=p_model.features
                    )
                )

        return SongPatternsResponse(
            song_id=song_id,
            title=song_title,
            patterns=patterns_response
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error analyzing patterns for song id {song_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error analyzing song patterns")
