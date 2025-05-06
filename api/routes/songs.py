import sqlite3
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Optional

from ..schemas import (
    SongListResponse, SongChordsResponse, SongInfo, SongChordEntry,
    PatternInfo, SongPatternsResponse, ChordDuration
)

from ..dependencies import get_db_path, get_chord_parser, get_pattern_analyzer
from core.pattern_analysis.parser import DatabaseChordParser
from core.pattern_analysis.pattern_analyzer import PatternAnalyzer
from core.pattern_analysis.models import ChordWithDuration as ModelChordWithDuration
import core.pattern_analysis.models

router = APIRouter(
    prefix="/api/songs",
    tags=["Songs"],
)

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
            params_list.append(f"%{search}%")

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
            FROM song_chords WHERE songid = ? ORDER BY barid ASC
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
        chords_by_bar_raw = parser.parse_song_chords(song_id)
        if not chords_by_bar_raw:
            if not song_title:
                 raise HTTPException(status_code=404, detail=f"Song with id {song_id} not found.")
            return SongPatternsResponse(
                song_id=song_id, title=song_title,
                processed_song_chords=[], patterns=[]
            )

        processed_song_chords_list: List[ModelChordWithDuration] = analyzer._process_chord_sequence(chords_by_bar_raw)
        found_patterns_raw: List[core.pattern_analysis.models.ChordPattern] = analyzer.find_patterns(chords_by_bar_raw)

        patterns_response: List[PatternInfo] = []
        for p_model in found_patterns_raw:
            start_idx = p_model.start_bar
            end_idx = start_idx + len(p_model.chords) - 1
            if start_idx < len(processed_song_chords_list) and end_idx < len(processed_song_chords_list):
                patterns_response.append(
                    PatternInfo(
                        type=p_model.pattern_type, key=p_model.key,
                        start_index=start_idx, end_index=end_idx,
                        chords=[ChordDuration(chord=c.chord, duration=c.duration) for c in p_model.chords],
                        features=p_model.features
                    )
                )
            else:
                print(f"Warning: Pattern indices [{start_idx}-{end_idx}] out of bounds for processed song chords (len {len(processed_song_chords_list)}) for song {song_id}, pattern {p_model.pattern_type}")

        processed_chords_for_response = [
            ChordDuration(chord=c.chord, duration=c.duration) for c in processed_song_chords_list
        ]

        return SongPatternsResponse(
            song_id=song_id,
            title=song_title,
            processed_song_chords=processed_chords_for_response,
            patterns=patterns_response
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error analyzing patterns for song id {song_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error analyzing song patterns")
