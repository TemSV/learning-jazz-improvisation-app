import sqlite3
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List

from ..schemas import NoteData
from ..dependencies import get_db_path

router = APIRouter(
    prefix="/api/phrases",
    tags=["Phrases"],
)

# Database path will be injected by dependency
DB_PATH = Depends(get_db_path)

@router.get("/{melid}/notes", response_model=List[NoteData])
async def get_phrase_notes(
    melid: int,
    start_note_index: int = Query(..., ge=0, description="Start note index (inclusive) for the phrase within the melody."),
    end_note_index: int = Query(..., ge=0, description="End note index (exclusive) for the phrase within the melody."),
    db_path: str = DB_PATH
):
    """
    Retrieves the notes for a specific phrase segment of a melody.
    Notes are selected based on their 0-based index in the melody, ordered by onset.
    """
    if start_note_index >= end_note_index:
        raise HTTPException(status_code=400, detail="start_note_index must be less than end_note_index")

    notes_data = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetch all notes for the melid, ordered by onset, to correctly apply slicing by index
        cursor.execute("""
            SELECT pitch, onset, duration, loud_med, f0_med_dev
            FROM melody
            WHERE melid = ?
            ORDER BY onset ASC
        """, (melid,))
        all_melody_notes = cursor.fetchall()

        if not all_melody_notes:
            raise HTTPException(status_code=404, detail=f"No notes found for melid {melid}")

        actual_end_index = min(end_note_index, len(all_melody_notes))

        if start_note_index >= len(all_melody_notes) or start_note_index >= actual_end_index:
            return []

        selected_notes_raw = all_melody_notes[start_note_index:actual_end_index]

        for row in selected_notes_raw:
            notes_data.append(NoteData(
                pitch=row['pitch'],
                onset=row['onset'],
                duration=row['duration'],
                loud_med=row['loud_med'],
                f0_med_dev=row['f0_med_dev']
            ))

    except sqlite3.Error as e:
        print(f"Database error in get_phrase_notes: {e}")
        raise HTTPException(status_code=500, detail=f"Database error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error in get_phrase_notes: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    if not notes_data and not selected_notes_raw:
        pass

    return notes_data
