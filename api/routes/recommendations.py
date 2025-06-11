import sqlite3
import json
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict

from ..schemas import (
    RecommendationRequest, RecommendationResponse, RecommendedPhrase, ChordDuration
)

from ..dependencies import get_phrase_manager, get_db_path
from core.pattern_analysis.phrase_manager import PhraseManager
from core.utils.similarity_utils import calculate_cosine_similarity

router = APIRouter(
    prefix="/api/recommendations",
    tags=["Recommendations"],
)

def deserialize_processed_chords(chords_json_str: str | None) -> List[ChordDuration]:
    """Deserializes JSON string back to List[ChordDuration]. Handles None input."""
    if not chords_json_str:
        return []
    try:
        chords_list_of_dicts = json.loads(chords_json_str)
        return [ChordDuration(**c) for c in chords_list_of_dicts]
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error decoding chords JSON: {e} - Data: {chords_json_str[:100]}...")
        return []


@router.post("/phrases", response_model=RecommendationResponse)
async def get_phrase_recommendations(
    request_data: RecommendationRequest = Body(...),
    db_path: str = Depends(get_db_path)
):
    """
    Calculates and returns phrase recommendations based on pattern features
    by comparing with preprocessed phrases in the database.
    """
    pattern_features = request_data.pattern_features
    limit = request_data.limit

    if not pattern_features:
        raise HTTPException(status_code=400, detail="Pattern features cannot be empty.")

    preprocessed_phrases_data = []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pa.melid, pa.start_note_index, pa.end_note_index, pa.features_json, pa.processed_chords_json,
                   si.performer, si.title
            FROM phrase_analysis pa
            JOIN solo_info si ON pa.melid = si.melid
        """)
        rows = cursor.fetchall()

        skipped_deserialize = 0
        for row in rows:
            try:
                if row[3]:
                    features_dict = json.loads(row[3])
                    preprocessed_phrases_data.append({
                        "melid": row[0],
                        "start_note_index": row[1],
                        "end_note_index": row[2],
                        "features": features_dict,
                        "chords_json": row[4],
                        "performer": row[5],
                        "title": row[6]
                    })
                else:
                    skipped_deserialize +=1
            except json.JSONDecodeError:
                 skipped_deserialize += 1

        if skipped_deserialize > 0:
             print(f"Skipped {skipped_deserialize} phrases due to JSON decoding errors or missing features.")

    except sqlite3.Error as e:
        print(f"Database error loading preprocessed phrases: {e}")
        raise HTTPException(status_code=500, detail="Database error loading phrase data")
    except Exception as e:
        print(f"Unexpected error loading preprocessed phrases: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if conn:
            conn.close()

    if not preprocessed_phrases_data:
        print("Warning: No valid preprocessed phrase data found in DB.")
        return RecommendationResponse(recommendations=[])

    all_similarities = []
    calculation_errors = 0
    for phrase_data in preprocessed_phrases_data:
        if not phrase_data.get("features"):
            calculation_errors += 1
            continue
        try:
            similarity = calculate_cosine_similarity(pattern_features, phrase_data["features"])
            all_similarities.append({
                "melid": phrase_data["melid"],
                "start_note_index": phrase_data["start_note_index"],
                "end_note_index": phrase_data["end_note_index"],
                "similarity": similarity,
                "chords_json": phrase_data["chords_json"],
                "performer": phrase_data["performer"],
                "title": phrase_data["title"]
            })
        except Exception as e:
            print(f"Error computing similarity for MelID {phrase_data.get('melid')}: {e}")
            calculation_errors += 1

    if calculation_errors > 0:
        print(f"Encountered {calculation_errors} errors during similarity calculation.")

    all_similarities.sort(key=lambda item: item["similarity"], reverse=True)

    top_recommendations = []
    num_to_show = min(limit, len(all_similarities))

    for i in range(num_to_show):
        rec_data = all_similarities[i]
        deserialized_chords = deserialize_processed_chords(rec_data["chords_json"])

        top_recommendations.append(RecommendedPhrase(
            melid=rec_data["melid"],
            start_note_index=rec_data["start_note_index"],
            end_note_index=rec_data["end_note_index"],
            similarity=rec_data["similarity"],
            chords=deserialized_chords,
            performer=rec_data["performer"],
            title=rec_data["title"]
        ))

    return RecommendationResponse(recommendations=top_recommendations)
