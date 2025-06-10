import sqlite3
import json
import time
from typing import List, Dict, Tuple, Any

from core.pattern_analysis.harmony_analyzer import HarmonyAnalyzer
from core.pattern_analysis.phrase_manager import PhraseManager, PhraseInfo
from core.pattern_analysis.models import ChordWithDuration


DB_PATH = r"C:\polytech\Diploma\wjazzd.db"
INTERVAL_WEIGHT = 1.5
CHORD_TYPE_WEIGHT = 2.0
CHORD_DURATION_WEIGHT = 2.0
COMMIT_INTERVAL = 500

def serialize_processed_chords(chords: List[ChordWithDuration]) -> str:
    serializable_list = [{'chord': c.chord, 'duration': c.duration} for c in chords]
    return json.dumps(serializable_list)

def main():
    print("Starting phrase preprocessing...")
    start_script_time = time.time()

    print("Initializing components...")
    analyzer = HarmonyAnalyzer()
    phrase_mgr = PhraseManager(DB_PATH, analyzer)

    conn = None
    try:
        print(f"Connecting to database: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            print("Enabled WAL journaling mode.")
        except sqlite3.Error as e:
            print(f"Could not enable WAL mode: {e}")

        cursor = conn.cursor()

        print("Clearing previous phrase analysis data...")
        cursor.execute("DELETE FROM phrase_analysis")
        conn.commit()

        print("Fetching all phrase definitions from 'sections' table...")
        all_phrases = phrase_mgr.get_all_phrases()
        total_phrases = len(all_phrases)
        print(f"Found {total_phrases} phrase definitions.")

        if not all_phrases:
            print("No phrases found to process.")
            return

        skipped_onset_count = 0
        skipped_chords_count = 0
        skipped_features_count = 0
        skipped_serialize_count = 0

        print("Processing phrases and calculating features...")
        processed_count = 0
        skipped_count = 0
        insert_errors = 0

        for i, phrase_info in enumerate(all_phrases):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total_phrases} phrases...")

            if phrase_info.start_onset is None or phrase_info.end_onset is None:
                skipped_onset_count += 1
                continue
            processed_chords = phrase_mgr.get_processed_phrase_chords(phrase_info)
            if not processed_chords:
                skipped_chords_count += 1
                continue
            features_dict = phrase_mgr.compute_phrase_features(
                processed_chords,
                interval_weight=INTERVAL_WEIGHT,
                chord_type_weight=CHORD_TYPE_WEIGHT,
                chord_duration_weight=CHORD_DURATION_WEIGHT
            )
            if not features_dict:
                skipped_features_count += 1
                continue
            try:
                chords_json = serialize_processed_chords(processed_chords)
                features_json = json.dumps(features_dict)
            except Exception as e:
                print(f"Error serializing data for phrase MelID {phrase_info.melid} Indices {phrase_info.start_note_index}-{phrase_info.end_note_index}: {e}")
                skipped_serialize_count += 1
                continue

            data_to_insert = (
                phrase_info.melid,
                phrase_info.start_note_index,
                phrase_info.end_note_index,
                phrase_info.value,
                chords_json,
                features_json
            )

            try:
                cursor.execute("""
                    INSERT INTO phrase_analysis (
                        melid, start_note_index, end_note_index, phrase_value,
                        processed_chords_json, features_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, data_to_insert)
                processed_count += 1
            except sqlite3.Error as e:
                print(f"Database error inserting data for phrase MelID {phrase_info.melid}: {e}")
                insert_errors += 1

            if (i + 1) % COMMIT_INTERVAL == 0:
                print(f"  Committing after processing {i + 1}/{total_phrases} phrases...")
                conn.commit()
                print(f"  Commit successful.")

        print("Committing final changes to the database...")
        conn.commit()
        print("Final commit successful.")

        end_script_time = time.time()
        print("\n--- Preprocessing Summary ---")
        print(f"Total phrase definitions found: {total_phrases}")
        print(f"Successfully processed and stored: {processed_count}")
        total_skipped = skipped_onset_count + skipped_chords_count + skipped_features_count + skipped_serialize_count
        print(f"Skipped (no onsets, chords, or features): {total_skipped}")
        print(f"Database insertion errors: {insert_errors}")
        print(f"Total time: {end_script_time - start_script_time:.2f} seconds.")

        print("\n--- Skipping Breakdown ---")
        print(f"Skipped due to onset errors: {skipped_onset_count}")
        print(f"Skipped due to chord processing issues: {skipped_chords_count}")
        print(f"Skipped due to feature calculation issues: {skipped_features_count}")
        print(f"Skipped due to serialization errors: {skipped_serialize_count}")
        print(f"Total skipped (calculated): {total_skipped}")

    except sqlite3.Error as e:
        print(f"An error occurred during database operations: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
