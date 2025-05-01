import sqlite3
import json
import time
from typing import List, Dict, Tuple, Any # Добавили Any

# Импорты ваших классов
from core.pattern_analysis.parser import DatabaseChordParser
from core.pattern_analysis.pattern_analyzer import PatternAnalyzer
from core.pattern_analysis.phrase_manager import PhraseManager, PhraseInfo
from core.pattern_analysis.models import ChordPattern, ChordWithDuration

# --- Конфигурация ---
DB_PATH = r"C:\polytech\Diploma\wjazzd.db"
SONG_ID_TO_ANALYZE = 3
TOP_N_RECOMMENDATIONS = 5

def main():
    print("Initializing components...")
    start_time = time.time()
    parser = DatabaseChordParser(DB_PATH)
    analyzer = PatternAnalyzer()
    # PhraseManager все еще нужен для compute_similarity
    phrase_mgr = PhraseManager(DB_PATH, analyzer)
    init_time = time.time()
    print(f"Initialization took {init_time - start_time:.2f} seconds.")

    # --- 1. Analyze the Song ---
    print(f"\nAnalyzing song ID: {SONG_ID_TO_ANALYZE}...")
    song_chords_by_bar = parser.parse_song_chords(SONG_ID_TO_ANALYZE)
    if not song_chords_by_bar:
        print(f"No chords found or parsed for song ID {SONG_ID_TO_ANALYZE}")
        return

    song_patterns = analyzer.find_patterns(song_chords_by_bar)
    analyze_song_time = time.time()
    print(f"Song analysis took {analyze_song_time - init_time:.2f} seconds.")

    if not song_patterns:
        print("No patterns found in the song.")
        return
    print(f"Found {len(song_patterns)} patterns in the song.")

    # --- 2. Load Preprocessed Phrase Data ---
    print("\nLoading preprocessed phrase features...")
    preprocessed_phrases = [] # List to store (melid, start_idx, end_idx, features_dict)
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT melid, start_note_index, end_note_index, features_json
            FROM phrase_analysis
        """)
        rows = cursor.fetchall()
        load_start_time = time.time()

        skipped_deserialize = 0
        for row in rows:
            melid, start_idx, end_idx, features_json_str = row
            try:
                if features_json_str:
                    features_dict = json.loads(features_json_str)
                    preprocessed_phrases.append( (melid, start_idx, end_idx, features_dict) )
                else:
                     skipped_deserialize += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding features JSON for MelID {melid}, Indices {start_idx}-{end_idx}: {e}")
                skipped_deserialize += 1

        load_end_time = time.time()
        print(f"Loaded {len(preprocessed_phrases)} preprocessed phrases (skipped {skipped_deserialize} due to missing/bad JSON).")
        print(f"Loading took {load_end_time - load_start_time:.2f} seconds.")

    except sqlite3.Error as e:
        print(f"Database error loading preprocessed phrases: {e}")
        return # Не можем продолжать без данных фраз
    finally:
        if conn:
            conn.close()

    if not preprocessed_phrases:
        print("No preprocessed phrase features found in the database. Run preprocess_phrases.py first.")
        return

    # --- 3. Compare Patterns and Phrases & 4. Recommend ---
    print(f"\n--- Recommendations for Song ID: {SONG_ID_TO_ANALYZE} ---")
    recommend_start_time = time.time()

    for i, pattern in enumerate(song_patterns):
        pattern_chords_str = " -> ".join([c.chord for c in pattern.chords])
        print(f"\n--- Pattern #{i+1} ---")
        print(f"  Type:       {pattern.pattern_type}")
        print(f"  Key (Info): {pattern.key}")
        print(f"  Chords:     {pattern_chords_str}")
        print(f"  Start Index:{pattern.start_bar}")
        print(f"  Duration:   {pattern.total_duration:.2f} beats")
        # print(f"  Features:   {pattern.features}") # Optional: print features for debugging

        if not pattern.features:
            print("  (Pattern has no features to compare)")
            continue

        pattern_similarities = []
        # Итерация по предзагруженным данным
        for melid, start_idx, end_idx, phrase_features in preprocessed_phrases:
            if not phrase_features: # Дополнительная проверка
                continue

            similarity = phrase_mgr.compute_similarity(pattern.features, phrase_features)
            # Сохраняем идентификаторы фразы и сходство
            pattern_similarities.append( (melid, start_idx, end_idx, similarity) )

        pattern_similarities.sort(key=lambda item: item[3], reverse=True) # Сортировка по similarity

        print(f"\n  Top {TOP_N_RECOMMENDATIONS} similar phrases:")
        if not pattern_similarities:
            print("    No similar phrases found.")
        else:
            num_to_show = min(TOP_N_RECOMMENDATIONS, len(pattern_similarities))
            # Распаковка кортежа с идентификаторами
            for melid_rec, start_idx_rec, end_idx_rec, score in pattern_similarities[:num_to_show]:
                 print(f"    - MelID: {melid_rec: <4} "
                       f"Indices: {start_idx_rec}-{end_idx_rec} "
                       f"(Similarity: {score:.4f})")
                 # Вывод аккордов теперь не делаем, так как их нет в памяти
                 # Чтобы вывести аккорды, нужен будет доп. запрос к phrase_analysis по ID/индексам

        print("-" * 40)

    recommend_end_time = time.time()
    print(f"\nRecommendation phase took: {recommend_end_time - recommend_start_time:.2f} seconds.")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()