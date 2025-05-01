import sqlite3
import json
import time
from typing import List, Dict, Tuple, Any # Добавили Any

# Предполагаем, что эти классы импортируются корректно
from core.pattern_analysis.pattern_analyzer import PatternAnalyzer
from core.pattern_analysis.phrase_manager import PhraseManager, PhraseInfo
from core.pattern_analysis.models import ChordWithDuration # Нужен для типизации

# --- Конфигурация ---
DB_PATH = r"C:\polytech\Diploma\wjazzd.db"
# Вес для интервалов (должен совпадать с используемым при анализе паттернов)
INTERVAL_WEIGHT = 3.0
COMMIT_INTERVAL = 500 # Коммитить каждые 500 фраз (можно настроить)

def serialize_processed_chords(chords: List[ChordWithDuration]) -> str:
    """Преобразует список ChordWithDuration в JSON-строку."""
    # Преобразуем каждый объект в словарь перед сериализацией
    serializable_list = [{'chord': c.chord, 'duration': c.duration} for c in chords]
    return json.dumps(serializable_list)

def main():
    print("Starting phrase preprocessing...")
    start_script_time = time.time()

    print("Initializing components...")
    analyzer = PatternAnalyzer()
    phrase_mgr = PhraseManager(DB_PATH, analyzer)

    conn = None
    try:
        print(f"Connecting to database: {DB_PATH}")
        conn = sqlite3.connect(DB_PATH, timeout=10.0) # Увеличим таймаут на всякий случай
        # Включим WAL режим для лучшей параллельности чтения/записи (рекомендуется)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            print("Enabled WAL journaling mode.")
        except sqlite3.Error as e:
            print(f"Could not enable WAL mode: {e}")

        cursor = conn.cursor()

        # 1. Очистить таблицу перед заполнением (простой вариант)
        # В продакшене может потребоваться более сложная логика обновления
        print("Clearing previous phrase analysis data...")
        cursor.execute("DELETE FROM phrase_analysis")
        conn.commit() # Важно закоммитить удаление перед вставкой

        # 2. Получить все определения фраз
        print("Fetching all phrase definitions from 'sections' table...")
        all_phrases = phrase_mgr.get_all_phrases()
        total_phrases = len(all_phrases)
        print(f"Found {total_phrases} phrase definitions.")

        if not all_phrases:
            print("No phrases found to process.")
            return

        # Внутри main() перед циклом
        skipped_onset_count = 0
        skipped_chords_count = 0
        skipped_features_count = 0
        skipped_serialize_count = 0

        # 3. Обработать каждую фразу и сохранить результаты
        print("Processing phrases and calculating features...")
        processed_count = 0
        skipped_count = 0
        insert_errors = 0

        for i, phrase_info in enumerate(all_phrases):
            if (i + 1) % 100 == 0: # Печатать прогресс каждые 100 фраз
                print(f"  Processed {i + 1}/{total_phrases} phrases...")

            if phrase_info.start_onset is None or phrase_info.end_onset is None:
                skipped_onset_count += 1 # <--- Счетчик 1
                continue
            processed_chords = phrase_mgr.get_processed_phrase_chords(phrase_info)
            if not processed_chords:
                skipped_chords_count += 1 # <--- Счетчик 2
                continue
            features_dict = phrase_mgr.compute_phrase_features(processed_chords, interval_weight=INTERVAL_WEIGHT)
            if not features_dict:
                skipped_features_count += 1 # <--- Счетчик 3
                continue
            try:
                chords_json = serialize_processed_chords(processed_chords)
                features_json = json.dumps(features_dict)
            except Exception as e:
                print(f"Error serializing data for phrase MelID {phrase_info.melid} Indices {phrase_info.start_note_index}-{phrase_info.end_note_index}: {e}")
                skipped_serialize_count += 1 # <--- Счетчик 4
                continue

            # Подготовка данных для вставки
            data_to_insert = (
                phrase_info.melid,
                phrase_info.start_note_index,
                phrase_info.end_note_index,
                phrase_info.value, # Получаем value из PhraseInfo
                chords_json,
                features_json
            )

            # Вставка в БД
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

            # Периодический коммит
            if (i + 1) % COMMIT_INTERVAL == 0:
                print(f"  Committing after processing {i + 1}/{total_phrases} phrases...")
                conn.commit()
                print(f"  Commit successful.")

        # 4. Финальный коммит для оставшихся записей
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
        print(f"Total skipped (calculated): {total_skipped}") # Сравнить с общим skipped_count

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
