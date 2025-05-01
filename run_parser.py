import time
from core.pattern_analysis.parser import DatabaseChordParser
from core.pattern_analysis.pattern_analyzer import PatternAnalyzer
from core.pattern_analysis.phrase_manager import PhraseManager, PhraseInfo # Import PhraseInfo
from core.pattern_analysis.models import ChordPattern, ChordWithDuration # Ensure ChordWithDuration is imported

# --- Configuration ---
DB_PATH = r"C:\polytech\Diploma\wjazzd.db" # Use raw string for path
SONG_ID_TO_ANALYZE = 1
TOP_N_RECOMMENDATIONS = 5 # Number of phrases to recommend per pattern

def main():
    print("Initializing components...")
    start_time = time.time()
    parser = DatabaseChordParser(DB_PATH)
    analyzer = PatternAnalyzer()
    # PhraseManager now requires PatternAnalyzer
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

    # --- 2. Prepare Phrases ---
    print("\nLoading and processing all phrases...")
    all_phrases_info = phrase_mgr.get_all_phrases()
    if not all_phrases_info:
        print("No phrases found in the database.")
        return

    phrase_features_list = []
    processed_phrase_count = 0
    skipped_phrase_count = 0
    for phrase in all_phrases_info:
        # Ensure phrase onsets were calculated
        if phrase.start_onset is None or phrase.end_onset is None:
            # print(f"Skipping phrase (no onset): MelID {phrase.melid}, Indices {phrase.start_note_index}-{phrase.end_note_index}")
            skipped_phrase_count += 1
            continue

        processed_chords = phrase_mgr.get_processed_phrase_chords(phrase)
        if processed_chords:
            features = phrase_mgr.compute_phrase_features(processed_chords)
            if features: # Ensure features were actually computed
                 # Store tuple (PhraseInfo, features_dict, processed_chords)
                 # Storing processed_chords here avoids recalculating later
                 phrase_features_list.append((phrase, features, processed_chords))
                 processed_phrase_count += 1
            else:
                 # print(f"Skipping phrase (no features): MelID {phrase.melid}, Indices {phrase.start_note_index}-{phrase.end_note_index}")
                 skipped_phrase_count += 1
        else:
            # print(f"Skipping phrase (no chords): MelID {phrase.melid}, Indices {phrase.start_note_index}-{phrase.end_note_index}")
            skipped_phrase_count += 1

    process_phrases_time = time.time()
    print(f"Phrase processing took {process_phrases_time - analyze_song_time:.2f} seconds.")
    print(f"Processed {processed_phrase_count} phrases with features (skipped {skipped_phrase_count}).")

    if not phrase_features_list:
        print("No phrases could be processed to generate features.")
        return

    # --- 3. Compare Patterns and Phrases & 4. Recommend ---
    print(f"\n--- Recommendations for Song ID: {SONG_ID_TO_ANALYZE} ---")
    for i, pattern in enumerate(song_patterns):
        pattern_chords_str = " -> ".join([c.chord for c in pattern.chords])
        print(f"\n--- Pattern #{i+1} ---")
        print(f"  Type:       {pattern.pattern_type}")
        print(f"  Key (Info): {pattern.key}") # Key is just informational now
        print(f"  Chords:     {pattern_chords_str}")
        print(f"  Start Index:{pattern.start_bar}")
        print(f"  Duration:   {pattern.total_duration:.2f} beats")
        # print(f"  Features:   {pattern.features}") # Optional: print features for debugging

        if not pattern.features:
            print("  (Pattern has no features to compare)")
            continue

        # Calculate similarity to all processed phrases
        pattern_similarities = []
        # Iterate through the extended list of tuples
        for phrase_info, phrase_features, phrase_proc_chords in phrase_features_list:
            if not phrase_features: # Should not happen based on previous check, but belt-and-suspenders
                continue

            # Use the compute_similarity method (available in PhraseManager)
            similarity = phrase_mgr.compute_similarity(pattern.features, phrase_features)
            # Store tuple (phrase_object, similarity_score, processed_chords)
            pattern_similarities.append((phrase_info, similarity, phrase_proc_chords))

        # Sort phrases by similarity score in descending order
        pattern_similarities.sort(key=lambda item: item[1], reverse=True)

        # Print top N recommendations for this pattern
        print(f"\n  Top {TOP_N_RECOMMENDATIONS} similar phrases:")
        if not pattern_similarities:
            print("    No similar phrases found.")
        else:
            # Take top N, or fewer if less were found
            num_to_show = min(TOP_N_RECOMMENDATIONS, len(pattern_similarities))
            # Unpack the tuple including processed chords
            for phrase_rec, score, processed_chords_rec in pattern_similarities[:num_to_show]:
                 # phrase_rec is a PhraseInfo object
                 print(f"    - MelID: {phrase_rec.melid: <4} "
                       f"Indices: {phrase_rec.start_note_index}-{phrase_rec.end_note_index} "
                       # f"Value: '{phrase_rec.value}' " # Value might be long, optional
                       f"(Similarity: {score:.4f})")
                 # Format and print the chords from the processed list
                 if processed_chords_rec:
                     chords_repr = " -> ".join([f"{c.chord} ({c.duration:.1f})" for c in processed_chords_rec])
                     print(f"      Chords: {chords_repr}")
                 else:
                      print("      Chords: (Not available)") # Fallback
        print("-" * 40)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()