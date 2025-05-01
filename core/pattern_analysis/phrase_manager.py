import sqlite3
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import math

from .pattern_analyzer import PatternAnalyzer
from .models import ChordPattern, ChordWithDuration, BeatInfo, PhraseSectionInfo, PhraseInfo



class PhraseManager:
    """Manages retrieval and processing of musical phrases."""

    def __init__(self, db_path: str, pattern_analyzer: PatternAnalyzer):
        """
        Initializes the PhraseManager.

        Args:
            db_path: Path to the SQLite database file.
            pattern_analyzer: An instance of PatternAnalyzer for analyzing chords.
        """
        self.db_path = db_path
        self.pattern_analyzer = pattern_analyzer

    def _get_phrase_onset_range(self, melid: int, start_note_index: int, end_note_index: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Finds the start and end onset times for a phrase based on note indices.
        Assumes start/end indices refer to 0-based order of notes within the melody,
        sorted by onset time. End index is treated as exclusive.
        """
        start_onset, end_onset = None, None
        notes = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Get all note onsets and durations for the melody, ordered by onset
                cursor.execute("""
                    SELECT onset, duration
                    FROM melody
                    WHERE melid = ?
                    ORDER BY onset ASC
                """, (melid,))
                notes = cursor.fetchall()

                if not notes:
                    return None, None

                # Clamp indices to valid range
                start_idx = max(0, start_note_index)
                # Use end_note_index directly as exclusive upper bound
                end_idx = end_note_index

                if start_idx < len(notes) and start_idx < end_idx:
                    start_onset = notes[start_idx][0]
                else:
                    return None, None

                # Determine end onset:
                if end_idx < len(notes):
                    end_onset = notes[end_idx][0]
                elif end_idx == len(notes) and len(notes) > 0 :
                    last_included_note_index = end_idx - 1
                    if last_included_note_index < 0: return None, None
                    last_note_onset, last_note_duration = notes[last_included_note_index]
                    end_onset = last_note_onset + last_note_duration
                else:
                     if not notes: return None, None
                     last_note_onset, last_note_duration = notes[-1]
                     end_onset = last_note_onset + last_note_duration


                # Sanity check: ensure end_onset is not before start_onset
                if end_onset is not None and start_onset is not None and end_onset < start_onset:
                    end_onset = start_onset

        except sqlite3.Error as e:
            print(f"Database error getting phrase onset range for melid {melid}: {e}")
            return None, None
        except IndexError:
            return None, None

        return start_onset, end_onset

    def get_all_phrases(self, melid: Optional[int] = None) -> List[PhraseInfo]:
        """
        Retrieves phrase definitions from the 'sections' table, filtering by type='phrase'.
        Optionally filters further by melid.
        Populates the start_onset and end_onset fields for each phrase.
        """
        phrases_info = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # The query targets type='phrase'
                query = "SELECT melid, type, start, end, value FROM sections WHERE type = 'PHRASE'"
                params: tuple = ()
                if melid is not None:
                    query += " AND melid = ?"
                    params = (melid,)
                cursor.execute(query, params)
                rows = cursor.fetchall()

                for i, row in enumerate(rows):
                    try:
                        melid_val, type_val, start_val, end_val, value_val = row
                        phrase = PhraseInfo(
                            melid=int(melid_val), phrase_type=str(type_val),
                            start_note_index=int(start_val), end_note_index=int(end_val),
                            value=str(value_val)
                        )
                        # Calculate and store onset range
                        start_onset, end_onset = self._get_phrase_onset_range(
                            phrase.melid, phrase.start_note_index, phrase.end_note_index
                        )
                        if start_onset is not None and end_onset is not None:
                            phrase.start_onset = start_onset
                            phrase.end_onset = end_onset
                            phrases_info.append(phrase)

                    except (ValueError, TypeError) as e:
                        print(f"Skipping section row due to parsing error: {row} - {e}")

        except sqlite3.Error as e:
            print(f"Database error retrieving phrases: {e}")

        return phrases_info

    def get_phrase_chords_sequence(self, phrase: PhraseInfo) -> List[Tuple[int, List[str]]]:
        """
        Retrieves the chord sequence for a given phrase from the 'beats' table.
        Requires phrase.start_onset and phrase.end_onset to be populated.
        Returns data in the format required by PatternAnalyzer: List[(bar_num, [chords])].
        Handles propagation of chords for bars without explicit chord changes within the phrase.
        """
        if phrase.start_onset is None or phrase.end_onset is None:
            return []

        start_onset = phrase.start_onset
        end_onset = phrase.end_onset

        chord_sequence_map: Dict[int, List[str]] = {}
        min_bar = float('inf')
        max_bar = float('-inf')
        found_chords_in_range = False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get chords *starting* strictly within the phrase's onset range
                cursor.execute("""
                    SELECT bar, onset, chord
                    FROM beats
                    WHERE melid = ? AND onset >= ? AND onset < ?
                    ORDER BY onset ASC
                """, (phrase.melid, start_onset, end_onset))
                rows_in_range = cursor.fetchall()

                for bar, onset, chord in rows_in_range:
                    if chord and chord.strip():
                        found_chords_in_range = True
                        bar = int(bar) # Ensure bar is int
                        if bar not in chord_sequence_map:
                            chord_sequence_map[bar] = []
                        # Add only if different from the last chord in this bar
                        if not chord_sequence_map[bar] or chord_sequence_map[bar][-1] != chord:
                            chord_sequence_map[bar].append(chord)
                        min_bar = min(min_bar, bar)
                        max_bar = max(max_bar, bar)

                # Determine the chord active *at* the start of the phrase
                cursor.execute("""
                    SELECT bar, chord
                    FROM beats
                    WHERE melid = ? AND onset <= ?
                    ORDER BY onset DESC, beatid DESC
                    LIMIT 1
                """, (phrase.melid, start_onset))
                start_chord_row = cursor.fetchone()

                # Determine the starting bar of the phrase from melody table
                cursor.execute("""
                    SELECT bar
                    FROM melody
                    WHERE melid = ?
                    ORDER BY onset ASC
                    LIMIT 1 OFFSET ?
                """, (phrase.melid, phrase.start_note_index))
                start_note_bar_row = cursor.fetchone()
                phrase_start_bar = int(start_note_bar_row[0]) if start_note_bar_row else None # Ensure int

                if phrase_start_bar is None:
                    return []

                # Handle the chord active at the start
                start_bar_chord = None
                if start_chord_row:
                    start_chord_bar, start_bar_chord_text = start_chord_row
                    if start_bar_chord_text and start_bar_chord_text.strip():
                        start_bar_chord = start_bar_chord_text # Store the valid chord text

                # If no chords started *within* the range, the phrase holds the start_chord
                if not found_chords_in_range and start_bar_chord:
                     chord_sequence_map[phrase_start_bar] = [start_bar_chord]
                     min_bar = phrase_start_bar
                     max_bar = phrase_start_bar

                # Ensure the chord active at the start is represented in the map if the start bar wasn't already populated
                elif start_bar_chord and phrase_start_bar not in chord_sequence_map:
                     chord_sequence_map[phrase_start_bar] = [start_bar_chord]
                     min_bar = min(min_bar, phrase_start_bar)
                     max_bar = max(max_bar, phrase_start_bar)


                # --- Convert map to the List[Tuple[int, List[str]]] format with propagation ---
                if min_bar > max_bar or min_bar == float('inf'): # Check if any valid chords were actually determined
                    return []

                final_sequence = []
                last_valid_chord_list: List[str] = []
                if start_bar_chord: # Initialize propagation with the chord active at start
                    last_valid_chord_list = [start_bar_chord]

                # Iterate through all bars from the phrase's start bar up to the last bar with a chord change
                for bar_num in range(phrase_start_bar, max_bar + 1):
                    if bar_num in chord_sequence_map:
                        current_bar_chords = chord_sequence_map[bar_num]
                        final_sequence.append((bar_num, current_bar_chords))
                        last_valid_chord_list = current_bar_chords # Update last seen chords for propagation
                    else:
                        # Propagate the last known chord list
                        final_sequence.append((bar_num, list(last_valid_chord_list))) # Use list copy

        except sqlite3.Error as e:
            print(f"Database error getting phrase chords for melid {phrase.melid}, range {start_onset}-{end_onset}: {e}")
            return []
        except Exception as e:
             print(f"Unexpected error getting phrase chords sequence: {e}")
             import traceback
             traceback.print_exc() # Print stack trace for debugging
             return []

        return final_sequence


    def get_processed_phrase_chords(self, phrase: PhraseInfo) -> List[ChordWithDuration]:
        """
        Gets the raw chord sequence for a phrase and processes it into a list of
        ChordWithDuration, merging consecutive identical chords and calculating durations.
        Uses PatternAnalyzer._process_chord_sequence logic.

        Args:
            phrase: The PhraseInfo object (must have onsets calculated).

        Returns:
            List of ChordWithDuration representing the phrase's harmony.
        """
        raw_sequence = self.get_phrase_chords_sequence(phrase)
        if not raw_sequence:
            return []

        # Delegate processing to PatternAnalyzer
        # We assume PatternAnalyzer has the _process_chord_sequence method
        # Note: _process_chord_sequence might need access to time signature eventually
        try:
             # Ensure the analyzer's process method is robust
             processed_chords = self.pattern_analyzer._process_chord_sequence(raw_sequence)
             return processed_chords
        except Exception as e:
             print(f"Error processing chord sequence for phrase {phrase.melid} with PatternAnalyzer: {e}")
             return []


    def compute_phrase_features(self, phrase_chords: List[ChordWithDuration], interval_weight: float = 3.0) -> Dict[str, float]:
        """
        Computes key-invariant features for the phrase, mirroring
        PatternAnalyzer._compute_comparison_features logic including interval normalization.
        """
        features = {}
        if not phrase_chords:
            return features

        total_duration = sum(c.duration for c in phrase_chords)
        if total_duration <= 0:
            return features

        num_chords = len(phrase_chords)
        # Use same normalization as in PatternAnalyzer
        features['feat_num_chords'] = num_chords / 10.0
        features['feat_total_duration'] = total_duration / 20.0

        for i, chord_dur in enumerate(phrase_chords):
            # Chord position
            features[f'feat_chord_{i}_position'] = i / num_chords if num_chords > 1 else 0

            # Chord relative duration
            features[f'feat_chord_{i}_duration'] = chord_dur.duration / total_duration

            # Chord type (using analyzer's centralized method)
            chord_type = self.pattern_analyzer._get_chord_type(chord_dur.chord)
            features[f'feat_chord_{i}_type'] = hash(chord_type) % 100 / 100.0

            # Interval to next chord (if exists)
            if i < num_chords - 1:
                interval = self.pattern_analyzer.get_relative_interval(chord_dur.chord, phrase_chords[i+1].chord)
                # Use same interval feature calculation as in PatternAnalyzer
                interval_feature_normalized = 0.5 # Default for None interval
                if interval is not None:
                     # Normalize interval from range [-5, 6] to [0, 1]
                     normalized_interval = (interval + 5.0) / 11.0
                     interval_feature_normalized = max(0.0, min(1.0, normalized_interval))
                # Apply weight
                features[f'feat_interval_{i}'] = interval_feature_normalized * interval_weight

        return features

    def compute_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """
        Calculates cosine similarity between two feature vectors (dictionaries).
        Handles potentially different sets of keys by using 0.0 for missing keys.
        """
        # Get all unique keys from both dictionaries
        keys = set(features1.keys()) | set(features2.keys())

        if not keys: # Avoid division by zero if both feature sets are empty
             return 1.0 if not features1 and not features2 else 0.0

        # Create vectors based on the union of keys, using 0.0 for missing values
        vector1 = [features1.get(k, 0.0) for k in keys]
        vector2 = [features2.get(k, 0.0) for k in keys]

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector1, vector2))

        # Calculate magnitudes (norms)
        norm1 = math.sqrt(sum(a * a for a in vector1))
        norm2 = math.sqrt(sum(b * b for b in vector2))

        # Calculate cosine similarity
        if norm1 == 0.0 or norm2 == 0.0:
            # If one vector is all zeros, similarity is 0 unless both are zeros
            return 1.0 if norm1 == 0.0 and norm2 == 0.0 else 0.0
        else:
            similarity = dot_product / (norm1 * norm2)
             # Clamp similarity to [0, 1] (or [-1, 1] if features aren't guaranteed non-negative)
             # Cosine similarity is naturally in [-1, 1]. Given our hashing % 100 / 100.0, features are [0, 1).
             # Clamping to [0, 1] seems reasonable.
            return max(0.0, min(1.0, similarity))


    def get_phrase_beats(self, section_id: int) -> List[BeatInfo] | None:
        """
        Retrieves the sequence of beats (including chords) within the boundaries
        of a specific phrase section, ordered by onset time.

        Args:
            section_id: The unique ID of the section in the 'sections' table.

        Returns:
            A list of BeatInfo objects or None if an error occurred.
            Returns an empty list if the phrase is found but contains no beats.
        """
        boundaries = self._get_phrase_boundaries_from_eventids(section_id)
        if not boundaries:
            return None # Error message already printed

        melid, start_onset, end_onset = boundaries

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Fetch beats within the phrase onset range [start_onset, end_onset)
                query = """
                    SELECT onset, bar, beat, signature, chord
                    FROM beats
                    WHERE melid = ? AND onset >= ? AND onset < ?
                    ORDER BY onset
                """
                cursor.execute(query, (melid, start_onset, end_onset))
                rows = cursor.fetchall()

                phrase_beats = [
                    BeatInfo(
                        onset=row[0], bar=row[1], beat=row[2],
                        signature=row[3], chord=row[4]
                    ) for row in rows
                ]
                return phrase_beats

        except sqlite3.Error as e:
            print(f"Database error fetching beats for phrase section {section_id}: {e}")
            return None


    def format_beats_for_analyzer(self, phrase_beats: List[BeatInfo]) -> List[Tuple[int, List[str]]]:
        """
        Formats a list of BeatInfo objects into the List[Tuple[int, List[str]]]
        structure required by PatternAnalyzer. Handles chord propagation for empty bars.

        Args:
            phrase_beats: A list of BeatInfo objects for the phrase, ordered by onset.

        Returns:
            A list of tuples: (bar_number, [list_of_chords_in_bar]).
        """
        if not phrase_beats:
            return []

        beats_by_bar: Dict[int, List[BeatInfo]] = {}
        min_bar = phrase_beats[0].bar
        max_bar = phrase_beats[-1].bar

        for beat_info in phrase_beats:
            if beat_info.bar not in beats_by_bar:
                beats_by_bar[beat_info.bar] = []
            beats_by_bar[beat_info.bar].append(beat_info)

        chord_sequence = []
        last_chord = None # Track the last explicitly defined chord

        for bar_num in range(min_bar, max_bar + 1):
            bar_chords_list = []
            if bar_num in beats_by_bar:
                # Get unique chords within the bar, ordered by beat
                bar_beat_infos = sorted(beats_by_bar[bar_num], key=lambda b: b.beat)
                current_bar_chords_unique = []
                for bi in bar_beat_infos:
                    # Add chord if it's the first or different from the last added
                    if bi.chord and (not current_bar_chords_unique or bi.chord != current_bar_chords_unique[-1]):
                         current_bar_chords_unique.append(bi.chord)

                if current_bar_chords_unique:
                     bar_chords_list = current_bar_chords_unique
                     last_chord = current_bar_chords_unique[-1] # Update last known chord
                     chord_sequence.append((bar_num, bar_chords_list))
                else: # Bar exists in beats_by_bar but has no actual chords? Propagate.
                     if last_chord:
                         chord_sequence.append((bar_num, [last_chord]))
                     else:
                         chord_sequence.append((bar_num, [])) # Truly empty bar
            else:
                # Bar number is within phrase range but has no entries in 'beats' table
                if last_chord:
                    chord_sequence.append((bar_num, [last_chord])) # Propagate last chord
                else:
                    chord_sequence.append((bar_num, [])) # Empty bar at the beginning

        return chord_sequence

    def get_phrase_chord_sequence_for_analysis(self, section_id: int) -> List[Tuple[int, List[str]]] | None:
        """
        High-level method to get the chord sequence of a phrase formatted
        for the PatternAnalyzer.

        Args:
            section_id: The unique ID of the section in the 'sections' table.

        Returns:
            The formatted chord sequence List[Tuple[int, List[str]]],
            an empty list if the phrase has no beats, or None if an error occurred.
        """
        phrase_beats = self.get_phrase_beats(section_id)
        if phrase_beats is None: # Error during fetch
             return None
        # No error, but phrase might be empty or contain no beats
        return self.format_beats_for_analyzer(phrase_beats)

    def get_all_phrase_section_ids(self, phrase_type: str = 'phrase') -> List[int]:
         """
         Gets the unique IDs (e.g., primary keys) of all sections matching the given type.

         Args:
             phrase_type: The value in the 'type' column to filter by (default 'phrase').

         Returns:
             A list of section IDs.
         """
         ids = []
         try:
             with sqlite3.connect(self.db_path) as conn:
                 cursor = conn.cursor()
                 cursor.execute(f"SELECT section_id FROM sections WHERE type = ?", (phrase_type,))
                 rows = cursor.fetchall()
                 ids = [row[0] for row in rows]
         except sqlite3.Error as e:
             print(f"Database error fetching phrase section IDs: {e}")
         return ids
