import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

from .models import ChordPattern
from .pattern_analyzer import PatternAnalyzer


@dataclass
class SongChord:
    barid: int
    songid: int
    bar: int
    signature: str
    chords: str
    form: str


def parse_bar_chord(chord_text: str) -> List[str]:
    """
    Parses the chord string for a single bar.
    Returns an empty list if the bar is empty or contains only whitespace.
    """
    if not chord_text or not chord_text.strip():
        return []
    return chord_text.strip().split()


class DatabaseChordParser:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.current_chord = None # Tracks the last chord seen for handling empty bars
        self.pattern_analyzer = PatternAnalyzer()

    def get_song_signature(self, song_id: int) -> str | None:
        """
        Retrieves the time signature for a specific song from the database.
        Returns None if not found or on error.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = """
                    SELECT signature
                    FROM song_chords
                    WHERE songid = ?
                    LIMIT 1
                """
                cursor.execute(query, (song_id,))
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            print(f"Error fetching song signature from DB: {e}")
            return None

    def get_song_chords(self, song_id: int) -> List[SongChord]:
        """
        Retrieves all chord entries for a specific song from the database,
        ordered by bar number.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = """
                    SELECT barid, songid, bar, signature, chords, form
                    FROM song_chords
                    WHERE songid = ?
                    ORDER BY bar
                """
                cursor.execute(query, (song_id,))
                rows = cursor.fetchall()

                return [
                    SongChord(
                        barid=row[0],
                        songid=row[1],
                        bar=row[2],
                        signature=row[3],
                        chords=row[4],
                        form=row[5]
                    )
                    for row in rows
                ]
        except sqlite3.Error as e:
            print(f"Error fetching song chords from DB: {e}")
            return []

    def parse_song_chords(self, song_id: int) -> List[Tuple[int, List[str]]]:
        """
        Retrieves and parses the chord sequence for a song.
        Handles empty bars by propagating the last known chord.
        Returns a list of tuples: (bar_number, [list_of_chords_in_bar]).
        """
        song_data = self.get_song_chords(song_id)
        chord_sequence = []
        self.current_chord = None # Reset for each song parse

        for bar_data in song_data:
            bar_chords = parse_bar_chord(bar_data.chords)

            if not bar_chords:
                # If bar is empty, use the last known chord (if any)
                if self.current_chord:
                    # Represent this bar as containing the held chord
                    bar_chords_to_add = [self.current_chord]
                else:
                    # No previous chord and bar is empty (e.g., start of song)
                    bar_chords_to_add = [] # Represent as truly empty
            else:
                # Update the last known chord if the bar wasn't empty
                self.current_chord = bar_chords[-1]
                bar_chords_to_add = bar_chords

            chord_sequence.append((bar_data.bar, bar_chords_to_add))

        return chord_sequence

    def analyze_song(self, song_id: int) -> List[ChordPattern]:
        """
        Analyzes harmonic patterns in the specified song.
        """
        print(f"\nAnalyzing song ID: {song_id}")
        chord_sequence = self.parse_song_chords(song_id)

        # Use the unified find_patterns method
        patterns = self.pattern_analyzer.find_patterns(chord_sequence)

        print(f"\nFound patterns in song {song_id}:")
        if not patterns:
            print("  No patterns found.")
        else:
            for pattern in patterns:
                chord_strs = [f"{c.chord} ({c.duration} beats)" for c in pattern.chords]
                print(f"  - {pattern.pattern_type} in {pattern.key} starting at index {pattern.start_bar}: [{ ' | '.join(chord_strs) }]")
        return patterns


def main():
    # Example usage:
    db_path = r"C:\polytech\Diploma\wjazzd.db"
    parser = DatabaseChordParser(db_path)

    song_id_to_analyze = 1
    found_patterns = parser.analyze_song(song_id_to_analyze)



if __name__ == "__main__":
    main()
