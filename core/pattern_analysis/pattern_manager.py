import sqlite3
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import DatabaseChordParser
from .parser import DatabaseChordParser, parse_bar_chord # Import parse_bar_chord if needed elsewhere, or remove if fully encapsulated
from .models import ChordPattern
from .pattern_analyzer import PatternAnalyzer


@dataclass
class PhrasePattern:
    """Class to store information about a pattern within a phrase"""
    phrase_id: int
    pattern_type: str
    start_bar: int
    key: str
    chords: str  # JSON string with chords and their durations
    features: Dict[str, float]  # Vector features of the pattern


class PatternManager:
    # Accept DatabaseChordParser instance in constructor
    def __init__(self, db_path: str, parser: DatabaseChordParser):
        self.db_path = db_path
        self.parser = parser # Store the parser instance
        self.pattern_analyzer = PatternAnalyzer()

    def initialize_database(self):
        """Creates necessary tables in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table to store found patterns
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patterns (
                        pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        song_id INTEGER,
                        pattern_type TEXT,
                        start_bar INTEGER,
                        key TEXT,
                        chords TEXT,  -- JSON string with chords and durations
                        features TEXT, -- JSON string with vector features
                        FOREIGN KEY (song_id) REFERENCES songs(id)
                    )
                """)
                
                # Table to link patterns with phrases
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS phrase_patterns (
                        phrase_id INTEGER,
                        pattern_id INTEGER,
                        similarity FLOAT,
                        FOREIGN KEY (phrase_id) REFERENCES phrases(id),
                        FOREIGN KEY (pattern_id) REFERENCES patterns(pattern_id),
                        PRIMARY KEY (phrase_id, pattern_id)
                    )
                """)
                
                conn.commit()
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")

    def store_pattern(self, song_id: int, pattern: ChordPattern):
        """Saves a found pattern to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert chords to JSON
                chords_json = json.dumps([
                    {"chord": c.chord, "duration": c.duration}
                    for c in pattern.chords
                ])
                
                # Convert features to JSON
                features_json = json.dumps(pattern.features)
                
                cursor.execute("""
                    INSERT INTO patterns (
                        song_id, pattern_type, start_bar, key, 
                        chords, features
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    song_id, pattern.pattern_type, pattern.start_bar,
                    pattern.key, chords_json, features_json
                ))
                
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error storing pattern: {e}")
            return None

    def find_similar_phrases(self, pattern_id: int, threshold: float = 0.7) -> List[Tuple[int, float]]:
        """Finds similar phrases for a given pattern"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get pattern features
                cursor.execute("""
                    SELECT features FROM patterns WHERE pattern_id = ?
                """, (pattern_id,))
                pattern_features_row = cursor.fetchone()
                if not pattern_features_row:
                    print(f"Pattern with ID {pattern_id} not found.")
                    return []
                pattern_features = json.loads(pattern_features_row[0])
                
                # Get all phrases
                cursor.execute("""
                    SELECT id, chords FROM phrases
                """)
                phrases = cursor.fetchall()
                
                similar_phrases = []
                for phrase_id, phrase_chords_json in phrases:
                    # TODO: Need a way to parse phrase_chords_json into a format
                    # that self.pattern_analyzer.find_patterns can use.
                    # This depends on how phrase chords are stored in the 'phrases' table.
                    # Assuming phrase_chords_json is similar to the input of find_patterns
                    # e.g., List[Tuple[int, List[str]]] after json.loads()
                    # Or maybe phrase_chords represents something else entirely.
                    # For now, this part is commented out as it needs clarification.

                    # phrase_chords_sequence = json.loads(phrase_chords_json) # Assuming JSON list of [bar, [chords]]
                    # phrase_patterns = self.pattern_analyzer.find_patterns(phrase_chords_sequence)

                    # Placeholder: Replace with actual pattern analysis on phrase chords
                    phrase_patterns = []
                    
                    # Calculate similarity for each found pattern in the phrase
                    max_similarity = 0
                    for phrase_pattern in phrase_patterns:
                        similarity = self.compute_similarity(
                            pattern_features,
                            phrase_pattern.features # Assumes phrase_pattern is ChordPattern
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    if max_similarity >= threshold:
                        similar_phrases.append((phrase_id, max_similarity))
                        
                        # Save the link between pattern and phrase
                        cursor.execute("""
                            INSERT OR REPLACE INTO phrase_patterns 
                            (phrase_id, pattern_id, similarity)
                            VALUES (?, ?, ?)
                        """, (phrase_id, pattern_id, max_similarity))
                
                conn.commit()
                return similar_phrases
        except sqlite3.Error as e:
            print(f"Error finding similar phrases: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON data: {e}") # Handle potential JSON errors
            return []

    def compute_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculates cosine similarity between feature vectors"""
        # Get all unique keys
        keys = set(features1.keys()) | set(features2.keys())
        
        # Create vectors
        vector1 = [features1.get(k, 0.0) for k in keys]
        vector2 = [features2.get(k, 0.0) for k in keys]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = sum(a * a for a in vector1) ** 0.5
        norm2 = sum(b * b for b in vector2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def analyze_and_store_song_patterns(self, song_id: int) -> List[int]:
        """Analyzes a song and stores all found patterns"""
        # Use the parser instance to get chords
        song_chords_sequence = self.parser.parse_song_chords(song_id)
        if not song_chords_sequence:
             print(f"Could not retrieve or parse chords for song {song_id}. Aborting analysis.")
             return []

        # Analyze patterns using the unified finder
        patterns = self.pattern_analyzer.find_patterns(song_chords_sequence)

        # Store patterns in the database
        pattern_ids = []
        for pattern in patterns:
            pattern_id = self.store_pattern(song_id, pattern)
            if pattern_id:
                pattern_ids.append(pattern_id)

        return pattern_ids
