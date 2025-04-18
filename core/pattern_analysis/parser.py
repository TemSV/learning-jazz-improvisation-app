import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

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
    Парсит содержимое одного такта
    """
    if not chord_text or chord_text.strip() == '':
        return []

    chords = chord_text.strip().split()
    return chords


class DatabaseChordParser:
    def __init__(self, db_path: str):
        """
        Инициализация парсера с путем к базе данных
        """
        self.db_path = db_path
        self.current_chord = None
        self.pattern_analyzer = PatternAnalyzer()

    def get_song_signature(self, songid: int) -> str:
        """
        Получает сигнатуру для конкретной песни из базы данных
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = """
                    SELECT signature 
                    FROM song_chords 
                    WHERE songid = ?
                """     
                cursor.execute(query, (songid,))
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            print(f"Ошибка при получении данных из БД: {e}")
            return None

    def get_song_chords(self, songid: int) -> List[SongChord]:
        """
        Получает все аккорды для конкретной песни из базы данных
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
                cursor.execute(query, (songid,))
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
            print(f"Ошибка при получении данных из БД: {e}")
            return []

    def parse_song_chords(self, songid: int) -> List[Tuple[int, List[str]]]:
        """
        Получает и парсит последовательность аккордов для песни
        """
        song_data = self.get_song_chords(songid)
        chord_sequence = []
        
        for bar_data in song_data:
            bar_chords = parse_bar_chord(bar_data.chords)

            if not bar_chords:
                if self.current_chord:
                    bar_chords = [self.current_chord]
            else:
                self.current_chord = bar_chords[-1]
            
            chord_sequence.append((bar_data.bar, bar_chords))
        
        return chord_sequence

    def analyze_patterns(self, songid: int):
        """
        Анализирует паттерны в песне
        """
        chord_sequence = self.parse_song_chords(songid)
        patterns = self.pattern_analyzer.find_two_five_one(chord_sequence)
        
        print(f"\nНайденные II-V-I в песне {songid}:")
        for start_pos, pattern in patterns:
            print(f"\nПозиция {start_pos}:")
            for chord in pattern:
                print(f"  {chord.chord} (длительность: {chord.duration} такт(а))")


def main():
    parser = DatabaseChordParser("C:\polytech\Diploma\wjazzd.db")
    song_chords = parser.parse_song_chords(1)
    analyzer = PatternAnalyzer()
    patterns = analyzer.find_two_five_one(song_chords)
    print(patterns)


if __name__ == "__main__":
    main()
