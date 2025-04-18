import sqlite3
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .models import ChordPattern
from .pattern_analyzer import PatternAnalyzer


@dataclass
class PhrasePattern:
    """Класс для хранения информации о паттерне во фразе"""
    phrase_id: int
    pattern_type: str
    start_bar: int
    key: str
    chords: str  # JSON строка с аккордами и их длительностями
    features: Dict[str, float]  # Векторные характеристики паттерна


class PatternManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pattern_analyzer = PatternAnalyzer()

    def initialize_database(self):
        """Создает необходимые таблицы в базе данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Таблица для хранения найденных паттернов
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patterns (
                        pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        song_id INTEGER,
                        pattern_type TEXT,
                        start_bar INTEGER,
                        key TEXT,
                        chords TEXT,  -- JSON строка с аккордами и длительностями
                        features TEXT, -- JSON строка с векторными характеристиками
                        FOREIGN KEY (song_id) REFERENCES songs(id)
                    )
                """)
                
                # Таблица для связи паттернов с фразами
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
            print(f"Ошибка при инициализации базы данных: {e}")

    def store_pattern(self, song_id: int, pattern: ChordPattern):
        """Сохраняет найденный паттерн в базу данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Преобразуем аккорды в JSON
                chords_json = json.dumps([
                    {"chord": c.chord, "duration": c.duration}
                    for c in pattern.chords
                ])
                
                # Преобразуем характеристики в JSON
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
            print(f"Ошибка при сохранении паттерна: {e}")
            return None

    def find_similar_phrases(self, pattern_id: int, threshold: float = 0.7) -> List[Tuple[int, float]]:
        """Находит похожие фразы для данного паттерна"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Получаем характеристики паттерна
                cursor.execute("""
                    SELECT features FROM patterns WHERE pattern_id = ?
                """, (pattern_id,))
                pattern_features = json.loads(cursor.fetchone()[0])
                
                # Получаем все фразы с их паттернами
                cursor.execute("""
                    SELECT id, chords FROM phrases
                """)
                phrases = cursor.fetchall()
                
                similar_phrases = []
                for phrase_id, phrase_chords in phrases:
                    # Анализируем паттерны во фразе
                    phrase_patterns = self.pattern_analyzer.analyze_patterns(phrase_chords)
                    
                    # Для каждого найденного паттерна вычисляем сходство
                    max_similarity = 0
                    for phrase_pattern in phrase_patterns:
                        similarity = self.compute_similarity(
                            pattern_features,
                            phrase_pattern.features
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    if max_similarity >= threshold:
                        similar_phrases.append((phrase_id, max_similarity))
                        
                        # Сохраняем связь паттерна с фразой
                        cursor.execute("""
                            INSERT OR REPLACE INTO phrase_patterns 
                            (phrase_id, pattern_id, similarity)
                            VALUES (?, ?, ?)
                        """, (phrase_id, pattern_id, max_similarity))
                
                conn.commit()
                return similar_phrases
        except sqlite3.Error as e:
            print(f"Ошибка при поиске похожих фраз: {e}")
            return []

    def compute_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Вычисляет косинусное сходство между векторами характеристик"""
        # Получаем все уникальные ключи
        keys = set(features1.keys()) | set(features2.keys())
        
        # Создаем векторы
        vector1 = [features1.get(k, 0.0) for k in keys]
        vector2 = [features2.get(k, 0.0) for k in keys]
        
        # Вычисляем косинусное сходство
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = sum(a * a for a in vector1) ** 0.5
        norm2 = sum(b * b for b in vector2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def analyze_and_store_song_patterns(self, song_id: int) -> List[int]:
        """Анализирует песню и сохраняет все найденные паттерны"""
        # Получаем аккорды песни
        song_chords = self.get_song_chords(song_id)
        
        # Анализируем все возможные паттерны
        patterns = []
        for pattern_type in self.pattern_analyzer.patterns:
            found_patterns = pattern_type.find_patterns(song_chords)
            patterns.extend(found_patterns)
        
        # Сохраняем паттерны в базу
        pattern_ids = []
        for pattern in patterns:
            pattern_id = self.store_pattern(song_id, pattern)
            if pattern_id:
                pattern_ids.append(pattern_id)
        
        return pattern_ids

    def get_song_chords(self, song_id: int) -> List[Tuple[int, List[str]]]:
        """Получает аккорды песни из базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT bar, chords 
                    FROM song_chords 
                    WHERE song_id = ? 
                    ORDER BY bar
                """, (song_id,))
                return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Ошибка при получении аккордов песни: {e}")
            return [] 