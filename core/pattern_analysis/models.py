from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ChordQuality(Enum):
    MAJOR = ''        # C, F, G и т.д.
    MINOR = '-'       # C-, F-, G- и т.д.
    DOMINANT = '7'    # C7, F7, G7 и т.д.
    MAJOR7 = 'j7'     # Cj7, Fj7, Gj7 и т.д.

@dataclass
class ChordInfo:
    root: str  # Основной тон аккорда (C, D, E, etc.)
    quality: ChordQuality  # Качество аккорда (maj, min, 7, etc.)

@dataclass
class ChordWithDuration:
    chord: str
    duration: float  # длительность в долях

@dataclass
class ChordPattern:
    """Класс для хранения информации о гармоническом паттерне"""
    pattern_type: str  # Тип паттерна (например, "II-V-I", "Blues", etc.)
    chords: List[ChordWithDuration]  # Список аккордов с их длительностями
    start_bar: int  # Начальный такт
    total_duration: float  # Общая длительность паттерна в долях
    key: str  # Тональность паттерна (например, "F" для II-V-I в F)
    features: Dict[str, float] = None  # Векторное представление паттерна
    
    def __post_init__(self):
        if self.features is None:
            self.features = self._compute_features()
        if not hasattr(self, 'total_duration'):
            self.total_duration = sum(chord.duration for chord in self.chords)
    
    def _compute_features(self) -> Dict[str, float]:
        """Вычисляет векторные характеристики паттерна"""
        features = {}
        
        # Характеристики на основе типа паттерна
        features['pattern_type'] = hash(self.pattern_type) % 100 / 100.0
        
        # Характеристики на основе тональности
        features['key'] = hash(self.key) % 100 / 100.0
        
        # Характеристики на основе аккордов
        for i, chord in enumerate(self.chords):
            # Позиция аккорда в паттерне
            features[f'chord_{i}_position'] = i / len(self.chords)
            
            # Длительность аккорда
            features[f'chord_{i}_duration'] = chord.duration / self.total_duration
            
            # Тип аккорда (мажор, минор, доминант и т.д.)
            chord_type = self._get_chord_type(chord.chord)
            features[f'chord_{i}_type'] = hash(chord_type) % 100 / 100.0
        
        return features
    
    def _get_chord_type(self, chord: str) -> str:
        """Определяет тип аккорда"""
        if 'j7' in chord:
            return 'maj7'
        elif 'm7b5' in chord:
            return 'half_dim'
        elif '-7' in chord:
            return 'min7'
        elif '7' in chord:
            return 'dom7'
        elif '-' in chord:
            return 'min'
        elif 'dim' in chord:
            return 'dim'
        elif 'aug' in chord:
            return 'aug'
        else:
            return 'maj'
            
    def to_dict(self) -> Dict:
        """Преобразует паттерн в словарь для сохранения в базу данных"""
        return {
            'pattern_type': self.pattern_type,
            'chords': [(c.chord, c.duration) for c in self.chords],
            'start_bar': self.start_bar,
            'total_duration': self.total_duration,
            'key': self.key,
            'features': self.features
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChordPattern':
        """Создает объект паттерна из словаря"""
        chords = [ChordWithDuration(chord, duration) 
                 for chord, duration in data['chords']]
        return cls(
            pattern_type=data['pattern_type'],
            chords=chords,
            start_bar=data['start_bar'],
            total_duration=data['total_duration'],
            key=data['key'],
            features=data['features']
        )
    
    def get_feature_vector(self) -> List[float]:
        """Возвращает векторное представление паттерна для сравнения"""
        # Здесь будет логика создания векторного представления паттерна
        # Например, можно учитывать:
        # - Интервалы между аккордами
        # - Длительности аккордов
        # - Типы аккордов (мажор, минор, доминант)
        # - Тональность
        pass
