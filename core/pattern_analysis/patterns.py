from abc import ABC, abstractmethod
from typing import List

from .models import ChordPattern, ChordWithDuration, ChordQuality

class HarmonicPattern(ABC):
    """Базовый класс для всех гармонических паттернов"""
    
    def __init__(self):
        self.pattern_type: str = self.__class__.__name__
    
    @abstractmethod
    def match(self, chords: List[ChordWithDuration], analyzer: 'PatternAnalyzer') -> bool:
        """Проверяет, соответствует ли последовательность аккордов паттерну"""
        pass
    
    @abstractmethod
    def create_pattern(self, chords: List[ChordWithDuration], start_bar: int) -> ChordPattern:
        """Создает объект паттерна из последовательности аккордов"""
        pass
        
    @abstractmethod
    def get_window_size(self) -> int:
        """Возвращает размер окна для поиска паттерна (количество аккордов)"""
        pass


class TwoFiveOnePattern(HarmonicPattern):
    """Паттерн II-V-I"""
    
    def get_window_size(self) -> int:
        return 3
    
    def match(self, chords: List[ChordWithDuration], analyzer: 'PatternAnalyzer') -> bool:
        if len(chords) != 3:
            return False
            
        chord2, chord5, chord1 = chords
        
        # Проверяем интервалы
        interval1 = analyzer.get_relative_interval(chord2.chord, chord5.chord)
        interval2 = analyzer.get_relative_interval(chord5.chord, chord1.chord)
        
        if not (interval1 == 5 and (interval2 == 5 or interval2 == -7)):
            return False
            
        # Проверяем качество аккордов
        is_minor = analyzer.is_minor_chord(chord2.chord)
        is_dominant = analyzer.is_dominant_chord(chord5.chord)
        is_tonic = (analyzer.is_major_chord(chord1.chord) or 
                   analyzer.is_major7_chord(chord1.chord) or 
                   analyzer.is_dominant_chord(chord1.chord))
                   
        return is_minor and is_dominant and is_tonic
    
    def create_pattern(self, chords: List[ChordWithDuration], start_bar: int) -> ChordPattern:
        # Определяем тональность паттерна (это будет тоника)
        chord1 = chords[2].chord
        key = chord1[0]
        if len(chord1) > 1 and chord1[1] in ['#', 'b']:
            key = chord1[:2]
            
        return ChordPattern(
            pattern_type="II-V-I",
            chords=chords,
            start_bar=start_bar,
            total_duration=sum(c.duration for c in chords),
            key=key
        )


class MinorTwoFiveOnePattern(HarmonicPattern):
    """Минорный паттерн II-V-I (IIm7b5 - V7 - Im7)"""
    
    def get_window_size(self) -> int:
        return 3
    
    def match(self, chords: List[ChordWithDuration], analyzer: 'PatternAnalyzer') -> bool:
        if len(chords) != 3:
            return False
            
        chord2, chord5, chord1 = chords
        
        # Проверяем интервалы (те же, что и в мажоре)
        interval1 = analyzer.get_relative_interval(chord2.chord, chord5.chord)
        interval2 = analyzer.get_relative_interval(chord5.chord, chord1.chord)
        
        if not (interval1 == 5 and (interval2 == 5 or interval2 == -7)):
            return False
            
        # Проверяем качество аккордов для минорной прогрессии
        is_half_dim = analyzer.is_half_diminished_chord(chord2.chord)  # IIm7b5
        is_dominant = analyzer.is_dominant_chord(chord5.chord)         # V7
        is_minor = (analyzer.is_minor_chord(chord1.chord) or          # Im или Im7
                   analyzer.is_minor7_chord(chord1.chord))
                   
        return is_half_dim and is_dominant and is_minor
    
    def create_pattern(self, chords: List[ChordWithDuration], start_bar: int) -> ChordPattern:
        # Определяем тональность паттерна (это будет минорная тоника)
        chord1 = chords[2].chord
        key = chord1[0]
        if len(chord1) > 1 and chord1[1] in ['#', 'b']:
            key = chord1[:2]
            
        return ChordPattern(
            pattern_type="Minor-II-V-I",
            chords=chords,
            start_bar=start_bar,
            total_duration=sum(c.duration for c in chords),
            key=key
        )


class BluesPattern(HarmonicPattern):
    """Блюзовый паттерн"""
    
    def get_window_size(self) -> int:
        return 3
    
    def match(self, chords: List[ChordWithDuration], analyzer: 'PatternAnalyzer') -> bool:
        if len(chords) != 3:
            return False
            
        chord1, chord4, chord5 = chords
        
        # Проверяем интервалы
        interval1 = analyzer.get_relative_interval(chord1.chord, chord4.chord)
        interval2 = analyzer.get_relative_interval(chord4.chord, chord5.chord)
        
        if not (interval1 == 5 and interval2 == 3):
            return False
            
        # Проверяем качество аккордов
        is_tonic = (analyzer.is_major_chord(chord1.chord) or 
                   analyzer.is_major7_chord(chord1.chord) or 
                   analyzer.is_dominant_chord(chord1.chord))
        is_subdominant = analyzer.is_minor_chord(chord4.chord)
        is_dominant = analyzer.is_dominant_chord(chord5.chord)
                   
        return is_tonic and is_subdominant and is_dominant
    
    def create_pattern(self, chords: List[ChordWithDuration], start_bar: int) -> ChordPattern:
        # Определяем тональность паттерна (это будет тоника)
        chord1 = chords[0].chord
        key = chord1[0]
        if len(chord1) > 1 and chord1[1] in ['#', 'b']:
            key = chord1[:2]
            
        return ChordPattern(
            pattern_type="Blues",
            chords=chords,
            start_bar=start_bar,
            total_duration=sum(c.duration for c in chords),
            key=key
        )
