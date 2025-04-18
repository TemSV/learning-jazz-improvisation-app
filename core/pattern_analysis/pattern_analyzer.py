from typing import List, Tuple, Dict

from .models import ChordInfo, ChordQuality, ChordPattern, ChordWithDuration
from .patterns import HarmonicPattern, TwoFiveOnePattern, BluesPattern, MinorTwoFiveOnePattern


class PatternAnalyzer:
    def __init__(self):
        self.note_values = {
            'C': 0, 'C#': 1, 'Db': 1,
            'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4,
            'F': 5, 'F#': 6, 'Gb': 6,
            'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10,
            'B': 11
        }
        # Регистрируем доступные паттерны
        self.patterns: List[HarmonicPattern] = [
            TwoFiveOnePattern(),
            MinorTwoFiveOnePattern(),
            BluesPattern()
        ]

    def parse_chord(self, chord: str) -> ChordInfo:
        """
        Разбирает строковое представление аккорда на составляющие
        Например: "Dm7" -> ChordInfo(root="D", quality=ChordQuality.MINOR7)
        """
        # Получаем основной тон
        root = chord[0]
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            root = chord[:2]
            quality_part = chord[2:]
        else:
            quality_part = chord[1:]

        # Определяем качество аккорда
        if 'j7' in quality_part:
            quality = ChordQuality.MAJOR7
        elif 'm7b5' in quality_part:
            quality = ChordQuality.HALF_DIMINISHED
        elif '-7' in quality_part:
            quality = ChordQuality.MINOR7
        elif '7' in quality_part:
            quality = ChordQuality.DOMINANT
        elif '-' in quality_part:
            quality = ChordQuality.MINOR
        elif 'dim' in quality_part:
            quality = ChordQuality.DIMINISHED
        elif 'aug' in quality_part:
            quality = ChordQuality.AUGMENTED
        else:
            quality = ChordQuality.MAJOR

        return ChordInfo(root=root, quality=quality)

    def get_chord_intervals(self, chord_sequence: List[str]) -> List[int]:
        """
        Вычисляет интервалы между аккордами в полутонах
        """
        intervals = []
        for i in range(len(chord_sequence) - 1):
            current = self.parse_chord(chord_sequence[i])
            next_chord = self.parse_chord(chord_sequence[i + 1])
            
            current_value = self.note_values[current.root]
            next_value = self.note_values[next_chord.root]
            
            interval = (next_value - current_value) % 12
            if interval > 6:
                interval -= 12
            intervals.append(interval)
            
        return intervals

    def find_patterns(self, chord_sequence: List[Tuple[int, List[str]]]) -> List[ChordPattern]:
        """
        Ищет все зарегистрированные паттерны в последовательности аккордов
        """
        # Преобразуем последовательность в список аккордов с длительностями
        chords_with_duration = self._process_chord_sequence(chord_sequence)
        
        found_patterns = []
        
        # Для каждого паттерна проверяем все возможные подпоследовательности
        for pattern in self.patterns:
            window_size = pattern.get_window_size()
            for i in range(len(chords_with_duration) - window_size + 1):
                window = chords_with_duration[i:i + window_size]
                if pattern.match(window, self):
                    found_patterns.append(pattern.create_pattern(window, i))
        
        return found_patterns

    def _process_chord_sequence(self, chord_sequence: List[Tuple[int, List[str]]]) -> List[ChordWithDuration]:
        """
        Преобразует последовательность аккордов в список с длительностями
        """
        chords_with_duration = []
        current_chord = None
        current_duration = 0
        beats_per_bar = 4  # Размер 4/4
        
        for i, (bar_num, bar_chords) in enumerate(chord_sequence):
            if not bar_chords and current_chord:
                # Если такт пустой и есть текущий аккорд, увеличиваем его длительность
                current_duration += beats_per_bar
            else:
                # Если есть аккорды в такте
                if current_chord:
                    # Добавляем предыдущий аккорд с накопленной длительностью
                    chords_with_duration.append(ChordWithDuration(current_chord, current_duration))
                
                if bar_chords:
                    # Распределяем доли между аккордами в такте
                    beats_per_chord = beats_per_bar / len(bar_chords)
                    for chord in bar_chords[:-1]:
                        chords_with_duration.append(ChordWithDuration(chord, beats_per_chord))
                    
                    # Последний аккорд в такте может продолжаться дальше
                    current_chord = bar_chords[-1]
                    current_duration = beats_per_chord
        
        # Добавляем последний аккорд, если он есть
        if current_chord:
            chords_with_duration.append(ChordWithDuration(current_chord, current_duration))
            
        return chords_with_duration

        # Поиск II-V-I
        for i in range(len(chord_sequence) - 2):
            subset = chord_sequence[i:i+3]
            if self._is_two_five_one(subset):
                if "II-V-I" not in patterns:
                    patterns["II-V-I"] = []
                patterns["II-V-I"].append((i, subset))
        
        # Поиск блюзового квадрата (I-IV-I-V)
        for i in range(len(chord_sequence) - 3):
            subset = chord_sequence[i:i+4]
            if self._is_blues_progression(subset):
                if "Blues" not in patterns:
                    patterns["Blues"] = []
                patterns["Blues"].append((i, subset))
        
        return patterns

    def _is_two_five_one(self, chords: List[str]) -> bool:
        """
        Проверяет, является ли последовательность паттерном II-V-I
        """
        if len(chords) != 3:
            return False
            
        chord_info = [self.parse_chord(c) for c in chords]
        intervals = self.get_chord_intervals(chords)
        
        # Проверяем интервалы (вверх на кварту, вниз на кварту)
        interval_pattern = [5, -5]
        if intervals != interval_pattern:
            return False
            
        # Проверяем качество аккордов
        return (chord_info[0].quality in [ChordQuality.MINOR, ChordQuality.MINOR7] and
                chord_info[1].quality in [ChordQuality.DOMINANT] and
                chord_info[2].quality in [ChordQuality.MAJOR, ChordQuality.MAJOR7])

    def _is_blues_progression(self, chords: List[str]) -> bool:
        """
        Проверяет, является ли последовательность блюзовым паттерном
        """
        if len(chords) != 4:
            return False
            
        chord_info = [self.parse_chord(c) for c in chords]
        intervals = self.get_chord_intervals(chords)
        
        # I-IV-I-V
        interval_pattern = [5, -5, 2]
        return intervals == interval_pattern 

    def get_relative_interval(self, chord1: str, chord2: str) -> int:
        """
        Вычисляет интервал между двумя аккордами в полутонах
        Положительное значение означает движение вверх, отрицательное - вниз
        """
        root1 = chord1[0]
        root2 = chord2[0]
        
        if len(chord1) > 1 and chord1[1] in ['#', 'b']:
            root1 = chord1[:2]
        if len(chord2) > 1 and chord2[1] in ['#', 'b']:
            root2 = chord2[:2]
            
        val1 = self.note_values[root1]
        val2 = self.note_values[root2]
        
        # Вычисляем интервал в обоих направлениях
        interval_up = (val2 - val1) % 12
        interval_down = interval_up - 12
        
        # Выбираем кратчайший путь
        return interval_up if abs(interval_up) <= abs(interval_down) else interval_down

    def is_dominant_chord(self, chord: str) -> bool:
        """Проверяет, является ли аккорд доминантовым (C7, F7 и т.д.)"""
        quality_part = chord[1:] if len(chord) > 1 and chord[1] not in ['#', 'b'] else chord[2:]
        return '7' in quality_part and 'j7' not in quality_part

    def is_minor_chord(self, chord: str) -> bool:
        """Проверяет, является ли аккорд минорным (C-, F- и т.д.)"""
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            return len(chord) > 2 and chord[2] == '-'
        return len(chord) > 1 and chord[1] == '-'

    def is_major_chord(self, chord: str) -> bool:
        """Проверяет, является ли аккорд мажорным (C, F, G и т.д.)"""
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            return len(chord) == 2 or (len(chord) > 2 and chord[2] not in ['-', '7'])
        return len(chord) == 1 or (len(chord) > 1 and chord[1] not in ['-', '7'])

    def is_major7_chord(self, chord: str) -> bool:
        """Проверяет, является ли аккорд мажорным с большой септимой (Cj7, Fj7 и т.д.)"""
        quality_part = chord[1:] if len(chord) > 1 and chord[1] not in ['#', 'b'] else chord[2:]
        return 'j7' in quality_part

    def is_half_diminished_chord(self, chord: str) -> bool:
        """Проверяет, является ли аккорд полууменьшенным (Cm7b5, Fm7b5 и т.д.)"""
        quality_part = chord[1:] if len(chord) > 1 and chord[1] not in ['#', 'b'] else chord[2:]
        return 'm7b5' in quality_part

    def is_minor7_chord(self, chord: str) -> bool:
        """Проверяет, является ли аккорд минорным септаккордом (C-7, F-7 и т.д.)"""
        quality_part = chord[1:] if len(chord) > 1 and chord[1] not in ['#', 'b'] else chord[2:]
        return '-7' in quality_part

    def find_two_five_one(self, chord_sequence: List[Tuple[int, List[str]]]) -> List[ChordPattern]:
        """
        Ищет II-V-I прогрессии в последовательности аккордов
        Возвращает список найденных паттернов
        """
        patterns = []
        
        # Преобразуем последовательность в список аккордов с длительностями
        chords_with_duration = []
        current_chord = None
        current_duration = 0
        beats_per_bar = 4  # Размер 4/4
        
        for i, (bar_num, bar_chords) in enumerate(chord_sequence):
            if not bar_chords and current_chord:
                # Если такт пустой и есть текущий аккорд, увеличиваем его длительность
                current_duration += beats_per_bar
            else:
                # Если есть аккорды в такте
                if current_chord:
                    # Добавляем предыдущий аккорд с накопленной длительностью
                    chords_with_duration.append(ChordWithDuration(current_chord, current_duration))
                
                if bar_chords:
                    # Распределяем доли между аккордами в такте
                    beats_per_chord = beats_per_bar / len(bar_chords)
                    for chord in bar_chords[:-1]:
                        chords_with_duration.append(ChordWithDuration(chord, beats_per_chord))
                    
                    # Последний аккорд в такте может продолжаться дальше
                    current_chord = bar_chords[-1]
                    current_duration = beats_per_chord
        
        # Добавляем последний аккорд, если он есть
        if current_chord:
            chords_with_duration.append(ChordWithDuration(current_chord, current_duration))
        
        print("\nПреобразованная последовательность:")
        for chord in chords_with_duration:
            print(f"{chord.chord} (длительность: {chord.duration} долей)")
        
        # Ищем паттерны
        for i in range(len(chords_with_duration) - 2):
            chord2 = chords_with_duration[i]
            chord5 = chords_with_duration[i + 1]
            chord1 = chords_with_duration[i + 2]
            
            print(f"\nПроверяем тройку: {chord2.chord} -> {chord5.chord} -> {chord1.chord}")
            print(f"Длительности: {chord2.duration} -> {chord5.duration} -> {chord1.duration} долей")
            
            # Проверяем интервалы
            interval1 = self.get_relative_interval(chord2.chord, chord5.chord)
            interval2 = self.get_relative_interval(chord5.chord, chord1.chord)
            print(f"Интервалы: {interval1}, {interval2}")
            
            # Проверяем интервалы для II-V-I (движение на кварту вверх, затем на кварту вниз)
            if interval1 == 5 and (interval2 == 5 or interval2 == -7):
                print("Интервалы подходят")
                
                # Проверяем качество аккордов
                is_minor = self.is_minor_chord(chord2.chord)
                is_dominant = self.is_dominant_chord(chord5.chord)
                is_tonic = (self.is_major_chord(chord1.chord) or 
                           self.is_major7_chord(chord1.chord) or 
                           self.is_dominant_chord(chord1.chord))
                
                print(f"Проверка качества аккордов:")
                print(f"II минорный: {is_minor}")
                print(f"V доминантовый: {is_dominant}")
                print(f"I подходящий: {is_tonic}")
                
                if is_minor and is_dominant and is_tonic:
                    print("Найден паттерн II-V-I!")
                    # Определяем тональность паттерна (это будет тоника)
                    key = chord1.chord[0]
                    if len(chord1.chord) > 1 and chord1.chord[1] in ['#', 'b']:
                        key = chord1.chord[:2]
                    
                    # Создаем объект паттерна
                    pattern = ChordPattern(
                        pattern_type="II-V-I",
                        chords=[chord2, chord5, chord1],
                        start_bar=i,  # Начальный такт
                        total_duration=chord2.duration + chord5.duration + chord1.duration,
                        key=key
                    )
                    patterns.append(pattern)
        
        return patterns
                