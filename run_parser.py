from core.pattern_analysis.parser import DatabaseChordParser, PatternAnalyzer

def main():
    parser = DatabaseChordParser(r"C:\polytech\Diploma\wjazzd.db")
    song_chords = parser.parse_song_chords(1)
    analyzer = PatternAnalyzer()
    patterns = analyzer.find_two_five_one(song_chords)
    print(patterns)

if __name__ == "__main__":
    main()
