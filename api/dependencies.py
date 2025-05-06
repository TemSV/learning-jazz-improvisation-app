import os
from functools import lru_cache
from core.pattern_analysis.parser import DatabaseChordParser
from core.pattern_analysis.pattern_analyzer import PatternAnalyzer
from core.pattern_analysis.phrase_manager import PhraseManager
from fastapi import Depends

# --- Configuration ---
DB_PATH_ENV = os.getenv("JAZZ_DB_PATH", r"C:\polytech\Diploma\wjazzd.db")
DB_PATH = os.path.abspath(DB_PATH_ENV)

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database not found at configured path: {DB_PATH}")

# --- Dependency Providers ---

@lru_cache()
def get_db_path() -> str:
    return DB_PATH

@lru_cache()
def get_pattern_analyzer() -> PatternAnalyzer:
    print("Initializing PatternAnalyzer singleton...")
    return PatternAnalyzer()

@lru_cache()
def get_chord_parser() -> DatabaseChordParser:
    print("Initializing DatabaseChordParser singleton...")
    resolved_db_path = get_db_path()
    return DatabaseChordParser(db_path=resolved_db_path)

@lru_cache()
def get_phrase_manager(
    analyzer: PatternAnalyzer = Depends(get_pattern_analyzer)
) -> PhraseManager:
    print("Initializing PhraseManager singleton...")
    resolved_db_path = get_db_path()
    return PhraseManager(db_path=resolved_db_path, pattern_analyzer=analyzer)
