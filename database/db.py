"""Database connection and initialization manager for SQLite."""

import sqlite3
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

from config.settings import DATABASE_PATH
from database.models import DATABASE_SCHEMA_SQL
from utils.logging_config import logger


class DatabaseManager:
    """Manages SQLite connections and schema migrations."""

    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self._ensure_db_dir()
        self.init_database()

    def _ensure_db_dir(self) -> None:
        """Create parent directory for SQLite DB if missing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_database(self) -> None:
        """Execute DDL statements to ensure all required tables and indexes exist."""
        try:
            with self.get_connection() as conn:
                conn.executescript(DATABASE_SCHEMA_SQL)
                
                # Check for recommendations table column migrations
                cursor = conn.execute("PRAGMA table_info(recommendations)")
                rec_cols = [row["name"] for row in cursor.fetchall()]
                if "status" not in rec_cols:
                    conn.execute("ALTER TABLE recommendations ADD COLUMN status TEXT DEFAULT 'pending'")
                if "instructor_notes" not in rec_cols:
                    conn.execute("ALTER TABLE recommendations ADD COLUMN instructor_notes TEXT DEFAULT ''")
                
                conn.commit()
            logger.info(f"Database initialized successfully at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an active SQLite connection with row factory enabled."""
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=20.0
        )
        conn.row_factory = sqlite3.Row
        # Enable foreign keys and WAL mode for concurrent reads in Streamlit
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
        finally:
            conn.close()


# Global database manager instance
db_manager = DatabaseManager()
