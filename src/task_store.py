"""SQLite-backed task persistence for the PDF-to-LaTeX server.

Replaces the in-memory ``tasks: Dict`` with a thin SQLite wrapper so that
tasks survive server restarts and memory stays bounded.  SQLite requires no
external server — just a file on disk.

Cross-platform: works identically on Windows and macOS.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pdf2latex.task_store")

# Default TTL: tasks older than 24 hours are eligible for cleanup
DEFAULT_TTL_SECONDS = 86400


class TaskStore:
    """Persistent task storage backed by SQLite.

    All public methods are thread-safe (SQLite handles locking internally
    when using ``check_same_thread=False``).
    """

    def __init__(self, db_path: Optional[Path] = None, ttl: int = DEFAULT_TTL_SECONDS):
        """Initialize TaskStore.

        Args:
            db_path: Path to the SQLite database file.  Defaults to
                ``data/tasks.db`` relative to the project root.
            ttl: Time-to-live in seconds for old tasks during cleanup.
        """
        if db_path is None:
            db_path = Path("data") / "tasks.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._create_table()
        logger.info("TaskStore initialised: %s", self.db_path)

    def _create_table(self) -> None:
        """Create the tasks table if it does not exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                data        TEXT NOT NULL,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def put(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """Insert or update a task.

        Args:
            task_id: Unique task identifier.
            task_data: Arbitrary JSON-serialisable task dict.
        """
        now = time.time()
        data_json = json.dumps(task_data, default=str)
        self._conn.execute(
            """INSERT INTO tasks (id, data, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at""",
            (task_id, data_json, now, now),
        )
        self._conn.commit()

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a task by ID.

        Args:
            task_id: Unique task identifier.

        Returns:
            Task dict, or ``None`` if not found.
        """
        row = self._conn.execute(
            "SELECT data FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["data"])

    def update_field(self, task_id: str, key: str, value: Any) -> bool:
        """Update a single field inside a stored task dict.

        Args:
            task_id: Unique task identifier.
            key: Field name to update.
            value: New value.

        Returns:
            True if the task existed and was updated.
        """
        task = self.get(task_id)
        if task is None:
            return False
        task[key] = value
        self.put(task_id, task)
        return True

    def delete(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Unique task identifier.

        Returns:
            True if a row was deleted.
        """
        cur = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all stored tasks (most recent first).

        Returns:
            List of task dicts.
        """
        rows = self._conn.execute(
            "SELECT data FROM tasks ORDER BY updated_at DESC"
        ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def contains(self, task_id: str) -> bool:
        """Check whether a task exists.

        Args:
            task_id: Unique task identifier.

        Returns:
            True if the task is in the store.
        """
        row = self._conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup(self) -> int:
        """Remove tasks older than *ttl* seconds.

        Returns:
            Number of tasks deleted.
        """
        cutoff = time.time() - self.ttl
        cur = self._conn.execute(
            "DELETE FROM tasks WHERE created_at < ?", (cutoff,)
        )
        self._conn.commit()
        deleted = cur.rowcount
        if deleted:
            logger.info("TaskStore cleanup: removed %d stale tasks", deleted)
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Dict-like convenience (for easier migration from in-memory dict)
    # ------------------------------------------------------------------

    def __contains__(self, task_id: str) -> bool:
        return self.contains(task_id)

    def __getitem__(self, task_id: str) -> Dict[str, Any]:
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def __setitem__(self, task_id: str, task_data: Dict[str, Any]) -> None:
        self.put(task_id, task_data)
