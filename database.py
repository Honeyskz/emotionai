"""
database.py — Historial emocional con SQLite
"""

import sqlite3
import time
import logging
from contextlib import contextmanager

DB_PATH = "emotionai.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Crea la tabla si no existe."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion    TEXT    NOT NULL,
                confidence REAL    NOT NULL,
                timestamp  REAL    NOT NULL
            )
        """)
        # Índice para acelerar consultas de historial
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON detections (timestamp DESC)
        """)
    logging.info("✅ Base de datos inicializada")


# Throttle: guardar como máximo 1 registro por segundo para no llenar el disco
_last_save: float = 0.0

def save_detection(emotion: str, confidence: float):
    global _last_save
    now = time.time()
    if now - _last_save < 1.0:
        return
    _last_save = now

    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO detections (emotion, confidence, timestamp) VALUES (?, ?, ?)",
                (emotion, round(confidence, 4), now),
            )
    except Exception as e:
        logging.error(f"Error guardando detección: {e}")


def get_history(limit: int = 50) -> list[dict]:
    """Devuelve las últimas `limit` detecciones, más reciente primero."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT emotion, confidence, timestamp
                   FROM detections
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "emotion":    r["emotion"],
                "confidence": r["confidence"],
                "timestamp":  r["timestamp"],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"Error leyendo historial: {e}")
        return []


def get_stats() -> dict:
    """Cuenta detecciones por emoción en las últimas 24 horas."""
    since = time.time() - 86400
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT emotion, COUNT(*) as cnt, AVG(confidence) as avg_conf
                   FROM detections
                   WHERE timestamp >= ?
                   GROUP BY emotion
                   ORDER BY cnt DESC""",
                (since,),
            ).fetchall()
        return {
            r["emotion"]: {
                "count":       r["cnt"],
                "avg_confidence": round(r["avg_conf"], 3),
            }
            for r in rows
        }
    except Exception as e:
        logging.error(f"Error calculando stats: {e}")
        return {}


def purge_old(days: int = 7):
    """Borra registros más antiguos que `days` días (mantenimiento)."""
    cutoff = time.time() - days * 86400
    try:
        with get_conn() as conn:
            deleted = conn.execute(
                "DELETE FROM detections WHERE timestamp < ?", (cutoff,)
            ).rowcount
        logging.info(f"Purga: {deleted} registros eliminados (>{days} días)")
    except Exception as e:
        logging.error(f"Error en purge_old: {e}")
