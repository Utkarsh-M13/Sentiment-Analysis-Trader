# watermark.py
from sqlalchemy import text

def get_watermark(db, key: str, default=None):
    row = db.execute(text("SELECT value FROM pipeline_state WHERE key=:k"), {"k": key}).fetchone()
    return row[0] if row else default

def set_watermark(db, key: str, value: str):
    db.execute(text("""
        INSERT INTO pipeline_state(key, value)
        VALUES (:k, :v)
        ON CONFLICT (key)
        DO UPDATE SET value = EXCLUDED.value, updated_at = now()
    """), {"k": key, "v": value})