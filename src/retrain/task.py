from pathlib import Path
from src.database import SessionLocal
from src.retrain.service import export_sleep_retrain_npy, export_stress_retrain_npy

BASE_MODEL_DIR = Path("/app/model")

def run_monthly_export(user_email: str, year: int, month: int):
    db = SessionLocal()
    try:
        BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        sleep_out = BASE_MODEL_DIR / f"sleep_{user_email}_{year:04d}{month:02d}.npy"
        stress_out = BASE_MODEL_DIR / f"stress_{user_email}_{year:04d}{month:02d}.npy"

        export_sleep_retrain_npy(db, user_email, year, month, sleep_out)
        export_stress_retrain_npy(db, user_email, year, month, stress_out)
    finally:
        db.close()
