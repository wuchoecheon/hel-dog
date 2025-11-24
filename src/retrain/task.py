from pathlib import Path
from src.database import SessionLocal
from src.retrain.service import export_sleep_retrain_npy, export_stress_retrain_npy

def run_monthly_export(user_email: str, year: int, month: int):
    db = SessionLocal()
    try:
        sleep_out = Path()
        stress_out = Path()

        export_sleep_retrain_npy(db, user_email, year, month, sleep_out)
        export_stress_retrain_npy(db, user_email, year, month, stress_out)
    finally:
        db.close()
