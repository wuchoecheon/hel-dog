import sys
import uvicorn
from pathlib import Path
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auth import router as auth_router
from src.robot.router import router as robot_router
from src.device import router as device_router
from src.fall import router as fall_router
from src.cough import router as cough_router
from src.sleep import router as sleep_router
from src.health import router as health_router
from src.fhir.router import router as fhir_router
from src.fcm import router as fcm_router
from src.posture import router as posture_router
from src.stress import router as stress_router
from src.retrain.router import router as retrain_router
from src.retrain.scheduler import start_scheduler

app = FastAPI()

app.include_router(auth_router)
app.include_router(robot_router)
app.include_router(device_router)
app.include_router(fall_router)
app.include_router(cough_router)
app.include_router(sleep_router)
app.include_router(fcm_router)
app.include_router(health_router)
app.include_router(fhir_router)
app.include_router(posture_router)
app.include_router(stress_router)
app.include_router(retrain_router)

@app.on_event("startup")
def on_startup():
    start_scheduler()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
