from sqlalchemy.orm import Session

from src.cough.service import summarize_cough_last_3h
from src.stress.service import summarize_stress_last_12h
from src.sleep.service import summarize_sleep_last_7h

def calc_health_score(db: Session, user_email: str) -> dict:
    sleep_summary = summarize_sleep_last_7h(db, user_email)
    if sleep_summary is None:
        sleep_score = 70.0
    else:
        sleep_raw = sleep_summary.get("sleep_score", 70.0)
        sleep_score = max(0.0, min(100.0, float(sleep_raw)))

    cough_summary = summarize_cough_last_3h(db, user_email)
    if cough_summary is None:
        cough_num = 0
    else:
        cough_num = int(cough_summary.get("cough_num", 0))

    if cough_num <= 2:
        cough_score = 100.0
    elif cough_num <= 5:
        cough_score = 80.0
    elif cough_num <= 10:
        cough_score = 50.0
    else:
        cough_score = 20.0

    stress_summary = summarize_stress_last_12h(db, user_email)
    if stress_summary is None:
        stress_count = 0
    else:
        stress_count = int(stress_summary.get("count_last_12h", 0))

    # threshold: 3
    if stress_count <= 2:
        stress_score = 100.0
    elif stress_count <= 5:
        stress_score = 70.0
    elif stress_count <= 10:
        stress_score = 40.0
    else:
        stress_score = 20.0

    health_score = (
        0.5 * sleep_score
        + 0.3 * stress_score
        + 0.2 * cough_score
    )
    health_score = round(health_score, 1)

    if health_score >= 80:
        status = "good"
    elif health_score >= 60:
        status = "moderate"
    else:
        status = "bad"

    return {
        "health_score": health_score,
        "status": status,
        "components": {
            "sleep_score": sleep_score,
            "cough_score": cough_score,
            "cough_count_3h": cough_num,
            "stress_score": stress_score,
            "stress_count_12h": stress_count,
        },
    }
