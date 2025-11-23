from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Annotated
from fastapi import Depends

from src.auth.models import User
from src.auth.utils import get_user

from src.cough.service import summarize_cough_last_3h
from src.stress.service import summarize_stress_last_12h
from src.sleep.service import summarize_sleep_last_7h
from src.posture.service import summarize_posture_last_3h
from src.fall.service import summarize_fall_last_24h
from src.health.service import calc_health_score

from .builder import build_fhir_bundle_xml

@dataclass
class CoughSummary:
    total_3h: int

@dataclass
class StressSummary:
    count_12h: int
    stressed: bool

@dataclass
class SleepSummary:
    score_7h: float

@dataclass
class PostureSummary:
    total_3h: int
    by_label: dict

@dataclass
class FallSummary:
    last_fall_at: datetime | None
    fall_count_24h: int

@dataclass
class HealthScoreSummary:
    last_fall_at: datetime | None
    fall_count_24h: int

@dataclass
class HealthDataBundle:
    cough: CoughSummary | None = None
    stress: StressSummary | None = None
    sleep: SleepSummary | None = None
    posture: PostureSummary | None = None
    fall: FallSummary | None = None

def get_cough_summary(user_email: str, db: Session) -> CoughSummary | None:
    summary = summarize_cough_last_3h(db, user_email)
    if summary is None:
        return None

    return CoughSummary(
        total_3h=summary["cough_num"]
    )

def get_stress_summary(user_email: str, db: Session) -> StressSummary | None:
    summary = summarize_stress_last_12h(db, user_email)
    if summary is None:
        return None

    return StressSummary(
        count_12h=summary["count_last_12h"],
        stressed=summary["stressed"]
    )

def get_sleep_summary(user_email: str, db: Session) -> SleepSummary | None:
    summary = summarize_sleep_last_7h(db, user_email)
    if summary is None:
        return None

    return SleepSummary(
        score_7h=summary["sleep_score"]
    )

def get_posture_summary(user_email: str, db: Session) -> PostureSummary | None:
    summary = summarize_posture_last_3h(db, user_email)
    if summary is None:
        return None

    return PostureSummary(
        total_3h=summary["total"],
        by_label=summary["counts_by_label"]
    )

def get_fall_summary(user_email: str, db: Session) -> FallSummary | None:
    summary = summarize_fall_last_24h(db, user_email)
    if summary is None:
        return None

    return FallSummary(
        last_fall_at=summary["last_fall_at"],
        fall_count_24h=summary["fall_count_24h"]
    )

def build_fhir_bundle_for_user(user: User, db: Session) -> bytes | None:
    cough = get_cough_summary(user_email=user.email, db=db)
    stress = get_stress_summary(user_email=user.email, db=db)
    sleep = get_sleep_summary(user_email=user.email, db=db)
    posture = get_posture_summary(user_email=user.email, db=db)
    fall = get_fall_summary(user_email=user.email, db=db)

    if not any([cough, stress, sleep, posture, fall]):
        return None

    data_bundle = HealthDataBundle(
        cough=cough,
        stress=stress,
        sleep=sleep,
        posture=posture,
        fall=fall,
    )

    xml_bytes = build_fhir_bundle_xml(
        user_id=user.email,
        data=data_bundle
    )

    return xml_bytes
