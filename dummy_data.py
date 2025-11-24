import os
import random
import calendar
from datetime import datetime, timedelta, timezone
from typing import List

import httpx
from httpx import ConnectError

# Basic script to seed /api/retrain/sleep and /api/retrain/stress with dummy data.

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
EMAIL = os.getenv("DUMMY_EMAIL", "user@example.com")
PASSWORD = os.getenv("DUMMY_PASSWORD", "asdf1234")


# ============================================================
# 0. 공통: 로그인해서 Authorization 헤더 얻기
# ============================================================

def get_auth_headers(email: str = EMAIL, password: str = PASSWORD) -> dict:
    """Sign up (idempotent) and return Authorization header."""
    try:
        with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
            # Sign-up is idempotent for an existing user; ignore failure if already registered.
            client.post(
                "/api/auth/signup",
                json={"email": email, "password": password, "password2": password},
            )
            login = client.post(
                "/api/auth/login",
                json={"email": email, "password": password},
            )
            login.raise_for_status()
            body = login.json()
            token = f"{body['token_type']} {body['access_token']}"
            return {"Authorization": token}
    except ConnectError as exc:
        raise SystemExit(
            f"Cannot reach API at {API_BASE_URL}. "
            "Start the FastAPI server or set API_BASE_URL to the correct address."
        ) from exc


# ============================================================
# 1. SLEEP 더미 생성 로직 (Wake, N1, N2, N3, REM)
#    → random_sleep_payload 에서 사용
# ============================================================

def get_sleep_stage_cycle(min_from_start: int) -> int:
    """
    90분 수면 사이클 반복:
    Wake(0) → N1(1) → N2(2) → N3(3) → REM(4)
    """
    cycle_len = 90
    pos = min_from_start % cycle_len

    if pos < 5:
        return 0  # Wake
    elif pos < 15:
        return 1  # N1
    elif pos < 55:
        return 2  # N2
    elif pos < 75:
        return 3  # N3
    else:
        return 4  # REM


def simulate_sleep_values(stage: int) -> tuple[float, float, float, float]:
    """
    수면 단계별 HR / ACC 분포 생성.
    SaO2, BVP는 전부 0으로 둘 예정.
    """
    hr_ranges = {
        0: (78, 90),  # Wake
        1: (70, 80),  # N1
        2: (65, 75),  # N2
        3: (55, 65),  # N3 (deep)
        4: (72, 85),  # REM
    }
    hr = random.uniform(*hr_ranges[stage])

    acc_noise = {
        0: 0.20,
        1: 0.12,
        2: 0.08,
        3: 0.04,
        4: 0.10,
    }[stage]

    acc_x = random.gauss(0.0, acc_noise)
    acc_y = random.gauss(0.0, acc_noise)
    acc_z = random.gauss(1.0, acc_noise)  # gravity + noise

    return acc_x, acc_y, acc_z, hr


def random_sleep_payload(sleep_start_time: datetime, idx: int) -> dict:
    """
    - idx: 수면 시작으로부터 지난 "초"
    - timestamp: ms 단위 offset (백엔드가 measured_at = sleep_start_time + offset 사용)

    90분 수면 사이클은 '분' 기준이라 stage 계산은 idx // 60을 사용.
    """
    minutes_from_start = idx // 60
    stage = get_sleep_stage_cycle(minutes_from_start)
    acc_x, acc_y, acc_z, hr = simulate_sleep_values(stage)

    timestamp_ms = idx * 1_000  # 1초 간격

    return {
        "sleep_start_time": sleep_start_time.isoformat(),
        "timestamp": timestamp_ms,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "hr": hr,
        "sleep_stage": stage,
    }


def create_meaningful_sleep_logs(
    headers: dict,
    year: int,
    month: int,
    total_count: int = 14_400,  # 딱 14,400개 생성
) -> None:
    """
    한 달 중 특정 날짜의 4시간짜리(14,400초) 수면 세션 1개 생성.
    total_count만큼 정확히 생성한다.
    """
    # 예시: 그 달 1일, KST 23:00을 UTC로 환산 (KST - 9h)
    sleep_start = datetime(year, month, 1, 14, 0, tzinfo=timezone.utc)

    with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
        for idx in range(total_count):
            payload = random_sleep_payload(sleep_start, idx)
            r = client.post("/api/retrain/sleep", headers=headers, json=payload)
            if r.status_code >= 400:
                print(
                    f"[SLEEP] idx={idx} 실패: {r.status_code} {r.text}"
                )
                break

    print(f"[SLEEP] {year}-{month:02d} 수면 로그 생성 완료 (total={total_count})")


# ============================================================
# 2. STRESS 더미 생성 로직 (HR ↔ HRV 연관 있게)
#    → random_stress_payload 에서 사용
# ============================================================

def circadian_hr(hour: int) -> float:
    """
    단순 일중(circadian) HR 패턴
    """
    if 0 <= hour < 6:
        return 60.0
    elif 6 <= hour < 10:
        return 70.0
    elif 10 <= hour < 18:
        return 80.0
    elif 18 <= hour < 22:
        return 75.0
    else:
        return 65.0


def simulate_hrv_from_hr(hr: float, stressed: bool) -> tuple[float, float]:
    """
    HR 기반으로 SDNN / RMSSD를 생리적으로 말 되게 생성.
    - HR ↑ → HRV(특히 RMSSD, SDNN) ↓
    - stressed=True 이면 HRV 추가 감소
    """
    ans = 1.5 - (hr - 60.0) / 40.0
    ans = max(0.2, min(1.8, ans))

    if stressed:
        ans *= random.uniform(0.4, 0.7)

    rmssd_mean = ans * 40.0
    sdnn_mean = ans * 50.0

    rmssd = max(5.0, random.gauss(rmssd_mean, 5.0))
    sdnn = max(5.0, random.gauss(sdnn_mean, 5.0))

    return sdnn, rmssd


def random_stress_payload(
    base_time: datetime,
    idx: int,
    event_points: list[int],
    event_duration_slots: int = 6,
) -> dict:
    """
    기존 random_stress_payload를 '의미 있는 스트레스 패턴' 기반으로 교체한 버전.

    - 10분 간격 슬롯을 가정
    - base_time + idx*10분 = 해당 측정의 시간
    - 일부 슬롯은 스트레스 이벤트 구간으로 HR ↑, HRV ↓, ACC ↑
    """
    created_at = base_time + timedelta(minutes=idx * 10)
    hour = created_at.hour

    base_hr = circadian_hr(hour)
    stressed = any(start <= idx < start + event_duration_slots for start in event_points)

    if stressed:
        hr = base_hr + random.uniform(12.0, 20.0)
    else:
        hr = base_hr + random.uniform(-5.0, 5.0)

    sdnn, rmssd = simulate_hrv_from_hr(hr, stressed)

    if stressed:
        acc_mag_mean = random.uniform(0.8, 1.4)
        acc_std = random.uniform(0.15, 0.3)
    else:
        acc_mag_mean = random.uniform(0.3, 0.6)
        acc_std = random.uniform(0.05, 0.12)

    return {
        "heart_rate_bpm": hr,
        "hrv_sdnn_ms": sdnn,
        "hrv_rmssd_ms": rmssd,
        "acc_x_mean": random.gauss(0.0, acc_std),
        "acc_y_mean": random.gauss(0.0, acc_std),
        "acc_z_mean": random.gauss(1.0, acc_std),
        "acc_mag_mean": acc_mag_mean,
        "acc_mag_std": acc_std,
    }

def create_meaningful_stress_logs(
    headers: dict,
    year: int,
    month: int,
    total_count: int = 1_008,  # 1008개 정확히 생성
) -> None:
    """
    10분 간격 스트레스 로그 생성.
    total_count 만큼 생성 (기본: 7일치 10분 간격 = 1008개)
    """
    base_time = datetime(year, month, 1, 0, 0, tzinfo=timezone.utc)

    # total_count 범위 안에서 스트레스 이벤트 시작 인덱스 몇 개 선정
    events_per_week = 5
    event_points = sorted(
        random.sample(range(total_count), k=min(events_per_week, total_count))
    )

    with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
        for idx in range(total_count):
            payload = random_stress_payload(base_time, idx, event_points)
            r = client.post("/api/retrain/stress", headers=headers, json=payload)
            if r.status_code >= 400:
                print(
                    f"[STRESS] idx={idx} 실패: {r.status_code} {r.text}"
                )
                break

    print(f"[STRESS] {year}-{month:02d} 스트레스 로그 생성 완료 (total={total_count})")


def make_meaningful_retrain_logs(
    year: int,
    month: int,
    sleep_total: int = 5,
    stress_total: int = 5,
) -> None:
    headers = get_auth_headers()
    create_meaningful_sleep_logs(
        headers=headers,
        year=year,
        month=month,
        total_count=sleep_total,
    )
    create_meaningful_stress_logs(
        headers=headers,
        year=year,
        month=month,
        total_count=stress_total,
    )


if __name__ == "__main__":
    make_meaningful_retrain_logs(2025, 11)
