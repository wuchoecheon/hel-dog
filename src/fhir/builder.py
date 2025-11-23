from xml.etree.ElementTree import Element, SubElement, tostring
from datetime import datetime

FHIR_NS = "http://hl7.org/fhir"


def _el(tag: str, parent: Element | None = None, value: str | None = None) -> Element:
    """FHIR 네임스페이스를 포함한 Element 생성 헬퍼."""
    if parent is None:
        el = Element(f"{{{FHIR_NS}}}{tag}")
    else:
        el = SubElement(parent, f"{{{FHIR_NS}}}{tag}")
    if value is not None:
        el.set("value", str(value))
    return el


def _add_patient_entry(bundle: Element, user_id: str) -> None:
    """
    간단한 Patient 리소스 추가.
    user_id에는 email을 그대로 사용해도 무방함 (FHIR에서 id는 문자열).
    """
    entry = SubElement(bundle, f"{{{FHIR_NS}}}entry")
    res = _el("resource", entry)
    patient = _el("Patient", res)
    _el("id", patient, f"user-{user_id}")


def _add_observation_entry(
    bundle: Element,
    user_id: str,
    obs_id: str,
    code_system: str,
    code: str,
    display: str,
    value: float | int | str,
    unit: str | None = None,
    value_type: str = "Quantity",  # "Quantity" or "CodeableConcept"
    effective: datetime | None = None,
) -> None:
    entry = SubElement(bundle, f"{{{FHIR_NS}}}entry")
    res = _el("resource", entry)
    obs = _el("Observation", res)
    _el("id", obs, obs_id)

    # subject -> Patient 참조
    subject = _el("subject", obs)
    _el("reference", subject, f"Patient/user-{user_id}")

    # code
    code_el = _el("code", obs)
    coding = _el("coding", code_el)
    _el("system", coding, code_system)
    _el("code", coding, code)
    _el("display", coding, display)

    # value
    if value_type == "Quantity":
        vq = _el("valueQuantity", obs)
        _el("value", vq, value)
        if unit:
            _el("unit", vq, unit)
    else:
        vcc = _el("valueCodeableConcept", obs)
        coding2 = _el("coding", vcc)
        _el("system", coding2, code_system)
        _el("code", coding2, value)
        _el("display", coding2, str(value))

    # 시간 정보
    effective = effective or datetime.utcnow()
    _el("effectiveDateTime", obs, effective.isoformat(timespec="seconds") + "Z")


def build_fhir_bundle_xml(user_id: str, data) -> bytes:
    """
    HealthDataBundle(헬독 요약 데이터)을 FHIR Bundle(XML)로 직렬화.
    user_id는 email 문자열을 그대로 사용.
    """
    bundle = Element(f"{{{FHIR_NS}}}Bundle")
    _el("type", bundle, "collection")

    # Patient 추가
    _add_patient_entry(bundle, user_id=user_id)

    # 기침: 최근 3시간 기침 횟수
    if data.cough:
        _add_observation_entry(
            bundle=bundle,
            user_id=user_id,
            obs_id="cough-3h",
            code_system="https://hel-dog.app/codes/health",
            code="cough-count-3h",
            display="Cough count in last 3h",
            value=data.cough.total_3h,
            unit="count",
            value_type="Quantity",
        )

    # 스트레스: 12시간 내 스트레스 로그 횟수, 스트레스 여부
    if data.stress:
        # 12시간 내 스트레스 이벤트 횟수
        _add_observation_entry(
            bundle=bundle,
            user_id=user_id,
            obs_id="stress-count-12h",
            code_system="https://hel-dog.app/codes/health",
            code="stress-count-12h",
            display="Stress event count in last 12h",
            value=data.stress.count_12h,
            unit="count",
            value_type="Quantity",
        )

        # 스트레스 상태
        _add_observation_entry(
            bundle=bundle,
            user_id=user_id,
            obs_id="stress-status-12h",
            code_system="https://hel-dog.app/codes/health",
            code="stress-status-12h",
            display="Stress state in last 12h",
            value="stressed" if data.stress.stressed else "not-stressed",
            unit=None,
            value_type="CodeableConcept",
        )

    # 수면질: 최근 7시간 sleep score
    if data.sleep:
        _add_observation_entry(
            bundle=bundle,
            user_id=user_id,
            obs_id="sleep-7h",
            code_system="https://hel-dog.app/codes/health",
            code="sleep-score-7h",
            display="Sleep score in last 7h",
            value=round(data.sleep.score_7h, 1),
            unit="score",
            value_type="Quantity",
        )

    # 앉은 자세: 최근 3시간 나쁜자세 총 횟수
    if data.posture:
        _add_observation_entry(
            bundle=bundle,
            user_id=user_id,
            obs_id="posture-3h",
            code_system="https://hel-dog.app/codes/health",
            code="bad-posture-count-3h",
            display="Bad posture count in last 3h",
            value=data.posture.total_3h,
            unit="count",
            value_type="Quantity",
        )

    # 낙상: 최근 24시간 낙상 횟수, 최근 낙상 시각
    if data.fall:
        _add_observation_entry(
            bundle=bundle,
            user_id=user_id,
            obs_id="fall-24h",
            code_system="https://hel-dog.app/codes/health",
            code="fall-event-count-24h",
            display="Fall event count in last 24h",
            value=data.fall.fall_count_24h,
            unit="count",
            value_type="Quantity",
        )
        # 최근 낙상 이벤트 시각
        if data.fall.last_fall_at:
            _add_observation_entry(
                bundle=bundle,
                user_id=user_id,
                obs_id="fall-last",
                code_system="https://hel-dog.app/codes/health",
                code="fall-last-event",
                display="Last fall event time",
                value="fall",
                unit=None,
                value_type="CodeableConcept",
                effective=data.fall.last_fall_at,
            )

    return tostring(bundle, encoding="utf-8", xml_declaration=True)
