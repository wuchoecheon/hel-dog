import pytest
from fastapi.testclient import TestClient
from src.main import app
import xml.etree.ElementTree as ET

client = TestClient(app)

def get_token():
    _ = client.post(
        "/api/auth/signup",
        json={
            "email": "user@example.com",
            "password": "asdf1234",
            "password2": "asdf1234",
        }
    )

    login = client.post(
        "/api/auth/login",
        json={
            "email": "user@example.com",
            "password": "asdf1234",
        }
    )

    body = login.json()
    token = "{} {}".format(body["token_type"], body["access_token"])
    return token


def setup(robot_id: str = "robot", device_id: str = "device"):
    token = get_token()

    _ = client.post(
        "/api/robot/register",
        headers={"Authorization": token},
        json={
            "robot_id": robot_id,
        }
    )

    _ = client.post(
        "/api/device/register",
        headers={"Authorization": token},
        json={
            "device_id": device_id,
        }
    )

    return token


def test_fhir_export_basic():
    token = setup()

    client.post("/api/cough/robot")
    client.post("/api/cough/robot")

    client.post(
        "/api/sleep/device",
        headers={"Authorization": token},
        json={"label": 3}
    )

    client.post("/api/fall/robot")

    client.post("/api/stress/device")

    client.post("/api/posture/robot")

    res = client.get(
        "/api/fhir/export",
        headers={"Authorization": token},
    )

    assert res.status_code == 200

    xml_bytes = res.content
    root = ET.fromstring(xml_bytes)

    ns = {"f": "http://hl7.org/fhir"}

    patient_nodes = root.findall("f:entry/f:resource/f:Patient", ns)
    assert len(patient_nodes) == 1

    observations = root.findall("f:entry/f:resource/f:Observation", ns)

    obs_ids = []
    for obs in observations:
        id_node = obs.find("f:id", ns)
        if id_node is not None and "value" in id_node.attrib:
            obs_ids.append(id_node.attrib["value"])

    assert "cough-3h" in obs_ids
    assert "sleep-7h" in obs_ids
    assert "fall-24h" in obs_ids
    assert "stress-count-12h" in obs_ids
    assert "posture-3h" in obs_ids
