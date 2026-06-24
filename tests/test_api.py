"""
API Integration Tests — SecOpsAI
Tests authentication, detection endpoint, and rate limiting.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_FEATURES = {
    "byte_ratio": 0.9,
    "timing_entropy": 0.05,
    "size_consistency": 0.1,
    "connection_symmetry": 0.8,
    "flow_speed": 5.0,
    "Flow Duration": 950000,
    "Total Fwd Packets": 2,
    "Total Backward Packets": 1,
    "Flow Bytes/s": 50,
    "Flow Packets/s": 0.002,
    "Flow IAT Mean": 980000,
    "Flow IAT Std": 3000,
    "Fwd IAT Mean": 980000,
    "Bwd IAT Mean": 980000,
    "Packet Length Mean": 75,
    "Packet Length Std": 10,
    "Packet Length Variance": 100,
    "Average Packet Size": 75,
    "SYN Flag Count": 1,
    "ACK Flag Count": 1,
    "PSH Flag Count": 1,
    "FIN Flag Count": 0,
    "source_ip": "192.168.1.100"
}


def get_token():
    response = client.post(
        "/token",
        data={"username": "analyst", "password": "secopsai123"}
    )
    return response.json()["access_token"]


def test_health_check():
    """Health endpoint returns 200 without auth."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_login_valid_credentials():
    """Valid credentials return access token."""
    response = client.post(
        "/token",
        data={"username": "analyst", "password": "secopsai123"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_login_invalid_credentials():
    """Invalid credentials return 401."""
    response = client.post(
        "/token",
        data={"username": "analyst", "password": "wrongpassword"}
    )
    assert response.status_code == 401


def test_detect_requires_auth():
    """Detection endpoint returns 401 without token."""
    response = client.post("/detect", json=SAMPLE_FEATURES)
    assert response.status_code == 401


def test_detect_with_valid_token():
    """Detection endpoint returns prediction with valid token."""
    token    = get_token()
    response = client.post(
        "/detect",
        json=SAMPLE_FEATURES,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["ATTACK", "BENIGN"]
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_detect_latency_under_200ms():
    """Inference latency must be under 200ms (p99 requirement)."""
    token    = get_token()
    response = client.post(
        "/detect",
        json=SAMPLE_FEATURES,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    latency = response.json()["latency_ms"]
    assert latency < 200, f"Latency {latency}ms exceeds 200ms requirement"


def test_detect_invalid_input():
    """Invalid input returns 422 validation error."""
    token = get_token()
    response = client.post(
        "/detect",
        json={"invalid_field": "bad_data"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 422


def test_model_info_endpoint():
    """Model info endpoint returns metadata."""
    token    = get_token()
    response = client.get(
        "/model/info",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "clean_f1" in data


def test_metrics_endpoint():
    """Prometheus metrics endpoint returns data."""
    response = client.get("/metrics")
    assert response.status_code == 200
