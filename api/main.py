"""
SecOpsAI Detection API
FastAPI service exposing the hardened XGBoost detection engine.
Implements JWT auth, rate limiting, input validation, audit logging.
"""

import pickle
import logging
import time
import json
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import pandas as pd

from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, field_validator
from typing import Optional

from api.auth import authenticate_user, create_access_token, get_current_user
from api.alert_pipeline import process_alert


# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("secopsai.api")

# Audit log — append-only file for every detection request
AUDIT_LOG_PATH = "logs/audit.log"
os.makedirs("logs", exist_ok=True)


def write_audit_log(entry: dict):
    with open(AUDIT_LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + "\n")


# ── Rate limiter ───────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="SecOpsAI Detection API",
    description="AI-powered network threat detection system",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load hardened model ────────────────────────────────────────
MODEL_PATH = "models/xgboost_detector_hardened_v2.pkl"
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Hardened model loaded from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model not found at {MODEL_PATH}")
    model = None


# ── Input schema ───────────────────────────────────────────────
class NetworkFlowFeatures(BaseModel):
    """
    Input schema for detection requests.
    All fields are validated — rejects malformed or out-of-range inputs.
    """
    byte_ratio:            float = Field(..., ge=0.0, le=1.0)
    timing_entropy:        float = Field(..., ge=0.0)
    size_consistency:      float = Field(..., ge=0.0)
    connection_symmetry:   float = Field(..., ge=0.0, le=1.0)
    flow_speed:            float = Field(..., ge=0.0)
    Flow_Duration:         float = Field(..., ge=0.0, alias="Flow Duration")
    Total_Fwd_Packets:     float = Field(..., ge=0.0, alias="Total Fwd Packets")
    Total_Bwd_Packets:     float = Field(..., ge=0.0, alias="Total Backward Packets")
    Flow_Bytes_s:          float = Field(..., ge=0.0, alias="Flow Bytes/s")
    Flow_Packets_s:        float = Field(..., ge=0.0, alias="Flow Packets/s")
    Flow_IAT_Mean:         float = Field(..., alias="Flow IAT Mean")
    Flow_IAT_Std:          float = Field(..., ge=0.0, alias="Flow IAT Std")
    Fwd_IAT_Mean:          float = Field(..., alias="Fwd IAT Mean")
    Bwd_IAT_Mean:          float = Field(..., alias="Bwd IAT Mean")
    Packet_Length_Mean:    float = Field(..., ge=0.0, alias="Packet Length Mean")
    Packet_Length_Std:     float = Field(..., ge=0.0, alias="Packet Length Std")
    Packet_Length_Variance:float = Field(..., ge=0.0, alias="Packet Length Variance")
    Average_Packet_Size:   float = Field(..., ge=0.0, alias="Average Packet Size")
    SYN_Flag_Count:        float = Field(..., ge=0.0, alias="SYN Flag Count")
    ACK_Flag_Count:        float = Field(..., ge=0.0, alias="ACK Flag Count")
    PSH_Flag_Count:        float = Field(..., ge=0.0, alias="PSH Flag Count")
    FIN_Flag_Count:        float = Field(..., ge=0.0, alias="FIN Flag Count")
    source_ip: Optional[str] = None

    model_config = {"populate_by_name": True}


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Public health endpoint — no auth required."""
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "timestamp":    datetime.utcnow().isoformat(),
        "version":      "1.0.0"
    }


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Returns a JWT token for valid credentials.
    Use this token as Bearer auth on all other endpoints.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/detect")
@limiter.limit("60/minute")
async def detect(
    request: Request,
    features: NetworkFlowFeatures,
    current_user: dict = Depends(get_current_user)
):
    """
    Main detection endpoint.
    Requires Bearer token authentication.
    Rate limited to 60 requests per minute per IP.
    Returns prediction, confidence score, and alert details.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Build feature vector in correct order
    feature_order = [
        'byte_ratio', 'timing_entropy', 'size_consistency',
        'connection_symmetry', 'flow_speed', 'Flow Duration',
        'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
        'Packet Length Mean', 'Packet Length Std',
        'Packet Length Variance', 'Average Packet Size',
        'SYN Flag Count', 'ACK Flag Count',
        'PSH Flag Count', 'FIN Flag Count'
    ]

    feature_dict = {
        'byte_ratio':              features.byte_ratio,
        'timing_entropy':          features.timing_entropy,
        'size_consistency':        features.size_consistency,
        'connection_symmetry':     features.connection_symmetry,
        'flow_speed':              features.flow_speed,
        'Flow Duration':           features.Flow_Duration,
        'Total Fwd Packets':       features.Total_Fwd_Packets,
        'Total Backward Packets':  features.Total_Bwd_Packets,
        'Flow Bytes/s':            features.Flow_Bytes_s,
        'Flow Packets/s':          features.Flow_Packets_s,
        'Flow IAT Mean':           features.Flow_IAT_Mean,
        'Flow IAT Std':            features.Flow_IAT_Std,
        'Fwd IAT Mean':            features.Fwd_IAT_Mean,
        'Bwd IAT Mean':            features.Bwd_IAT_Mean,
        'Packet Length Mean':      features.Packet_Length_Mean,
        'Packet Length Std':       features.Packet_Length_Std,
        'Packet Length Variance':  features.Packet_Length_Variance,
        'Average Packet Size':     features.Average_Packet_Size,
        'SYN Flag Count':          features.SYN_Flag_Count,
        'ACK Flag Count':          features.ACK_Flag_Count,
        'PSH Flag Count':          features.PSH_Flag_Count,
        'FIN Flag Count':          features.FIN_Flag_Count,
    }

    X = pd.DataFrame([feature_dict])[feature_order]

    proba      = model.predict_proba(X)[0]
    prediction = int(np.argmax(proba))
    confidence = float(proba[prediction])

    result = {
        "prediction":    "ATTACK" if prediction == 1 else "BENIGN",
        "confidence":    round(confidence, 4),
        "label_code":    prediction,
        "latency_ms":    round((time.time() - start_time) * 1000, 2),
        "model_version": "xgboost_hardened_v2",
        "timestamp":     datetime.utcnow().isoformat()
    }

    # Fire alert pipeline if attack detected
    alert_info = None
    if prediction == 1:
        alert_info = process_alert(
            detection_result=result,
            source_ip=features.source_ip,
            confidence=confidence,
            raw_features=feature_dict
        )
        result["alert"] = alert_info

    # Write audit log entry
    write_audit_log({
        "timestamp":  result["timestamp"],
        "user":       current_user["username"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "source_ip":  features.source_ip,
        "latency_ms": result["latency_ms"]
    })

    return result


@app.get("/model/info")
async def model_info(current_user: dict = Depends(get_current_user)):
    """Returns metadata about the currently loaded model."""
    return {
        "model_type":    "XGBoost (adversarially hardened v2)",
        "feature_count": 22,
        "classes":       ["BENIGN", "ATTACK"],
        "hardening":     "2 rounds iterative adversarial training",
        "attacks_survived": "4/5 (ZOO, HopSkipJump, Boundary, Manual Perturbation)",
        "clean_f1":      0.9906
    }
