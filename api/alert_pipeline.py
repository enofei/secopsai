"""
Alert Pipeline — SecOpsAI
Enriches detections with threat intelligence and sends notifications.
Implements mock automated containment.
"""

import json
import os
import requests
import logging
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("secopsai.alerts")

# Load from environment variables in production
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")


def enrich_with_virustotal(ip_address: str) -> dict:
    """
    Looks up an IP address on VirusTotal to get threat intelligence.
    Returns enrichment data if API key is configured,
    otherwise returns a mock response for demo purposes.
    """
    if not VIRUSTOTAL_API_KEY or not ip_address:
        return {
            "source": "mock",
            "malicious_votes": 3,
            "harmless_votes": 45,
            "reputation_score": -3,
            "note": "Configure VIRUSTOTAL_API_KEY for real enrichment"
        }

    try:
        url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip_address}"
        headers = {"x-apikey": VIRUSTOTAL_API_KEY}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            stats = data.get("data", {}).get("attributes", {}).get(
                "last_analysis_stats", {}
            )
            return {
                "source": "virustotal",
                "malicious_votes": stats.get("malicious", 0),
                "harmless_votes":  stats.get("harmless", 0),
                "reputation_score": data.get("data", {}).get(
                    "attributes", {}
                ).get("reputation", 0)
            }
    except Exception as e:
        logger.warning(f"VirusTotal lookup failed: {e}")

    return {"source": "error", "note": "Enrichment lookup failed"}


def send_slack_alert(alert_payload: dict) -> bool:
    """
    Sends an alert notification to Slack via webhook.
    Configure SLACK_WEBHOOK_URL environment variable to enable.
    """
    if not SLACK_WEBHOOK_URL:
        logger.info(f"[ALERT] Slack not configured — alert logged locally only")
        logger.info(f"[ALERT] {json.dumps(alert_payload, indent=2)}")
        return False

    try:
        message = {
            "text": f"🚨 *SecOpsAI Alert* — {alert_payload['attack_type']} detected",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*🚨 SecOpsAI Detection Alert*\n"
                            f"*Attack Type:* {alert_payload['attack_type']}\n"
                            f"*Confidence:* {alert_payload['confidence']:.2%}\n"
                            f"*Source IP:* {alert_payload.get('source_ip', 'Unknown')}\n"
                            f"*Time:* {alert_payload['timestamp']}\n"
                            f"*Threat Intel:* {alert_payload.get('enrichment', {})}"
                        )
                    }
                }
            ]
        }
        response = requests.post(SLACK_WEBHOOK_URL, json=message, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Slack alert failed: {e}")
        return False


def execute_containment(source_ip: str, confidence: float) -> dict:
    """
    Executes automated containment when confidence is high.
    This is a mock implementation — in production this would
    call a firewall API, EDR platform, or network controller.
    Confidence threshold of 0.90+ triggers containment.
    """
    if confidence < 0.90:
        return {
            "action": "none",
            "reason": f"Confidence {confidence:.2%} below containment threshold (90%)"
        }

    # Mock containment — log the action that would be taken
    containment_action = {
        "action":     "firewall_block",
        "target_ip":  source_ip or "unknown",
        "rule":       f"DROP src_ip={source_ip}",
        "timestamp":  datetime.utcnow().isoformat(),
        "status":     "mock_executed",
        "note": (
            "In production: calls firewall API to block IP. "
            "Logged for audit trail."
        )
    }

    logger.info(f"[CONTAINMENT] Mock firewall rule: {containment_action['rule']}")
    return containment_action


def process_alert(
    detection_result: dict,
    source_ip: str = None,
    confidence: float = 0.0
) -> dict:
    """
    Main alert processing function.
    Called whenever the model detects an attack.

    Steps:
    1. Enrich with threat intelligence
    2. Build alert payload
    3. Send Slack notification
    4. Execute containment if confidence is high
    5. Return full alert record for audit logging
    """
    timestamp = datetime.utcnow().isoformat()

    enrichment = enrich_with_virustotal(source_ip)

    alert_payload = {
        "timestamp":   timestamp,
        "attack_type": detection_result.get("prediction", "UNKNOWN"),
        "confidence":  confidence,
        "source_ip":   source_ip or "unknown",
        "enrichment":  enrichment
    }

    send_slack_alert(alert_payload)

    containment = execute_containment(source_ip, confidence)

    return {
        "alert":       alert_payload,
        "containment": containment,
        "notified":    bool(SLACK_WEBHOOK_URL)
    }
