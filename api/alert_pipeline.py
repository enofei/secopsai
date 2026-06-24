"""
Alert Pipeline — SecOpsAI
Enriches detections with threat intelligence, sends notifications,
and executes four automated containment/response actions:
1. Firewall IP block
2. User account disable
3. NGFW behavioral profile
4. Incident ticket creation + analyst email assignment
"""

import json
import os
import smtplib
import requests
import logging
import hashlib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("secopsai.alerts")

SLACK_WEBHOOK_URL  = os.getenv("SLACK_WEBHOOK_URL", "")
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY", "")
EMAIL_SENDER       = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD     = os.getenv("EMAIL_PASSWORD", "")
EMAIL_ANALYST      = os.getenv("EMAIL_ANALYST", "")

# In-memory ticket store (PostgreSQL in production)
TICKET_STORE = {}


def enrich_with_virustotal(ip_address: str) -> dict:
    if not VIRUSTOTAL_API_KEY or not ip_address:
        return {
            "source":           "mock",
            "malicious_votes":  3,
            "harmless_votes":   45,
            "reputation_score": -3,
            "country":          "Unknown",
            "note": "Configure VIRUSTOTAL_API_KEY for real enrichment"
        }
    try:
        url      = f"https://www.virustotal.com/api/v3/ip_addresses/{ip_address}"
        headers  = {"x-apikey": VIRUSTOTAL_API_KEY}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data  = response.json()
            attrs = data.get("data", {}).get("attributes", {})
            stats = attrs.get("last_analysis_stats", {})
            return {
                "source":           "virustotal",
                "malicious_votes":  stats.get("malicious", 0),
                "harmless_votes":   stats.get("harmless", 0),
                "reputation_score": attrs.get("reputation", 0),
                "country":          attrs.get("country", "Unknown"),
                "network":          attrs.get("network", "Unknown")
            }
    except Exception as e:
        logger.warning(f"VirusTotal lookup failed: {e}")
    return {"source": "error", "note": "Enrichment lookup failed"}


def create_incident_ticket(
    attack_type: str,
    source_ip: str,
    confidence: float,
    enrichment: dict,
    containment_actions: list
) -> dict:
    """
    Creates a structured incident ticket for analyst review.

    In production this would POST to:
    - ServiceNow REST API
    - Jira Service Management
    - PagerDuty Events API
    - TheHive (open-source SOAR)

    For this project we generate a fully structured ticket
    and assign it to the analyst via email.
    """
    ticket_id       = f"INC-{uuid.uuid4().hex[:8].upper()}"
    created_at      = datetime.utcnow().isoformat()

    # Determine severity based on confidence and attack type
    if confidence >= 0.95:
        severity = "CRITICAL"
        sla_hours = 1
    elif confidence >= 0.85:
        severity = "HIGH"
        sla_hours = 4
    elif confidence >= 0.70:
        severity = "MEDIUM"
        sla_hours = 8
    else:
        severity = "LOW"
        sla_hours = 24

    # Determine category from attack type
    category_map = {
        "C2-Beaconing":       "Command & Control",
        "Lateral-Movement":   "Lateral Movement",
        "DNS-Tunnelling":     "Exfiltration",
        "Slow-Exfiltration":  "Exfiltration",
        "ATTACK":             "Malicious Traffic"
    }
    category = category_map.get(attack_type, "Unknown Threat")

    # Recommended investigation steps per MITRE ATT&CK
    investigation_steps = [
        f"1. Verify source IP {source_ip} in asset inventory",
        "2. Check DHCP logs to identify the associated host",
        "3. Review firewall logs for traffic history from this IP",
        "4. Check if the associated user account shows other anomalies",
        "5. Search SIEM for other alerts from this IP in last 24hrs",
        "6. Collect memory dump from source host if available",
        "7. Review DNS query logs for tunnelling indicators",
        "8. Document findings and update ticket with investigation notes"
    ]

    ticket = {
        "ticket_id":             ticket_id,
        "status":                "OPEN",
        "severity":              severity,
        "category":              category,
        "attack_type":           attack_type,
        "source_ip":             source_ip or "unknown",
        "confidence":            round(confidence, 4),
        "created_at":            created_at,
        "sla_deadline":          f"Resolve within {sla_hours} hour(s)",
        "assigned_to":           EMAIL_ANALYST or "analyst@secopsai.local",
        "enrichment":            enrichment,
        "containment_actions":   containment_actions,
        "investigation_steps":   investigation_steps,
        "mitre_techniques": {
            "C2-Beaconing":      "T1071 - Application Layer Protocol",
            "Lateral-Movement":  "T1021 - Remote Services",
            "DNS-Tunnelling":    "T1071.004 - DNS",
            "Slow-Exfiltration": "T1030 - Data Transfer Size Limits",
            "ATTACK":            "T1190 - Exploit Public Facing Application"
        }.get(attack_type, "T1190")
    }

    # Store ticket in memory
    TICKET_STORE[ticket_id] = ticket
    logger.info(f"[TICKET] Created incident ticket: {ticket_id} | Severity: {severity}")
    return ticket


def send_analyst_email(ticket: dict) -> bool:
    """
    Sends a formatted incident ticket to the assigned analyst
    via Gmail SMTP with full investigation context.
    """
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_ANALYST:
        logger.info("[EMAIL] Email not configured — ticket logged locally")
        logger.info(f"[EMAIL] Ticket: {json.dumps(ticket, indent=2)}")
        return False

    try:
        msg             = MIMEMultipart("alternative")
        msg["Subject"]  = (
            f"[{ticket['severity']}] SecOpsAI Incident {ticket['ticket_id']} — "
            f"{ticket['attack_type']} detected from {ticket['source_ip']}"
        )
        msg["From"]     = EMAIL_SENDER
        msg["To"]       = EMAIL_ANALYST

        enrichment   = ticket.get("enrichment", {})
        containments = ticket.get("containment_actions", [])

        containment_html = ""
        for action in containments:
            if action.get("action") != "none":
                containment_html += f"""
                <tr>
                    <td style='padding:8px;border:1px solid #ddd;'>
                        {action.get('method', 'N/A')}
                    </td>
                    <td style='padding:8px;border:1px solid #ddd;'>
                        {action.get('action', 'N/A')}
                    </td>
                    <td style='padding:8px;border:1px solid #ddd;color:green;'>
                        ✅ {action.get('status', 'executed')}
                    </td>
                </tr>"""

        steps_html = "".join(
            f"<li style='margin:6px 0;'>{step}</li>"
            for step in ticket.get("investigation_steps", [])
        )

        severity_colors = {
            "CRITICAL": "#dc3545",
            "HIGH":     "#fd7e14",
            "MEDIUM":   "#ffc107",
            "LOW":      "#28a745"
        }
        severity_color = severity_colors.get(ticket["severity"], "#6c757d")

        html = f"""
        <html><body style='font-family:Arial,sans-serif;max-width:800px;margin:0 auto;'>
            <div style='background:{severity_color};color:white;padding:20px;border-radius:8px 8px 0 0;'>
                <h1 style='margin:0;'>🚨 SecOpsAI Incident Alert</h1>
                <h2 style='margin:5px 0;'>Ticket: {ticket['ticket_id']}</h2>
            </div>

            <div style='background:#f8f9fa;padding:20px;border:1px solid #ddd;'>

                <table style='width:100%;border-collapse:collapse;margin-bottom:20px;'>
                    <tr>
                        <td style='padding:10px;background:#fff;border:1px solid #ddd;width:50%;'>
                            <strong>Severity:</strong>
                            <span style='color:{severity_color};font-weight:bold;'>
                                {ticket['severity']}
                            </span>
                        </td>
                        <td style='padding:10px;background:#fff;border:1px solid #ddd;'>
                            <strong>SLA:</strong> {ticket['sla_deadline']}
                        </td>
                    </tr>
                    <tr>
                        <td style='padding:10px;background:#fff;border:1px solid #ddd;'>
                            <strong>Attack Type:</strong> {ticket['attack_type']}
                        </td>
                        <td style='padding:10px;background:#fff;border:1px solid #ddd;'>
                            <strong>Category:</strong> {ticket['category']}
                        </td>
                    </tr>
                    <tr>
                        <td style='padding:10px;background:#fff;border:1px solid #ddd;'>
                            <strong>Source IP:</strong> {ticket['source_ip']}
                        </td>
                        <td style='padding:10px;background:#fff;border:1px solid #ddd;'>
                            <strong>Confidence:</strong> {ticket['confidence']:.2%}
                        </td>
                    </tr>
                    <tr>
                        <td style='padding:10px;background:#fff;border:1px solid #ddd;'>
                            <strong>MITRE ATT&CK:</strong> {ticket['mitre_techniques']}
                        </td>
                        <td style='padding:10px;background:#fff;border:1px solid #ddd;'>
                            <strong>Assigned To:</strong> {ticket['assigned_to']}
                        </td>
                    </tr>
                    <tr>
                        <td colspan='2' style='padding:10px;background:#fff;border:1px solid #ddd;'>
                            <strong>Detected At:</strong> {ticket['created_at']}
                        </td>
                    </tr>
                </table>

                <h3 style='color:#333;'>🔍 Threat Intelligence</h3>
                <table style='width:100%;border-collapse:collapse;margin-bottom:20px;'>
                    <tr style='background:#e9ecef;'>
                        <th style='padding:8px;border:1px solid #ddd;text-align:left;'>Source</th>
                        <th style='padding:8px;border:1px solid #ddd;text-align:left;'>Malicious Votes</th>
                        <th style='padding:8px;border:1px solid #ddd;text-align:left;'>Reputation</th>
                        <th style='padding:8px;border:1px solid #ddd;text-align:left;'>Country</th>
                    </tr>
                    <tr>
                        <td style='padding:8px;border:1px solid #ddd;'>
                            {enrichment.get('source', 'N/A')}
                        </td>
                        <td style='padding:8px;border:1px solid #ddd;'>
                            {enrichment.get('malicious_votes', 'N/A')}
                        </td>
                        <td style='padding:8px;border:1px solid #ddd;'>
                            {enrichment.get('reputation_score', 'N/A')}
                        </td>
                        <td style='padding:8px;border:1px solid #ddd;'>
                            {enrichment.get('country', 'N/A')}
                        </td>
                    </tr>
                </table>

                <h3 style='color:#333;'>🛡️ Automated Containment Actions</h3>
                <table style='width:100%;border-collapse:collapse;margin-bottom:20px;'>
                    <tr style='background:#e9ecef;'>
                        <th style='padding:8px;border:1px solid #ddd;text-align:left;'>Method</th>
                        <th style='padding:8px;border:1px solid #ddd;text-align:left;'>Action</th>
                        <th style='padding:8px;border:1px solid #ddd;text-align:left;'>Status</th>
                    </tr>
                    {containment_html}
                </table>

                <h3 style='color:#333;'>📋 Investigation Steps</h3>
                <ol style='background:#fff;padding:20px;border:1px solid #ddd;border-radius:4px;'>
                    {steps_html}
                </ol>

                <div style='background:#fff3cd;padding:15px;border:1px solid #ffc107;
                            border-radius:4px;margin-top:20px;'>
                    <strong>⚠️ Action Required:</strong>
                    Please investigate this incident and update the ticket status
                    within the SLA window: <strong>{ticket['sla_deadline']}</strong>
                </div>

            </div>

            <div style='background:#343a40;color:#adb5bd;padding:15px;
                        border-radius:0 0 8px 8px;font-size:12px;'>
                SecOpsAI Automated Detection System |
                Ticket {ticket['ticket_id']} |
                Do not reply to this email
            </div>
        </body></html>
        """

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_ANALYST, msg.as_string())

        logger.info(f"[EMAIL] Incident ticket {ticket['ticket_id']} sent to {EMAIL_ANALYST}")
        return True

    except Exception as e:
        logger.error(f"[EMAIL] Failed to send ticket: {e}")
        return False


def send_slack_alert(alert_payload: dict, ticket: dict = None) -> bool:
    if not SLACK_WEBHOOK_URL:
        logger.info("[ALERT] Slack not configured — logging locally")
        logger.info(f"[ALERT] {json.dumps(alert_payload, indent=2)}")
        return False

    try:
        enrichment   = alert_payload.get("enrichment", {})
        containments = alert_payload.get("containment_actions", [])

        containment_text = ""
        for action in containments:
            if action.get("action") != "none":
                detail = (
                    action.get("rule") or
                    action.get("target_user") or
                    action.get("behavioral_signature", "")[:40]
                )
                containment_text += f"\n• *{action.get('method')}:* `{detail}`"

        if not containment_text:
            containment_text = "\n• No automated action (below threshold)"

        ticket_text = ""
        if ticket:
            ticket_text = (
                f"\n*Incident Ticket:* `{ticket['ticket_id']}` | "
                f"Severity: *{ticket['severity']}* | "
                f"Assigned to: {ticket['assigned_to']}"
            )

        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "🚨 SecOpsAI Detection Alert"}
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Attack:*\n{alert_payload['attack_type']}"},
                        {"type": "mrkdwn", "text": f"*Confidence:*\n{alert_payload['confidence']:.2%}"},
                        {"type": "mrkdwn", "text": f"*Source IP:*\n{alert_payload.get('source_ip')}"},
                        {"type": "mrkdwn", "text": f"*Time:*\n{alert_payload['timestamp']}"}
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Threat Intel:* "
                            f"Malicious: {enrichment.get('malicious_votes', 'N/A')} | "
                            f"Reputation: {enrichment.get('reputation_score', 'N/A')} | "
                            f"Country: {enrichment.get('country', 'N/A')}"
                            f"{ticket_text}"
                        )
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Containment Actions (4):*{containment_text}\n• *ticket_creation:* `{ticket['ticket_id'] if ticket else 'N/A'}` ✅"
                    }
                },
                {"type": "divider"}
            ]
        }

        response = requests.post(SLACK_WEBHOOK_URL, json=message, timeout=5)
        if response.status_code == 200:
            logger.info("[ALERT] Slack notification sent")
        return response.status_code == 200

    except Exception as e:
        logger.error(f"Slack alert failed: {e}")
        return False


def containment_firewall_block(source_ip: str, confidence: float) -> dict:
    if confidence < 0.90 or not source_ip:
        return {"action": "none", "method": "firewall_block",
                "reason": f"Confidence {confidence:.2%} below threshold"}
    action = {
        "action": "firewall_block", "method": "firewall_block",
        "target_ip": source_ip, "rule": f"DROP src_ip={source_ip}",
        "timestamp": datetime.utcnow().isoformat(), "status": "mock_executed",
        "note": "Production: iptables -A INPUT -s {ip} -j DROP"
    }
    logger.info(f"[CONTAINMENT-1] Firewall block: {action['rule']}")
    return action


def containment_disable_user(source_ip: str, confidence: float) -> dict:
    if confidence < 0.85:
        return {"action": "none", "method": "user_disable",
                "reason": f"Confidence {confidence:.2%} below threshold"}
    ip_suffix   = source_ip.split(".")[-1] if source_ip else "unknown"
    target_user = f"user_{ip_suffix}@secopsai.local"
    action = {
        "action": "user_account_disabled", "method": "user_disable",
        "target_user": target_user, "source_ip": source_ip,
        "timestamp": datetime.utcnow().isoformat(), "status": "mock_executed",
        "note": f"Production: Disable {target_user} via AD/Okta/Azure API"
    }
    logger.info(f"[CONTAINMENT-2] User disabled: {target_user}")
    return action


def containment_ngfw_behavioral_profile(
    features: dict, confidence: float, attack_type: str
) -> dict:
    if confidence < 0.80:
        return {"action": "none", "method": "ngfw_behavioral_profile",
                "reason": f"Confidence {confidence:.2%} below threshold"}
    behavioral_features = {
        "flow_bytes_per_sec":    features.get("Flow Bytes/s", 0),
        "packet_timing_entropy": features.get("timing_entropy", 0),
        "connection_symmetry":   features.get("connection_symmetry", 0),
        "byte_ratio":            features.get("byte_ratio", 0),
        "syn_flag_pattern":      features.get("SYN Flag Count", 0),
    }
    signature_hash = hashlib.sha256(
        json.dumps(behavioral_features, sort_keys=True).encode()
    ).hexdigest()[:16]
    action = {
        "action": "ngfw_behavioral_policy_applied",
        "method": "ngfw_behavioral_profile",
        "rule_id": f"SECOPSAI-{signature_hash.upper()}",
        "behavioral_signature": signature_hash,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "mock_executed",
        "note": "Production: POST to Palo Alto PAN-OS / Fortinet FortiGate API"
    }
    logger.info(f"[CONTAINMENT-3] NGFW profile: Rule SECOPSAI-{signature_hash.upper()}")
    return action


def process_alert(
    detection_result: dict,
    source_ip: str = None,
    confidence: float = 0.0,
    raw_features: dict = None
) -> dict:
    """
    Full alert pipeline:
    Enrich → 3x Containment → Create Ticket → Email Analyst → Slack
    """
    timestamp  = datetime.utcnow().isoformat()
    enrichment = enrich_with_virustotal(source_ip)

    firewall_action = containment_firewall_block(source_ip, confidence)
    user_action     = containment_disable_user(source_ip, confidence)
    ngfw_action     = containment_ngfw_behavioral_profile(
        raw_features or {}, confidence,
        detection_result.get("prediction", "UNKNOWN")
    )
    containment_actions = [firewall_action, user_action, ngfw_action]

    # Create incident ticket
    ticket = create_incident_ticket(
        attack_type=detection_result.get("prediction", "UNKNOWN"),
        source_ip=source_ip,
        confidence=confidence,
        enrichment=enrichment,
        containment_actions=containment_actions
    )

    # Send email to assigned analyst
    email_sent = send_analyst_email(ticket)

    # Send Slack notification
    alert_payload = {
        "timestamp":           timestamp,
        "attack_type":         detection_result.get("prediction", "UNKNOWN"),
        "confidence":          confidence,
        "source_ip":           source_ip or "unknown",
        "enrichment":          enrichment,
        "containment_actions": containment_actions
    }
    slack_sent = send_slack_alert(alert_payload, ticket)

    return {
        "alert":               alert_payload,
        "containment_actions": containment_actions,
        "ticket":              ticket,
        "email_sent":          email_sent,
        "slack_sent":          slack_sent
    }
