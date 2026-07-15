#!/bin/bash
# SecOpsAI Live Demo Script
# Demonstrates full pipeline: inject traffic → detect → enrich → respond

echo "=================================================="
echo "SecOpsAI Live Demo — End-to-End Detection Pipeline"
echo "=================================================="

echo ""
echo "Step 1 — Verifying system health..."
curl -s http://localhost:8000/health | python3 -m json.tool
sleep 2

echo ""
echo "Step 2 — Authenticating analyst..."
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -d "username=analyst&password=secopsai123" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
echo "Token acquired: ${TOKEN:0:20}..."
sleep 2

echo ""
echo "Step 3 — Injecting BENIGN traffic..."
curl -s -X POST http://localhost:8000/detect \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "byte_ratio": 0.5,
    "timing_entropy": 0.8,
    "size_consistency": 0.6,
    "connection_symmetry": 0.5,
    "flow_speed": 50000.0,
    "Flow Duration": 100000,
    "Total Fwd Packets": 20,
    "Total Backward Packets": 18,
    "Flow Bytes/s": 200000,
    "Flow Packets/s": 500,
    "Flow IAT Mean": 5000,
    "Flow IAT Std": 2000,
    "Fwd IAT Mean": 5000,
    "Bwd IAT Mean": 5000,
    "Packet Length Mean": 400,
    "Packet Length Std": 200,
    "Packet Length Variance": 5000,
    "Average Packet Size": 400,
    "SYN Flag Count": 1,
    "ACK Flag Count": 5,
    "PSH Flag Count": 2,
    "FIN Flag Count": 1,
    "source_ip": "10.0.0.50"
  }' | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'  Prediction: {d[\"prediction\"]}')
print(f'  Confidence: {d[\"confidence\"]:.2%}')
print(f'  Latency:    {d[\"latency_ms\"]}ms')
"
sleep 3

echo ""
echo "Step 4 — Injecting C2 BEACONING attack traffic..."
curl -s -X POST http://localhost:8000/detect \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @~/secopsai/tests/test_payload.json | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'  Prediction: {d[\"prediction\"]}')
print(f'  Confidence: {d[\"confidence\"]:.2%}')
print(f'  Latency:    {d[\"latency_ms\"]}ms')
if 'alert' in d:
    ticket = d['alert'].get('ticket', {})
    actions = d['alert'].get('containment_actions', [])
    print(f'  Ticket ID:  {ticket.get(\"ticket_id\", \"N/A\")}')
    print(f'  Severity:   {ticket.get(\"severity\", \"N/A\")}')
    print(f'  Containment actions executed:')
    for action in actions:
        if action.get(\"action\") != \"none\":
            print(f'    - {action[\"method\"]}: {action[\"action\"]}')
"
sleep 3

echo ""
echo "Step 5 — Checking Prometheus metrics..."
curl -s http://localhost:8000/metrics | grep -E "secopsai_detections|secopsai_alerts|secopsai_containment"

echo ""
echo "=================================================="
echo "Demo Complete"
echo "Open http://localhost:3000 to see Grafana dashboard"
echo "Open http://localhost:9090 to see Prometheus metrics"
echo "Check Slack for alert notifications"
echo "Check email for incident ticket"
echo "=================================================="
