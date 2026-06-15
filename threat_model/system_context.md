# System Context Diagram — SecOpsAI

## Purpose
Defines the boundaries of the SecOpsAI detection system and all
external entities that interact with it.

## System Boundary
Everything inside the boundary is owned and controlled by SecOpsAI.
Everything outside is an external actor or data source.

## Internal Components
1. **Ingestion Pipeline** — Receives raw PCAP and endpoint log data
2. **Feature Engineering** — Converts raw data into ML-ready features
3. **ML Detector** — XGBoost/PyTorch model that classifies traffic
4. **FastAPI Service** — REST API exposing the detection engine
5. **Alert Pipeline** — Enriches and routes alerts
6. **PostgreSQL** — Stores detections, logs, and audit trails
7. **Redis** — Message queue between components
8. **Grafana Dashboard** — Visualizes system health and detections

## External Actors
| Actor | Role | Trust Level |
|---|---|---|
| Network Sensors | Send raw traffic data (PCAP) | Low — data must be validated |
| Security Analyst | Views dashboards, reviews alerts | Medium — authenticated user |
| VirusTotal API | Enriches alerts with threat intel | Medium — external service |
| Shodan API | Enriches IP reputation data | Medium — external service |
| Slack | Receives alert notifications | Low — outbound only |
| Attacker | Sends malicious traffic | Zero trust |

## Trust Boundaries
- **Boundary 1:** Between network sensors and ingestion pipeline
- **Boundary 2:** Between FastAPI and external consumers
- **Boundary 3:** Between alert pipeline and external APIs
