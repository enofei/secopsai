# Data Flow Diagram — SecOpsAI

## Data Flow Steps

### Flow 1: Raw Traffic Ingestion
- SOURCE: Network sensors / PCAP files
- PROCESS: Ingestion pipeline parses and validates
- DESTINATION: Redis queue
- DATA TYPE: Raw packet captures, endpoint logs
- SECURITY CONTROLS: Schema validation, size limits, integrity hash

### Flow 2: Feature Engineering
- SOURCE: Redis queue
- PROCESS: Feature extractor converts packets to ML features
- DESTINATION: Feature store (PostgreSQL)
- DATA TYPE: Numerical feature vectors
- SECURITY CONTROLS: Input sanitization, range validation

### Flow 3: ML Detection
- SOURCE: Feature store
- PROCESS: XGBoost/PyTorch model classifies traffic
- DESTINATION: Detection results table (PostgreSQL)
- DATA TYPE: Classification labels + confidence scores
- SECURITY CONTROLS: Model versioning via MLflow, output validation

### Flow 4: Alert Enrichment
- SOURCE: Detection results (if malicious)
- PROCESS: Alert pipeline queries VirusTotal + Shodan
- DESTINATION: Enriched alert store
- DATA TYPE: IP reputation, threat intel, geolocation
- SECURITY CONTROLS: API key rotation, rate limiting, timeout handling

### Flow 5: Notification & Response
- SOURCE: Enriched alert
- PROCESS: Notify analyst + execute containment
- DESTINATION: Slack/Email + Firewall/Host isolation
- DATA TYPE: Alert payload, containment commands
- SECURITY CONTROLS: Signed payloads, action audit logging

## Data Sensitivity Classification
| Data Type | Sensitivity | Storage | Retention |
|---|---|---|---|
| Raw PCAP | High | Encrypted at rest | 30 days |
| Feature vectors | Medium | PostgreSQL | 90 days |
| Detection results | High | PostgreSQL | 1 year |
| Audit logs | Critical | Append-only | 2 years |
| API keys | Critical | .env file only | Rotated monthly |
