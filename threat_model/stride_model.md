# STRIDE Threat Model — SecOpsAI

## Component 1: Ingestion Pipeline

| STRIDE | Threat | Mitigation | MITRE ATT&CK |
|---|---|---|---|
| Spoofing | Attacker sends fake PCAP data pretending to be a trusted sensor | Mutual TLS authentication on all data sources | T1001 |
| Tampering | Attacker modifies training data to poison the ML model | SHA-256 integrity hash on every data file | T1565 |
| Repudiation | No record of who sent which data | Append-only audit log with timestamps | T1562 |
| Info Disclosure | Raw PCAP contains sensitive user data | Encrypt PCAP at rest using AES-256 | T1020 |
| Denial of Service | Attacker floods pipeline with junk data | Rate limiting + input size validation | T1498 |
| Elevation of Privilege | Pipeline runs as root and gets exploited | Run pipeline as non-root service account | T1068 |

## Component 2: ML Model

| STRIDE | Threat | Mitigation | MITRE ATT&CK |
|---|---|---|---|
| Spoofing | Attacker loads a fake model file | Model signing + hash verification on load | T1195 |
| Tampering | Adversarial examples crafted to fool model | IBM ART adversarial training | T1565 |
| Repudiation | No record of which model version made a decision | Log model version + prediction in PostgreSQL | T1562 |
| Info Disclosure | Model reveals training data via inference attacks | Differential privacy, output rounding | T1590 |
| Denial of Service | Attacker sends complex inputs to slow inference | Input validation + p99 latency monitoring | T1498 |
| Elevation of Privilege | Model output used to trigger privileged actions without validation | Validate all model outputs before action | T1068 |

## Component 3: FastAPI Service

| STRIDE | Threat | Mitigation | MITRE ATT&CK |
|---|---|---|---|
| Spoofing | Unauthenticated requests to detection endpoint | JWT authentication on all endpoints | T1078 |
| Tampering | Malicious input payload to corrupt detection | Input schema validation via Pydantic | T1565 |
| Repudiation | No record of who called the API | Structured audit logs for every request | T1562 |
| Info Disclosure | API leaks model details in error messages | Generic error responses, no stack traces | T1590 |
| Denial of Service | API flooded with requests | Rate limiting via SlowAPI | T1498 |
| Elevation of Privilege | API endpoint grants admin access without auth | Role-based access control (RBAC) | T1068 |

## Component 4: Alert Pipeline

| STRIDE | Threat | Mitigation | MITRE ATT&CK |
|---|---|---|---|
| Spoofing | Fake alerts injected into pipeline | Sign all alert payloads with HMAC | T1001 |
| Tampering | Alert enrichment data modified in transit | TLS on all external API calls | T1565 |
| Repudiation | No record of automated response actions | Log every containment action taken | T1562 |
| Info Disclosure | Alert payload contains sensitive network data | Scrub sensitive fields before Slack notification | T1020 |
| Denial of Service | VirusTotal/Shodan API limits exceeded | Exponential backoff + circuit breaker | T1498 |
| Elevation of Privilege | Response pipeline executes as root | Run as least-privilege service account | T1068 |

## AI-Specific Threats

| Threat Type | Description | Mitigation |
|---|---|---|
| Model Poisoning | Attacker injects malicious training samples | Data validation + tamper-evident logging |
| Adversarial Evasion | Crafted traffic bypasses ML detector | IBM ART adversarial training |
| Model Inversion | Attacker reconstructs training data from model | Output rounding + rate limiting |
| Membership Inference | Attacker determines if a sample was in training set | Differential privacy techniques |
