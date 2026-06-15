# Architecture Decision Record — SecOpsAI

## ADR-001: XGBoost as Primary Classifier
- **Decision:** Use XGBoost as the primary ML classifier
- **Reason:** Handles tabular network features well, fast inference,
  interpretable via feature importance scores
- **Alternatives Considered:** Random Forest (slower), Neural Network
  (less interpretable, needs more data)
- **Trade-off:** Less accurate on raw packet sequences than deep learning

## ADR-002: FastAPI over Flask/Django
- **Decision:** Use FastAPI for the detection REST API
- **Reason:** Async support for high throughput, automatic OpenAPI docs,
  Pydantic input validation built in
- **Alternatives Considered:** Flask (no async), Django (too heavy)
- **Trade-off:** Smaller ecosystem than Django

## ADR-003: Redis over Kafka for Queuing
- **Decision:** Use Redis Streams as the message queue
- **Reason:** Simpler setup, sufficient throughput for this scale,
  already used for caching
- **Alternatives Considered:** Kafka (more powerful but complex to operate)
- **Trade-off:** Less durable than Kafka for very high volume

## ADR-004: PostgreSQL for Storage
- **Decision:** Use PostgreSQL for all persistent storage
- **Reason:** ACID compliance for audit logs, JSON support for
  flexible alert payloads, mature and reliable
- **Alternatives Considered:** MongoDB (less ACID), SQLite (not production grade)
- **Trade-off:** Requires more setup than SQLite

## ADR-005: IBM ART for Adversarial Testing
- **Decision:** Use IBM Adversarial Robustness Toolbox
- **Reason:** Industry standard, supports XGBoost and PyTorch,
  implements all required attack types
- **Alternatives Considered:** Foolbox (less XGBoost support)
- **Trade-off:** Steep learning curve

## Intentionally Left Out
- **Real-time PCAP capture:** Using pre-recorded datasets (CICIDS) instead
  to keep scope manageable
- **Kubernetes:** Docker Compose is sufficient for this scale
- **Custom SIEM:** Using Grafana/Kibana instead of building from scratch
