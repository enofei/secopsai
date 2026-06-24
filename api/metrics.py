"""
Prometheus Metrics — SecOpsAI
Tracks detection counts, latency, and model health.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response


# Total detections counter
detection_counter = Counter(
    'secopsai_detections_total',
    'Total number of detections',
    ['prediction', 'model_version']
)

# API latency histogram
latency_histogram = Histogram(
    'secopsai_inference_latency_ms',
    'Model inference latency in milliseconds',
    buckets=[10, 25, 50, 100, 150, 200, 300, 500]
)

# Alert counter
alert_counter = Counter(
    'secopsai_alerts_total',
    'Total alerts fired',
    ['attack_type', 'severity']
)

# False positive gauge (updated manually)
false_positive_gauge = Gauge(
    'secopsai_false_positive_rate',
    'Current estimated false positive rate'
)

# Model confidence histogram
confidence_histogram = Histogram(
    'secopsai_detection_confidence',
    'Distribution of model confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
)

# Containment actions counter
containment_counter = Counter(
    'secopsai_containment_actions_total',
    'Total containment actions executed',
    ['action_type']
)


def get_metrics():
    """Returns Prometheus metrics in text format."""
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
