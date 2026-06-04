import logging
import threading

from prometheus_client import Counter, Gauge, Histogram

COMMIT_TO_FIRST_DELTA = Histogram(
    "voxtral_commit_to_first_delta_seconds",
    "Time from input_audio_buffer.commit to the first transcription.delta (perceived TTFW)",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
)
COMMIT_TO_DONE = Histogram(
    "voxtral_commit_to_done_seconds",
    "Time from input_audio_buffer.commit to transcription.done (total segment latency)",
    buckets=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
)
DELTA_INTERVAL = Histogram(
    "voxtral_delta_interval_seconds",
    "Time between consecutive transcription.delta events (streaming smoothness)",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)
DELTAS_PER_SEGMENT = Histogram(
    "voxtral_deltas_per_segment",
    "Number of transcription.delta events per segment (streaming granularity)",
    buckets=[1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
)
SEGMENT_AUDIO_DURATION = Histogram(
    "voxtral_segment_audio_duration_seconds",
    "Duration of the speech segment sent for transcription",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
)
SESSION_DURATION = Histogram(
    "voxtral_session_duration_seconds",
    "Lifetime of a single WebSocket connection (session.created to close/error)",
    buckets=[30, 60, 120, 300, 600, 1200, 3600],
)
# livesum: aggregate across live job subprocesses, ignore files from exited processes
ACTIVE_SESSIONS = Gauge(
    "voxtral_active_sessions",
    "Number of currently active transcription pipeline tasks",
    multiprocess_mode="livesum",
)
SEGMENTS_TOTAL = Counter(
    "voxtral_segments_total",
    "Total segments processed",
    ["outcome"],
)
RECONNECTS_TOTAL = Counter(
    "voxtral_reconnects_total",
    "Total WebSocket reconnect events",
)


def start_metrics_server(port: int) -> None:
    """Start a Prometheus HTTP server that aggregates metrics from all job subprocesses."""
    from prometheus_client import CollectorRegistry, make_wsgi_app
    from prometheus_client.multiprocess import MultiProcessCollector
    from wsgiref.simple_server import WSGIRequestHandler, make_server

    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    app = make_wsgi_app(registry)

    class _Silent(WSGIRequestHandler):
        def log_message(self, *args):
            pass

    def _serve():
        with make_server("", port, app, handler_class=_Silent) as httpd:
            httpd.serve_forever()

    threading.Thread(target=_serve, daemon=True, name="metrics-server").start()
    logging.info("Prometheus metrics available on :%d/metrics", port)
