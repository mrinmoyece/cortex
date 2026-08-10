"""Metrics and tracing setup.

The interesting property here is not that tracing works — it is that
tracing *failing* must never take the service with it. Observability is
diagnostic; a crash in the diagnostics is strictly worse than no
diagnostics.
"""

from __future__ import annotations

from unittest.mock import patch

from prometheus_client import REGISTRY

from cortex.obs import metrics as m


class TestMetricCatalogue:
    def test_at_least_twenty_custom_metrics_are_registered(self):
        names = {n for n in REGISTRY._names_to_collectors if n.startswith("cortex_")}
        assert len(names) >= 20, f"only {len(names)} cortex_ metrics: {sorted(names)}"

    def test_every_metric_is_namespaced(self):
        """An unprefixed metric collides with whatever else scrapes into the
        same Prometheus, and you find out during an incident."""
        for attr in dir(m):
            obj = getattr(m, attr)
            name = getattr(obj, "_name", None)
            if name and attr.isupper() is False and hasattr(obj, "_type"):
                assert name.startswith("cortex_"), f"{attr} -> {name} is not namespaced"

    def test_label_names_are_bounded(self):
        """High-cardinality labels (run_id, user_id) are how a Prometheus
        instance falls over. Model and category are safe; identifiers are not."""
        banned = {"run_id", "user_id", "session_id", "tenant_id", "query"}
        for attr in dir(m):
            obj = getattr(m, attr)
            labels = getattr(obj, "_labelnames", None)
            if labels:
                offenders = banned & set(labels)
                assert not offenders, f"{attr} has unbounded labels {offenders}"


class TestTracingDegradesGracefully:
    def test_missing_otel_is_logged_not_raised(self):
        """`configure_tracing` is called at startup. If a missing optional
        dependency crashed it, the service would not boot because its
        *telemetry* was misconfigured."""
        with patch.dict("sys.modules", {"opentelemetry": None}):
            m.configure_tracing()  # must not raise

    def test_an_exporter_that_explodes_does_not_stop_startup(self):
        with patch(
            "opentelemetry.sdk.trace.TracerProvider", side_effect=RuntimeError("collector down")
        ):
            m.configure_tracing()  # must not raise

    def test_missing_phoenix_is_logged_not_raised(self):
        with patch.dict("sys.modules", {"phoenix.otel": None}):
            m.configure_phoenix()  # must not raise

    def test_configure_observability_survives_both_failing(self):
        with (
            patch.object(m, "configure_tracing", side_effect=None),
            patch.object(m, "configure_phoenix", side_effect=None),
        ):
            m.configure_observability()

    def test_observability_setup_is_idempotent(self):
        """Called twice — by the API and by a worker in the same process —
        it must not double-register or throw."""
        m.configure_observability()
        m.configure_observability()


class TestRunStatusMetricLabels:
    """`agent_runs_total.labels(status=state.status)` emitted the label
    "RunStatus.COMPLETED": `RunStatus` is a `(str, Enum)` and prometheus
    stringifies the member, not its value. The "started" and "error" labels
    emitted a few lines away were plain strings, so terminal and non-terminal
    runs did not share a label namespace, and every dashboard panel filtering
    on `status="completed"` matched nothing."""

    def test_an_enum_member_yields_its_value(self):
        from cortex.graph.cortex_graph import _status_label
        from cortex.graph.state import RunStatus

        assert _status_label(RunStatus.COMPLETED) == "completed"
        assert "RunStatus" not in _status_label(RunStatus.COMPLETED)

    def test_every_status_produces_a_lowercase_scalar(self):
        from cortex.graph.cortex_graph import _status_label
        from cortex.graph.state import RunStatus

        for status in RunStatus:
            label = _status_label(status)
            assert label == label.lower()
            assert "." not in label

    def test_a_plain_string_passes_through(self):
        """The literal "started" and "error" call sites must keep working."""
        from cortex.graph.cortex_graph import _status_label

        assert _status_label("started") == "started"
        assert _status_label("error") == "error"


class TestMetricCardinality:
    """A `Gauge` labelled by `run_id` minted a permanent time series per run.
    The classic cardinality mistake: invisible until Prometheus falls over
    weeks later, during an unrelated incident."""

    def test_no_metric_is_labelled_by_a_unique_identifier(self):
        import cortex.obs.metrics as metrics_module

        forbidden = {"run_id", "session_id", "user_id", "trace_id", "prompt"}
        offenders = []
        for name in dir(metrics_module):
            metric = getattr(metrics_module, name)
            labels = getattr(metric, "_labelnames", None)
            if labels and forbidden & set(labels):
                offenders.append((name, labels))
        assert not offenders, f"unbounded label cardinality: {offenders}"
