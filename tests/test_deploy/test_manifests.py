"""The shipped deployment manifests, checked for internal coherence.

Nothing in CI parses these files, and a human reads their shape rather than
their indentation - which is how `docker-compose.yml` shipped with a
`depends_on` block that was not valid YAML, making the documented
five-minute quickstart impossible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    return yaml.safe_load((ROOT / name).read_text())


def _load_all(name: str) -> list[dict]:
    return [doc for doc in yaml.safe_load_all((ROOT / name).read_text()) if doc]


class TestComposeFileIsUsable:
    def test_it_parses(self):
        assert _load("docker-compose.yml")["services"]

    def test_the_documented_services_are_present(self):
        services = _load("docker-compose.yml")["services"]
        for name in ("cortex-api", "cortex-mcp", "cortex-worker", "redis", "qdrant"):
            assert name in services

    def test_every_dependency_names_a_real_service(self):
        """A `depends_on` pointing at a service that does not exist stops the
        whole stack, not just the one container."""
        compose = _load("docker-compose.yml")
        services = set(compose["services"])
        for name, spec in compose["services"].items():
            for dependency in spec.get("depends_on", {}) or {}:
                assert dependency in services, f"{name} depends on unknown service {dependency}"

    def test_the_mcp_service_uses_a_reachable_transport(self):
        """Compose publishes a port for it, and the stdio default would leave
        that port dead."""
        env = _load("docker-compose.yml")["services"]["cortex-mcp"]["environment"]
        assert env["MCP_TRANSPORT"] == "http"

    def test_no_secret_is_hardcoded(self):
        text = (ROOT / "docker-compose.yml").read_text()
        assert "sk-" not in text
        assert "${GRAFANA_ADMIN_PASSWORD" in text


class TestKubernetesMetricsIsCoherent:
    """Metrics were scraped by pod annotation - which sends no credential -
    while the shipped Secret set `METRICS_TOKEN`. Every scrape would have
    401'd, the dashboards would have been blank, and the first person to
    debug it would have deleted the token."""

    def test_the_api_pod_is_annotated_for_scraping(self):
        deployments = [d for d in _load_all("deploy/k8s/deployment.yaml") if d["kind"]]
        api = next(d for d in deployments if d["metadata"]["name"] == "cortex-api")
        annotations = api["spec"]["template"]["metadata"]["annotations"]
        assert annotations["prometheus.io/scrape"] == "true"
        assert annotations["prometheus.io/path"] == "/metrics"

    def test_the_shipped_secret_does_not_set_a_metrics_token(self):
        secret = next(d for d in _load_all("deploy/k8s/service.yaml") if d["kind"] == "Secret")
        assert "METRICS_TOKEN" not in secret["stringData"], (
            "annotation scraping sends no credential, so a token here breaks "
            "the scrape instead of securing it"
        )

    def test_the_ingress_refuses_metrics_from_outside(self):
        """The ingress rule is a `/` prefix, so without this the unauthenticated
        metrics endpoint is published to the internet along with the API."""
        ingress = next(d for d in _load_all("deploy/k8s/service.yaml") if d["kind"] == "Ingress")
        snippet = ingress["metadata"]["annotations"].get(
            "nginx.ingress.kubernetes.io/server-snippet", ""
        )
        assert "/metrics" in snippet
        assert "403" in snippet

    def test_securing_metrics_properly_is_documented(self):
        text = (ROOT / "deploy/k8s/service.yaml").read_text()
        assert "credentials_file" in text


class TestKubernetesWorkloadsAreConsistent:
    def test_every_workload_runs_as_non_root(self):
        for doc in _load_all("deploy/k8s/deployment.yaml"):
            if doc["kind"] != "Deployment":
                continue
            security = doc["spec"]["template"]["spec"]["securityContext"]
            assert security["runAsNonRoot"] is True, doc["metadata"]["name"]

    def test_every_container_has_a_read_only_root_filesystem(self):
        for doc in _load_all("deploy/k8s/deployment.yaml"):
            if doc["kind"] != "Deployment":
                continue
            for container in doc["spec"]["template"]["spec"]["containers"]:
                assert container["securityContext"]["readOnlyRootFilesystem"] is True, (
                    f"{doc['metadata']['name']}/{container['name']}"
                )

    def test_selectors_match_pod_labels(self):
        """A selector that matches nothing produces a Deployment that reports
        healthy and serves no traffic."""
        for doc in _load_all("deploy/k8s/deployment.yaml"):
            if doc["kind"] != "Deployment":
                continue
            selector = doc["spec"]["selector"]["matchLabels"]
            labels = doc["spec"]["template"]["metadata"]["labels"]
            assert selector.items() <= labels.items(), doc["metadata"]["name"]

    def test_no_credential_is_committed(self):
        text = (ROOT / "deploy/k8s/service.yaml").read_text()
        assert "sk-" not in text
        for line in text.splitlines():
            if ":" in line and "REPLACE_WITH" not in line:
                assert "password@" not in line


class TestPrometheusConfig:
    def test_every_scrape_target_is_a_service_we_ship(self):
        compose = set(_load("docker-compose.yml")["services"])
        config = _load("obs/prometheus.yml")
        for job in config["scrape_configs"]:
            for static in job.get("static_configs", []):
                for target in static["targets"]:
                    host = target.split(":")[0]
                    if host in {"localhost", "prometheus"}:
                        continue
                    assert host in compose, f"{job['job_name']} scrapes unknown host {host}"

    def test_the_alert_rules_file_is_mounted(self):
        """Prometheus loaded a rule file that compose never mounted, so every
        alert in it was inert - indistinguishable from an alert that simply
        never fires."""
        rule_files = _load("obs/prometheus.yml").get("rule_files", [])
        if not rule_files:
            pytest.skip("no rule files configured")
        volumes = _load("docker-compose.yml")["services"]["prometheus"]["volumes"]
        mounted = " ".join(volumes)
        for rule_file in rule_files:
            assert Path(rule_file).name in mounted
