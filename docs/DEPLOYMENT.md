# Deployment Guide

## Local Development (Docker Compose)

### Prerequisites
- Docker Desktop 4.x+ with at least 8GB RAM allocated
- Python 3.11+ (for running scripts locally)
- An LLM API key (OpenAI recommended to start)

### Steps

```bash
# 1. Clone and configure
git clone https://github.com/your-org/cortex
cd cortex
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY and SECRET_KEY

# 2. Start all services
docker compose up -d

# 3. Wait for services to be healthy
docker compose ps   # All should show "healthy"

# 4. Generate a dev token
export TOKEN=$(python scripts/gen_token.py)

# 5. Verify the API
curl http://localhost:8000/health
# {"status": "ok", "service": "cortex", "environment": "local"}

# 6. Open observability UIs
open http://localhost:6006   # Arize Phoenix (LLM traces)
open http://localhost:3000   # Grafana (admin/cortex)
```

### Useful commands

```bash
# View API logs
docker compose logs cortex-api -f

# View Celery worker logs
docker compose logs cortex-worker -f

# Restart a single service
docker compose restart cortex-api

# Stop everything
docker compose down

# Stop and wipe all data volumes
docker compose down -v
```

---

## Cloud Deployment (Kubernetes)

Cortex is designed for Kubernetes. The manifests in `deploy/k8s/` cover a production-ready deployment.

### Prerequisites

- Kubernetes cluster (AKS / EKS / GKE / any)
- `kubectl` configured for the cluster
- Container registry (ACR / ECR / Artifact Registry)
- Managed Redis, PostgreSQL, and a Qdrant instance

### Build and push the image

```bash
# Build
docker build -t cortex:$(git rev-parse --short HEAD) .

# Tag and push (Azure example)
az acr login --name youracr
docker tag cortex:$(git rev-parse --short HEAD) youracr.azurecr.io/cortex:latest
docker push youracr.azurecr.io/cortex:latest
```

### Deploy

```bash
# Create namespace
kubectl apply -f deploy/k8s/service.yaml   # also creates namespace and configmap

# Update secrets (NEVER commit actual values)
kubectl create secret generic cortex-secrets \
  --namespace cortex \
  --from-literal=SECRET_KEY="$(openssl rand -hex 32)" \
  --from-literal=OPENAI_API_KEY="sk-..." \
  --from-literal=COHERE_API_KEY="..." \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy workloads
kubectl apply -f deploy/k8s/deployment.yaml

# Verify
kubectl get pods -n cortex
kubectl get svc -n cortex
```

### Managed services for production

| Component | AWS | Azure | GCP |
|-----------|-----|-------|-----|
| Redis | ElastiCache (Redis 7) | Azure Cache for Redis | Memorystore |
| PostgreSQL | RDS PostgreSQL 16 | Azure Database for PostgreSQL | Cloud SQL |
| Qdrant | Self-hosted on EKS / Qdrant Cloud | Qdrant Cloud | Qdrant Cloud |
| Secrets | AWS Secrets Manager | Azure Key Vault | Secret Manager |
| Container registry | ECR | ACR | Artifact Registry |

Update `deploy/k8s/service.yaml` ConfigMap with the managed service endpoints.

### Zero-downtime deployments

The deployment uses `RollingUpdate` strategy with `maxUnavailable=0`. A Pod Disruption Budget ensures at least 1 replica is always available during node maintenance.

For schema migrations, run Alembic before the new deployment:
```bash
kubectl run cortex-migrate --image=cortex:latest --restart=Never \
  --env-from=configmap/cortex-config \
  --env-from=secret/cortex-secrets \
  -- alembic upgrade head
```

### Scaling

The HPA in `deploy/k8s/service.yaml` scales the API deployment based on CPU (70%) and memory (80%). Scale workers based on Celery queue depth — configure KEDA for queue-based autoscaling.

```bash
# Manual scale
kubectl scale deployment cortex-api --replicas=5 -n cortex
kubectl scale deployment cortex-worker --replicas=8 -n cortex
```

---

## Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | ✅ | — | JWT signing key (≥32 chars) |
| `OPENAI_API_KEY` | One provider required | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | | — | Anthropic API key |
| `COHERE_API_KEY` | | — | Cohere API key (reranking) |
| `DATABASE_URL` | ✅ | local default | PostgreSQL connection string |
| `REDIS_URL` | ✅ | local default | Redis connection string |
| `QDRANT_URL` | ✅ | local default | Qdrant HTTP endpoint |
| `ENVIRONMENT` | | `local` | `local` / `staging` / `production` |
| `MAX_COST_PER_RUN_USD` | | `2.00` | Hard spend limit per run |
| `GUARDRAILS_ENABLED` | | `true` | Enable NeMo Guardrails |
