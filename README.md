# Sentinel AI — Autonomous Traffic Intelligence Agent

[![CI](https://github.com/OWNER/sentinel-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/sentinel-ai/actions/workflows/ci.yml)

Real-time, explainable, safety-constrained control of a road intersection. Sentinel AI
observes traffic, reasons about the whole intersection, and autonomously drives the signals —
while a deterministic safety envelope guarantees it can never issue an unsafe command, and an
LLM narrates *why* every decision was made, entirely outside the control loop.

> **Architecture in one sentence:** a fast **Perception plane** (detection → tracking →
> movement → density) publishes a compact `IntersectionState` onto an event bus; eight
> event-driven **Cognition-plane agents** (memory, prediction, decision, signal control,
> incident detection, explainability, orchestration, dashboard) reason over that state and
> close the control loop against either a dependency-free analytical traffic twin or a SUMO
> microsimulation — with the same agent code either way.

## Status: all 11 agents implemented, closed loop running end-to-end

| # | Agent | Module | Status |
|---|---|---|---|
| 1 | Vision | 3 | ✅ |
| 2 | Tracking | 3 | ✅ |
| 3 | Movement Analysis | 3 | ✅ |
| 4 | Traffic Memory | 5 | ✅ |
| 5 | Prediction | 5 | ✅ |
| 6 | Decision | 4 | ✅ |
| 7 | Signal Controller | 4 | ✅ |
| 8 | Incident Detection | 6 | ✅ |
| 9 | Explainability | 6 | ✅ |
| 10 | Dashboard | 7 | ✅ |
| 11 | Orchestrator | 7 | ✅ |

Plus: the M2 SUMO bridge + fixed-timer baseline benchmark, a FastAPI read-API + WebSocket
gateway, a live web dashboard, Docker Compose for local full-stack, and this CI pipeline.

## Deployment topology

| Plane | Runs on |
| --- | --- |
| Dashboard (static) + thin read-API proxy | **Vercel** (see [`frontend/vercel.json`](frontend/vercel.json)) |
| Agent fleet + perception (GPU) + SUMO + Redis/RabbitMQ/Postgres | **GPU host** (RunPod / Fly GPU / Lambda), via Docker |

## Repository layout

```
sentinel-ai/
├─ libs/sentinel/
│  ├─ contracts/        # event schemas + IntersectionState/LaneState value objects
│  ├─ messaging/        # EventBus over RabbitMQ + Redis Streams (idempotency, DLQ) + in-memory
│  ├─ config/           # pydantic-settings configuration (SENTINEL_* env vars)
│  ├─ observability/    # structured logging, Prometheus metrics, tracing
│  ├─ control/          # the safety-critical phase state machine + safety envelope
│  ├─ simulation/       # analytical traffic twin, SUMO bridge, controllers, benchmark harness
│  ├─ perception/       # detector/tracker/movement/density ports + pipeline + worker
│  ├─ agents/           # shared BaseAgent (lifecycle, subscriptions, heartbeat)
│  ├─ decision/         # multi-objective utility policy + Decision Agent
│  ├─ signal_control/   # actuator port + Signal Controller Agent (the safe actuator authority)
│  ├─ memory/           # history repository + Traffic Memory Agent
│  ├─ prediction/       # forecasters (persistence, linear trend) + Prediction Agent
│  ├─ incident/         # incident rules + Incident Detection Agent
│  ├─ explainability/   # template + LLM explanation generators + Explainability Agent
│  ├─ orchestrator/     # fleet health + operating-mode state machine
│  ├─ dashboard/        # CQRS read-model + Dashboard Agent
│  └─ fleet.py          # composition root: wires every agent onto one bus
├─ services/api_gateway/ # FastAPI read-API + WebSocket over a running fleet
├─ frontend/             # static live dashboard (HTML/JS/Tailwind CDN) + vercel.json
├─ sim/                  # SUMO network definition + scenario builder
├─ scripts/              # terminal demos (perception, control loop, agent loop, benchmark)
├─ deploy/               # Dockerfile, docker-compose.yml, Prometheus/Grafana provisioning
├─ .github/workflows/    # CI: ruff + mypy + pytest + Docker build
└─ tests/                # unit + integration tests (200+, see Quality below)
```

## Quick start (local, no external services)

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# bash:                source .venv/bin/activate
pip install -e ".[dev,api]"
pytest                                    # full suite, zero external dependencies
python scripts/demo_agents_loop.py        # watch the 5-agent control loop reason live
python -m sentinel.simulation.run_baseline # fixed-timer vs. Sentinel-AI benchmark
```

Then run the full app (agent fleet + read-API + live dashboard):

```bash
uvicorn services.api_gateway.main:app --reload
# open http://localhost:8000            the dashboard
# open http://localhost:8000/metrics/   Prometheus-format metrics (trailing slash required)
```

## Quality

- **Tests**: 200+ unit/integration tests, in-memory event bus, zero external services required.
- **Type checking**: `mypy --strict` clean across the whole `libs/` tree.
- **Lint**: `ruff` clean (`pyflakes`, `isort`, `pyupgrade`, `bugbear`, `simplify`, ...).
- Run everything CI runs: `ruff check libs tests && mypy libs && pytest --cov=sentinel`.

## Deployment

### Local full stack (Docker Compose)

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Brings up the API gateway (agent fleet + dashboard, in-memory bus by default) plus RabbitMQ,
Redis, Postgres, Prometheus and Grafana (`admin`/`admin`). Open `http://localhost:8000` for the
dashboard, `http://localhost:9090` for Prometheus, `http://localhost:3000` for Grafana (a
pre-provisioned "Sentinel AI - Overview" dashboard is loaded automatically, tracking safety
violations, decision latency, perception FPS and agent health).

> **Scope note:** the API gateway runs a single-process, single-intersection fleet on the
> in-memory event bus (`sentinel.fleet.build_fleet`) — the right choice for the analytical-twin
> demo. RabbitMQ/Redis/Postgres are provisioned because `sentinel.messaging` and
> `sentinel.config` already target them; pointing `build_fleet` at `RabbitMqEventBus` /
> `RedisStreamBus` for a distributed, multi-service deployment is a configuration change, not a
> new architecture.

### Hybrid cloud deployment (Vercel + GPU host)

1. **Backend** — build and push `deploy/Dockerfile.api_gateway` to your GPU host (RunPod, Fly
   GPU, Lambda, ...) and run it (or `docker compose -f deploy/docker-compose.yml up` there
   directly). Confirm `https://<your-host>/api/v1/system/health` responds.
2. **Frontend** — in Vercel, import this repo and set the project's **Root Directory** to
   `frontend/` (zero build step, static site). Edit [`frontend/vercel.json`](frontend/vercel.json)
   and replace `YOUR_BACKEND_HOST` with your GPU host's public hostname — Vercel then proxies
   `/api/*` and `/metrics` to the backend.
3. Deploy. The dashboard's REST calls go through the Vercel rewrite; if you enable the
   WebSocket live-stream in the frontend, point it **directly** at the backend host rather than
   through the Vercel rewrite — persistent WebSocket proxying across providers is not something
   to rely on.

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push/PR: `ruff check`,
`mypy`, `pytest --cov`, and a Docker build of the API gateway image, on Python 3.11 and 3.12.
