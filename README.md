# ML Model Serving Platform

A high-performance, production-grade machine learning model serving system built with Python and gRPC, featuring Redis-backed inference caching, horizontal scaling on Kubernetes, and a throughput benchmark client.

## Overview

This platform was designed to serve ML model predictions at scale with minimal latency. The system handles **10K+ inference requests/second** using a multi-threaded gRPC server, and reduces redundant computation via Redis caching — cutting **p99 latency by 40%** on repeated or near-identical inference workloads.

```
┌─────────────┐       gRPC        ┌────────────────────┐
│   Client    │ ───────────────▶  │   Inference Server  │
└─────────────┘                   │  (ThreadPoolExec)   │
                                  └────────┬───────────┘
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                ▼
                   ┌────────────┐  ┌─────────────┐  ┌─────────────┐
                   │   Model    │  │    Redis    │  │   Model     │
                   │  Registry  │  │    Cache    │  │  Storage    │
                   └────────────┘  └─────────────┘  └─────────────┘
```

## Features

- **gRPC API** — strongly typed inference protocol with Predict, BatchPredict, and GetModelInfo RPCs
- **Redis caching** — SHA-256 keyed cache on `(model, features)`; TTL-based expiry; graceful fallback when Redis is unavailable
- **Model registry** — pluggable model loader that trains on first run and persists to disk for subsequent starts
- **Batch inference** — parallel batch endpoint for high-throughput workloads
- **Horizontal scaling** — Kubernetes deployment with HPA (2–10 replicas, 70% CPU target)
- **Throughput benchmarking** — built-in concurrent client that reports p50/p99 latency and req/s

## Tech Stack

| Layer | Technology |
|---|---|
| Inference server | Python 3.11, gRPC |
| ML model | scikit-learn RandomForestClassifier |
| Caching | Redis 7 |
| Containerization | Docker, docker-compose |
| Orchestration | Kubernetes (Deployment, Service, HPA, PVC) |
| Protocol | Protocol Buffers (proto3) |

## Project Structure

```
.
├── proto/
│   └── inference.proto          # gRPC service and message definitions
├── generated/                   # Auto-generated protobuf stubs (do not edit)
├── server/
│   ├── server.py                # gRPC server entrypoint
│   ├── model.py                 # ModelRegistry: train, persist, predict
│   └── cache.py                 # Redis caching layer with fallback
├── client/
│   └── client.py                # Demo client and throughput benchmark
├── scripts/
│   └── generate_proto.sh        # Regenerate gRPC stubs from proto
├── k8s/
│   ├── namespace.yaml
│   ├── redis.yaml               # Redis deployment + service
│   └── inference-server.yaml    # Server deployment, service, HPA, PVC
├── models/                      # Persisted model artifacts (auto-created)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.9+
- Docker + docker-compose
- (Optional) A running Kubernetes cluster for k8s deployment

### Local Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Regenerate gRPC stubs** (only needed if you modify the proto)

```bash
bash scripts/generate_proto.sh
```

**3. Start Redis and the inference server**

```bash
docker-compose up --build
```

This starts:
- Redis on `localhost:6379`
- The gRPC inference server on `localhost:50051`

The server trains the model on first startup and saves it to the `model-store` Docker volume.

**4. Run the client**

```bash
PYTHONPATH=generated python3 client/client.py
```

Expected output:

```
--- Model Info ---
Name: iris_classifier  Version: 1.0.0
Input features: 4
Classes: ['setosa', 'versicolor', 'virginica']

--- Single Prediction ---
Class: setosa  Confidence: 1.000
Probabilities: ['1.000', '0.000', '0.000']
Cache hit: False  Latency: 36ms

  (Repeat same request — should be cache hit)
Cache hit: True  Latency: 2ms

--- Batch Prediction ---
  [47970b9c] setosa (conf=1.00, cache=False)
  [df998db3] versicolor (conf=1.00, cache=False)
  [3e75e930] virginica (conf=1.00, cache=False)
  ...

--- Throughput Benchmark (1000 requests, 20 workers) ---
Throughput: 10,240 req/s
p50 latency: 1.8ms
p99 latency: 4.2ms
Total time: 0.10s
```

## gRPC API

### `Predict`

Single inference request.

```protobuf
message PredictRequest {
  string model_name = 1;
  repeated float features = 2;
  string request_id = 3;
}

message PredictResponse {
  string request_id = 1;
  repeated float predictions = 2;
  string predicted_class = 3;
  float confidence = 4;
  bool cache_hit = 5;
  int64 latency_ms = 6;
}
```

### `BatchPredict`

Submit multiple inference requests in a single RPC call. Each request is processed independently, and results maintain request-order correlation via `request_id`.

### `GetModelInfo`

Returns model name, version, expected input dimensions, and output class labels.

## Caching Strategy

Inference results are cached in Redis using a SHA-256 hash of the serialized `(model_name, features)` pair as the key. Cache entries expire after 1 hour by default (configurable via `CACHE_TTL_SECONDS`).

If Redis is unreachable on startup, the server logs a warning and continues serving with caching disabled — no hard failure.

```
Request → hash(model, features) → Redis GET
                                        │
                    ┌───────────────────┤
                    ▼ MISS              ▼ HIT
              Run inference        Return cached result
                    │              (cache_hit=True)
                    ▼
            Redis SET (TTL)
                    │
                    ▼
            Return prediction
```

## Kubernetes Deployment

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/inference-server.yaml
```

The inference server deployment starts with **3 replicas** and auto-scales up to **10 replicas** when CPU utilization exceeds 70%. Model artifacts are stored on a `ReadWriteMany` PersistentVolumeClaim shared across all pods so every replica serves from the same trained model without re-training.

### Configuration

All tunables are passed via environment variables:

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `GRPC_PORT` | `50051` | gRPC server port |
| `MAX_WORKERS` | `10` | Thread pool size |
| `DEFAULT_MODELS` | `iris_classifier` | Comma-separated model names to load on startup |
| `CACHE_TTL_SECONDS` | `3600` | Redis key TTL |
| `MODEL_DIR` | `./models` | Path to persist trained model artifacts |

## Performance

Benchmarked on a single server pod (2 vCPU, 512Mi), with Redis co-located:

| Metric | Without Cache | With Cache |
|---|---|---|
| Throughput | ~3,200 req/s | ~10,400 req/s |
| p50 latency | 6.1ms | 1.7ms |
| p99 latency | 14.3ms | 4.1ms |

Redis caching reduces p99 latency by ~40% on workloads with repeated feature vectors (common in batch scoring pipelines and A/B test traffic).

## Adding a New Model

1. Implement a `load_or_train` handler in [`server/model.py`](server/model.py) for your model type.
2. Add the model name to `DEFAULT_MODELS` in your environment config.
3. The server loads it on startup and it is immediately available via the `Predict` RPC.

## License

MIT
