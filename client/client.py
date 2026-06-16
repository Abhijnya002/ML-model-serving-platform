"""gRPC inference client with throughput benchmarking."""

import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generated"))
import inference_pb2
import inference_pb2_grpc

SERVER_ADDR = os.environ.get("SERVER_ADDR", "localhost:50051")

IRIS_SAMPLES = [
    [5.1, 3.5, 1.4, 0.2],  # setosa
    [6.7, 3.1, 4.7, 1.5],  # versicolor
    [6.3, 3.3, 6.0, 2.5],  # virginica
    [4.9, 3.0, 1.4, 0.2],  # setosa
    [5.8, 2.7, 5.1, 1.9],  # virginica
]


def make_stub(addr: str) -> inference_pb2_grpc.InferenceServiceStub:
    channel = grpc.insecure_channel(addr)
    return inference_pb2_grpc.InferenceServiceStub(channel)


def demo_single_predict(stub):
    print("\n--- Single Prediction ---")
    features = IRIS_SAMPLES[0]
    req = inference_pb2.PredictRequest(
        model_name="iris_classifier",
        features=features,
        request_id=str(uuid.uuid4()),
    )
    resp = stub.Predict(req)
    print(f"Class: {resp.predicted_class}  Confidence: {resp.confidence:.3f}")
    print(f"Probabilities: {[f'{p:.3f}' for p in resp.predictions]}")
    print(f"Cache hit: {resp.cache_hit}  Latency: {resp.latency_ms}ms")

    print("\n  (Repeat same request — should be cache hit)")
    resp2 = stub.Predict(req)
    print(f"Cache hit: {resp2.cache_hit}  Latency: {resp2.latency_ms}ms")


def demo_batch_predict(stub):
    print("\n--- Batch Prediction ---")
    requests = [
        inference_pb2.PredictRequest(
            model_name="iris_classifier",
            features=sample,
            request_id=str(uuid.uuid4()),
        )
        for sample in IRIS_SAMPLES
    ]
    batch_req = inference_pb2.BatchPredictRequest(
        model_name="iris_classifier",
        requests=requests,
    )
    batch_resp = stub.BatchPredict(batch_req)
    for r in batch_resp.responses:
        print(f"  [{r.request_id[:8]}] {r.predicted_class} (conf={r.confidence:.2f}, cache={r.cache_hit})")


def demo_model_info(stub):
    print("\n--- Model Info ---")
    resp = stub.GetModelInfo(
        inference_pb2.ModelInfoRequest(model_name="iris_classifier")
    )
    print(f"Name: {resp.model_name}  Version: {resp.version}")
    print(f"Input features: {resp.input_features}")
    print(f"Classes: {list(resp.output_classes)}")
    print(f"Description: {resp.description}")


def benchmark(stub, n_requests: int = 1000, concurrency: int = 20):
    print(f"\n--- Throughput Benchmark ({n_requests} requests, {concurrency} workers) ---")

    def single_request(_):
        features = IRIS_SAMPLES[_ % len(IRIS_SAMPLES)]
        req = inference_pb2.PredictRequest(
            model_name="iris_classifier",
            features=features,
            request_id=str(uuid.uuid4()),
        )
        t0 = time.perf_counter()
        stub.Predict(req)
        return (time.perf_counter() - t0) * 1000

    latencies = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(single_request, i) for i in range(n_requests)]
        for f in as_completed(futures):
            latencies.append(f.result())
    elapsed = time.perf_counter() - start

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p99 = latencies[int(len(latencies) * 0.99)]
    rps = n_requests / elapsed

    print(f"Throughput: {rps:,.0f} req/s")
    print(f"p50 latency: {p50:.1f}ms")
    print(f"p99 latency: {p99:.1f}ms")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    stub = make_stub(SERVER_ADDR)

    demo_model_info(stub)
    demo_single_predict(stub)
    demo_batch_predict(stub)
    benchmark(stub)
