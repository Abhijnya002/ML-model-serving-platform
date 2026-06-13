"""gRPC inference server."""

import logging
import os
import sys
import time
from concurrent import futures

import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generated"))
import inference_pb2
import inference_pb2_grpc

from cache import InferenceCache
from model import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

GRPC_PORT = int(os.environ.get("GRPC_PORT", 50051))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 10))
DEFAULT_MODELS = os.environ.get("DEFAULT_MODELS", "iris_classifier").split(",")


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, registry: ModelRegistry, cache: InferenceCache):
        self._registry = registry
        self._cache = cache

    def Predict(self, request, context):
        start_ms = time.time() * 1000
        features = list(request.features)
        cache_key = ModelRegistry.feature_hash(request.model_name, features)

        cached = self._cache.get(cache_key)
        if cached:
            latency = int(time.time() * 1000 - start_ms)
            logger.info("Cache HIT for request %s (%.1fms)", request.request_id, latency)
            return inference_pb2.PredictResponse(
                request_id=request.request_id,
                predictions=cached["predictions"],
                predicted_class=cached["predicted_class"],
                confidence=cached["confidence"],
                cache_hit=True,
                latency_ms=latency,
            )

        try:
            result = self._registry.predict(request.model_name, features)
        except ValueError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return inference_pb2.PredictResponse()

        self._cache.set(cache_key, {
            "predictions": result.predictions,
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
        })

        latency = int(time.time() * 1000 - start_ms)
        logger.info(
            "Predicted '%s' (conf=%.2f) for request %s (%.1fms)",
            result.predicted_class, result.confidence, request.request_id, latency,
        )

        return inference_pb2.PredictResponse(
            request_id=request.request_id,
            predictions=result.predictions,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            cache_hit=False,
            latency_ms=latency,
        )

    def BatchPredict(self, request, context):
        responses = [self.Predict(r, context) for r in request.requests]
        return inference_pb2.BatchPredictResponse(responses=responses)

    def GetModelInfo(self, request, context):
        meta = self._registry.get_metadata(request.model_name)
        if meta is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model '{request.model_name}' not found")
            return inference_pb2.ModelInfoResponse()
        return inference_pb2.ModelInfoResponse(
            model_name=meta.name,
            version=meta.version,
            description=meta.description,
            input_features=meta.input_features,
            output_classes=meta.output_classes,
        )


def serve():
    registry = ModelRegistry()
    for model_name in DEFAULT_MODELS:
        model_name = model_name.strip()
        logger.info("Loading model: %s", model_name)
        registry.load_or_train(model_name)

    cache = InferenceCache()
    cache.connect()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(registry, cache), server
    )

    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    logger.info("gRPC server listening on port %d", GRPC_PORT)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
