FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ proto/
COPY server/ server/

RUN mkdir -p generated && \
    python -m grpc_tools.protoc \
      --proto_path=proto \
      --python_out=generated \
      --grpc_python_out=generated \
      proto/inference.proto && \
    sed -i 's/^import inference_pb2/from . import inference_pb2/' \
      generated/inference_pb2_grpc.py

ENV PYTHONPATH=/app/generated
ENV MODEL_DIR=/app/models
ENV GRPC_PORT=50051

EXPOSE 50051

CMD ["python", "server/server.py"]
