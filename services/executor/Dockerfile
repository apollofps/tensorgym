# TensorGym sandbox executor image (pinned).
#   PyTorch 2.12.0 (CPU)  +  Python 3.12
# Network is required only at BUILD time to fetch the CPU wheel. At RUN time the
# container is started with --network none (see docker_executor.py).
FROM python:3.12-slim

# CPU-only PyTorch from the official CPU wheel index (resolves to 2.12.0+cpu).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.12.0"

# NumPy so torch's array-interop is initialised (avoids a startup UserWarning that
# would otherwise leak into learner-visible stderr). Separate layer keeps the
# large torch layer cached.
RUN pip install --no-cache-dir "numpy"

# Unprivileged user; learner code runs as this user, never root.
RUN useradd --create-home --uid 10001 runner

COPY runner.py /runner.py

USER runner
WORKDIR /work

# The host appends the read-only job path, e.g. /work/job.json
ENTRYPOINT ["python", "/runner.py"]
