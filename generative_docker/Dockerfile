ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm
FROM ${PYTHON_IMAGE}
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY generative_docker/requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir --only-binary=:all: \
    --extra-index-url https://download.pytorch.org/whl/cu126 -r /app/requirements.lock.txt
COPY baseline/ /app/baseline/
COPY research/ /app/research/
COPY generative_docker/ /app/generative_docker/
ENTRYPOINT ["python", "-u", "-m", "generative_docker.app"]
CMD ["--data-dir", "/data"]
