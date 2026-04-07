FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-venv \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY knod/requirements.txt .

RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

COPY knod/ knod/
COPY *.md .

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV KNOD_HTTP_PORT=8080
ENV KNOD_TCP_PORT=7999

EXPOSE 8080 7999

CMD ["python3", "-m", "knod", "serve"]
