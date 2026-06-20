FROM ubuntu:24.04 AS runtime-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    libgcc-s1 \
    libssl3 \
    libsqlite3-0 \
    libzstd1 \
    zlib1g \
    libbz2-1.0 \
    liblzma5 \
    libgomp1 \
    libatomic1 \
    redis-tools \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM runtime-base AS packaged-runtime

COPY build/linux-package/ /app/
COPY docker/entrypoint.sh /usr/local/bin/agent-entrypoint

RUN chmod +x /usr/local/bin/agent-entrypoint && \
    mkdir -p /app/data /app/logs /app/dumps /app/tmp

ENV APP_ROOT=/app

EXPOSE 8080 50051 50052

ENTRYPOINT ["/usr/local/bin/agent-entrypoint"]
CMD ["gateway"]
