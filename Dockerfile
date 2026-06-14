# hotSpring guideStone — OCI container image
#
# Packages the pre-built validation artifact into a portable container
# that runs on any OS with Docker/Podman (Linux, macOS, Windows).
#
# Build:
#   docker build -t hotspring-guidestone:v0.7.0 .
#
# Run:
#   docker run --rm -v ./results:/opt/validation/results hotspring-guidestone:v0.7.0 validate
#
# GPU:
#   docker run --rm --gpus all -v ./results:/opt/validation/results hotspring-guidestone:v0.7.0 validate

FROM ubuntu:22.04 AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libvulkan1 \
        mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/validation

COPY validation/bin/          bin/
COPY validation/expected/     expected/
COPY validation/shaders/      shaders/
COPY validation/hotspring     hotspring
COPY validation/_lib.sh       _lib.sh
COPY validation/run           run
COPY validation/run-matrix    run-matrix
COPY validation/benchmark     benchmark
COPY validation/chuna-engine  chuna-engine
COPY validation/deploy-nucleus deploy-nucleus
COPY validation/run-overnight run-overnight
COPY validation/CHECKSUMS     CHECKSUMS
COPY validation/README        README
COPY validation/GUIDESTONE.md GUIDESTONE.md
COPY validation/LICENSE       LICENSE

RUN chmod +x hotspring _lib.sh run run-matrix benchmark chuna-engine \
              deploy-nucleus run-overnight \
    && find bin/ -type f -exec chmod +x {} +

RUN mkdir -p results

ENTRYPOINT ["./hotspring"]
CMD ["help"]
