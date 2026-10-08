# syntax=docker/dockerfile:1
#
# FORGE Optimization Runtime: solver CLI, checker, and web demo console.
#
#   docker build -t forge .
#   docker run --rm -p 8765:8765 forge                      # web console
#   docker run --rm -v "$PWD:/work" -w /work forge …
#
# Base is Ubuntu 24.04 with its stock GCC 13 / CMake 3.28, the same toolchain
# the native builds use, so pivot counts and results match a host build
# (deterministic FP stays ON, native arch stays OFF).

ARG UBUNTU=ubuntu:24.04

# ---------------------------------------------------------------- build ----
FROM ${UBUNTU} AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake python3 zlib1g-dev libboost-dev \
        libvulkan-dev glslang-tools \
    && rm -rf /var/lib/apt/lists/*

# The Vulkan backend bakes the absolute shader path (build/shaders) into the
# binary, so the build tree lives at the same path the runtime stage uses.
WORKDIR /opt/forge
COPY CMakeLists.txt ./
COPY src     src
COPY apps    apps
COPY tests   tests
COPY scripts scripts
COPY bindings bindings
COPY examples examples
# Instance files the ctest suite reads (build stage only; not shipped).
COPY benchmarks/netlib/mps     benchmarks/netlib/mps
COPY benchmarks/miplib-easy/mps benchmarks/miplib-easy/mps
COPY benchmarks/miplib2017/benchmark-v2.test benchmarks/miplib2017/benchmark-v2.test
COPY web/lp_text.py web/lp_text.py

ARG BUILD_TYPE=RelWithDebInfo
ARG ENABLE_VULKAN=ON
# RUN_TESTS=1 builds and runs the full ctest suite before producing the image.
ARG RUN_TESTS=0

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DSOR_ENABLE_VULKAN=${ENABLE_VULKAN} \
    && cmake --build build -j"$(nproc)" --target sor_solve sor_check \
    && if [ "${RUN_TESTS}" = "1" ]; then \
           cmake --build build -j"$(nproc)" && ctest --test-dir build --output-on-failure; \
       fi \
    && mkdir -p build/shaders

# -------------------------------------------------------------- runtime ----
FROM ${UBUNTU} AS runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-venv zlib1g libvulkan1 mesa-vulkan-drivers tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/forge

COPY web/requirements.txt web/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r web/requirements.txt

COPY --from=build /opt/forge/build/sor_solve /opt/forge/build/sor_check build/
COPY --from=build /opt/forge/build/shaders   build/shaders
COPY web      web
COPY examples examples
COPY scripts  scripts
COPY bindings bindings

RUN useradd --create-home --uid 10001 forge \
    && chown -R forge:forge /opt/forge/examples

ENV PATH="/opt/forge/build:/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    FORGE_BIN_DIR=/opt/forge/build \
    FORGE_BINARY_DIR=/opt/forge/build \
    FORGE_MODEL_DIRS=/opt/forge/examples \
    FORGE_WEB_HOST=0.0.0.0 \
    FORGE_WEB_PORT=8765

USER forge
EXPOSE 8765

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/opt/forge/web/app.py"]
