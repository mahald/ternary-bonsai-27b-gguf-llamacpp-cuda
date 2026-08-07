ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.3.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# v4 builds buun-llama-cpp, an experimental llama.cpp fork that adds the
# TurboQuant KV cache codecs and VBR (variable bit-rate KV cache). It carries
# the upstream CUDA Q2_0 backend, so the model file is unchanged from :v3
# (g64 packing, QK2_0=64).
#
# To go back to plain official llama.cpp, override both build args:
#   --build-arg LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp \
#   --build-arg LLAMACPP_REF=<a master commit>
ARG LLAMACPP_REPO=https://github.com/spiritbuun/buun-llama-cpp
ARG LLAMACPP_REF=7b9bf9d5052921d299a8fca947897514caea6261
# CUDA architectures to compile for. "default" lets ggml pick its broad
# multi-arch set — use this for published images. For a fast local build,
# narrow it to your GPU, e.g. --build-arg CUDA_DOCKER_ARCH=89 (Ada / RTX 40xx),
# or use Dockerfile.native, which detects it for you.
#
# Note: CUDA 13 dropped Maxwell, Pascal and Volta, so the "default" set here
# starts at Turing (7.5) — unlike the CUDA 12 based :v3 image.
ARG CUDA_DOCKER_ARCH=default

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential cmake python3 git libssl-dev libgomp1

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /app
RUN git init -q . && \
    git remote add origin ${LLAMACPP_REPO} && \
    git fetch --depth 1 origin ${LLAMACPP_REF} && \
    git checkout -q FETCH_HEAD

# GGML_CUDA_FA_ALL_QUANTS is required, not optional: without it the CMake glob
# only picks up the scalar turbo fattn-vec instances and skips the TCQ ones
# (turbo3_tcq / turbo2_tcq / turbo1_tcq) — exactly the tiers VBR degrades into.
#
# --allow-shlib-undefined: the VBR VMM pool calls the CUDA driver API
# (cuMemAddressFree etc.). libcuda.so only exists at runtime, so without this
# the executable link fails on undefined references.
RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        ${CMAKE_ARGS} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS server

RUN apt-get update && \
    apt-get install -y libgomp1 curl && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /app/lib/ /app
COPY --from=build /app/build/bin/llama-server /app

ENV LLAMA_ARG_HOST=0.0.0.0
WORKDIR /app
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]
