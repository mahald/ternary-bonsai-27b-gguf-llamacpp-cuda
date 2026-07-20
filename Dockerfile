ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# Official llama.cpp master plus the open CUDA Q2_0 PR (#25707, by PrismML).
# The pinned ref is GitHub's precomputed merge commit of refs/pull/25707/merge
# (master 4937ca83 + PR head 5eec7982). GitHub recomputes that ref as master
# moves, so if the fetch below ever fails, re-resolve it with
#   git ls-remote https://github.com/ggml-org/llama.cpp refs/pull/25707/merge
# and update LLAMACPP_REF. Once the PR is merged, pin a master commit instead.
# This build expects the g64 packing (QK2_0=64): use Ternary-Bonsai-*-Q2_g64.gguf.
ARG LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp
ARG LLAMACPP_REF=57f11de0e1d4ff48dd8722e1fe8d65535bb499cd
# CUDA architectures to compile for. "default" lets ggml pick its broad
# multi-arch set (Turing through Hopper/Blackwell) — use this for published
# images. For a fast local build, narrow it to your GPU, e.g.
# --build-arg CUDA_DOCKER_ARCH=89 (Ada / RTX 40xx).
ARG CUDA_DOCKER_ARCH=default

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential cmake python3 git libssl-dev libgomp1

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /app
RUN git init -q . && \
    git remote add origin ${LLAMACPP_REPO} && \
    git fetch --depth 1 origin ${LLAMACPP_REF} && \
    git checkout -q FETCH_HEAD

RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
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
