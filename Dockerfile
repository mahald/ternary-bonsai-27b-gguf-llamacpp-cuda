ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# PrismML fork of llama.cpp — required for the Q2_0 ternary kernels
# (upstream llama.cpp cannot load the Bonsai Q2_0 GGUF format).
ARG LLAMACPP_REPO=https://github.com/PrismML-Eng/llama.cpp
ARG LLAMACPP_REF=9fcaed763ccda38ea81068ad9d7f991aaddca451
# CUDA architectures to compile for. 89 = Ada (RTX 40xx).
# Broaden with e.g. "80;86;89;90" (A100, RTX 30xx, RTX 40xx, H100) at the
# cost of a much longer build.
ARG CUDA_DOCKER_ARCH=89

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential cmake python3 git libssl-dev libgomp1

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /app
RUN git init -q . && \
    git remote add origin ${LLAMACPP_REPO} && \
    git fetch --depth 1 origin ${LLAMACPP_REF} && \
    git checkout -q FETCH_HEAD

RUN cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} \
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
