# Ternary Bonsai 27B — llama.cpp CUDA server

Docker image and compose setup for serving
[prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
with CUDA and an OpenAI-compatible API.

The image builds the [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
(pinned commit) — **upstream llama.cpp cannot load the Bonsai Q2_0 ternary format**.

- Image: `mhald/ternary-bonsai-27b-gguf-llamacpp-cuda`
- Endpoint: `http://127.0.0.1:8080/v1` (`/v1/chat/completions`, `/v1/models`, …), no API key
- Model name: `bonsai-27b`, web UI at http://127.0.0.1:8080

## Quickstart

```bash
# 1. Download the model (7.17 GB)
mkdir -p models
curl -L -o models/Ternary-Bonsai-27B-Q2_0.gguf \
  https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_0.gguf

# 2. Start (needs Docker with NVIDIA container toolkit)
docker compose up -d

# 3. Test
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "bonsai-27b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 2048
}'
```

Bonsai is a thinking model: the final answer is in `message.content`, the
reasoning in `message.reasoning_content`. Set `max_tokens` generously
(reasoning tokens count toward the limit).

## Configuration notes

`docker-compose.yml` defaults (tuned for a 12 GB RTX 4080 Laptop GPU):

- **Context 184320 (180K)** with flash attention and 4-bit KV cache
  (`-fa on --cache-type-k q4_0 --cache-type-v q4_0`). The model supports 262K,
  but that needs ~12.8 GB. At 180K, VRAM usage is ~11.7 of 12.3 GiB — reduce
  `-c` if you hit OOM.
- **Sampling defaults** set server-side: temp 0.6, top-p 0.95, top-k 20,
  min-p 0.0, presence-penalty 0.0, repeat-penalty 1.0.
- Measured throughput: ~44 tok/s generation on an RTX 4080 Laptop GPU.

Use `Ternary-Bonsai-27B-Q2_0.gguf` (g128). The `Q2_g64` variant does **not**
load with the current fork master (`QK2_0 = 128` expects the g128 packing).

## Building

Locally:

```bash
docker build -t mhald/ternary-bonsai-27b-gguf-llamacpp-cuda:dev .
```

The Dockerfile pins the fork commit (`LLAMACPP_REF`) and compiles for CUDA
architecture 8.9 (Ada / RTX 40xx) by default; override with
`--build-arg CUDA_DOCKER_ARCH="80;86;89;90"` for a broader image.

CI: pushes to `main` publish `:latest`, tags `v*` publish the version tag to
Docker Hub (GitHub Actions, needs `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`
repo secrets).

## License

Setup files: MIT. llama.cpp (PrismML fork) and the Bonsai model are Apache 2.0
(see their repositories).
