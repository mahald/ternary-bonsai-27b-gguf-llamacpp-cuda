# Ternary Bonsai 27B — llama.cpp CUDA server

Docker image and compose setup for serving
[prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
with CUDA and an OpenAI-compatible API.

The image builds the [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
(pinned commit) — **upstream llama.cpp cannot load the Bonsai Q2_0 ternary format**.

- Image: `mhald/ternary-bonsai-27b-gguf-llamacpp-cuda` (see [Links](#links))
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

## docker-compose.yml example

The [`docker-compose.yml`](docker-compose.yml) in this repo, tuned for a
12 GB GPU (RTX 4080 Laptop):

```yaml
services:
  bonsai:
    image: mhald/ternary-bonsai-27b-gguf-llamacpp-cuda:v1
    container_name: bonsai-27b
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
    command: >
      -m /models/Ternary-Bonsai-27B-Q2_0.gguf
      -a bonsai-27b
      --host 0.0.0.0
      --port 8080
      -ngl 99
      -c 184320
      -fa on
      --cache-type-k q4_0
      --cache-type-v q4_0
      --temp 0.6
      --top-p 0.95
      --top-k 20
      --min-p 0.0
      --presence-penalty 0.0
      --repeat-penalty 1.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Notes:

- **Context 184320 (180K)** with flash attention and 4-bit KV cache. The model
  supports 262K, but that needs ~12.8 GB VRAM. At 180K, usage is ~11.7 of
  12.3 GiB — reduce `-c` if you hit OOM, or raise it on bigger GPUs.
- **Sampling defaults** are set server-side (`--temp 0.6 --top-p 0.95
  --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0`); they
  apply to any request that doesn't send its own values.
- Use `Ternary-Bonsai-27B-Q2_0.gguf` (g128). The `Q2_g64` variant does **not**
  load with the current fork master (`QK2_0 = 128` expects the g128 packing).
- Measured: ~44 tok/s generation on an RTX 4080 Laptop GPU.

## Using with the pi agent

Example for the [pi coding agent](https://pi.dev) — add the server as a
custom provider in `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "bonsai-local": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        {
          "id": "bonsai-27b",
          "name": "Ternary Bonsai 27B (local)",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 184320,
          "maxTokens": 8192,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```

Optionally make it the default in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "bonsai-local",
  "defaultModel": "bonsai-27b"
}
```

pi has no sampling settings of its own — the sampling defaults configured
server-side in the compose file apply. Keep `contextWindow` in sync with the
server's `-c` value.

## Building

Locally:

```bash
docker build -t mhald/ternary-bonsai-27b-gguf-llamacpp-cuda:dev .
```

The Dockerfile pins the fork commit (`LLAMACPP_REF`). By default it compiles
for a broad multi-arch CUDA set (ggml's default; Turing through
Hopper/Blackwell), so the published images run on most NVIDIA GPUs. The
container ships its own CUDA 12.8 runtime — the host only needs a reasonably
recent NVIDIA driver. For a much faster local build, narrow the target to
your GPU, e.g. `--build-arg CUDA_DOCKER_ARCH=89` (Ada / RTX 40xx).

CI: pushes to `main` publish `:latest`, tags `v*` publish the version tag to
Docker Hub (GitHub Actions, needs `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`
repo secrets).

## License

Setup files: MIT. llama.cpp (PrismML fork) and the Bonsai model are Apache 2.0
(see their repositories).

## Links

- **Docker Hub (built images):**
  [mhald/ternary-bonsai-27b-gguf-llamacpp-cuda](https://hub.docker.com/r/mhald/ternary-bonsai-27b-gguf-llamacpp-cuda)
  — `:latest` (tracks `main`) · `:v1`
- **This repo:** https://github.com/mahald/ternary-bonsai-27b-gguf-llamacpp-cuda
- **Model:** https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
- **llama.cpp fork (ternary kernels):** https://github.com/PrismML-Eng/llama.cpp
- **Whitepaper & demos:** https://github.com/PrismML-Eng/Bonsai-demo
