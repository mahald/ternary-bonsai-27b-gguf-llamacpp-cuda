# Ternary Bonsai 27B — llama.cpp CUDA server

Docker image and compose setup for serving
[prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
with CUDA and an OpenAI-compatible API.

The current image (`:v3` / `:latest`) builds **plain official llama.cpp
master** — the CUDA Q2_0 backend
([PR #25707](https://github.com/ggml-org/llama.cpp/pull/25707)) was merged
upstream on 2026-07-30, so all Q2_0 backends (CPU, Metal, Vulkan, CUDA) are
now upstream and no patches are carried anymore.

Pick the model file that matches the image tag — the two Q2_0 packings are
**incompatible**:

| Image tag | llama.cpp source | Model file |
|---|---|---|
| `:v3`, `:latest` | official master (CUDA Q2_0 merged; g64, `QK2_0=64`) | `Ternary-Bonsai-27B-Q2_g64.gguf` |
| `:v2` | official master + then-open PR #25707 (g64, `QK2_0=64`) | `Ternary-Bonsai-27B-Q2_g64.gguf` |
| `:v1` | [PrismML fork](https://github.com/PrismML-Eng/llama.cpp) (g128, `QK2_0=128`) | `Ternary-Bonsai-27B-Q2_0.gguf` |

- Image: `mhald/ternary-bonsai-27b-gguf-llamacpp-cuda` (see [Links](#links))
- Endpoint: `http://127.0.0.1:8080/v1` (`/v1/chat/completions`, `/v1/models`, …), no API key
- Model name: `bonsai-27b`, web UI at http://127.0.0.1:8080

## Quickstart

```bash
# 1. Download the model (7.59 GB, g64 packing for :v2 and newer)
mkdir -p models
curl -L -o models/Ternary-Bonsai-27B-Q2_g64.gguf \
  https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_g64.gguf

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
    image: mhald/ternary-bonsai-27b-gguf-llamacpp-cuda:v3
    container_name: bonsai-27b
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
    command: >
      -m /models/Ternary-Bonsai-27B-Q2_g64.gguf
      -a bonsai-27b
      --host 0.0.0.0
      --port 8080
      -ngl 99
      -c 153600
      -fa on
      --cache-type-k q4_0
      --cache-type-v q4_0
      --temp 0.6
      --top-p 0.95
      --top-k 20
      --min-p 0.0
      --presence-penalty 0.3
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

- **Context 153600 (150K)** with flash attention and 4-bit KV cache. The model
  supports 262K, but the g64 file is 0.42 GB larger than the old g128 one, so
  150K is the practical limit on 12 GB (measured: ~11.3 of 12.3 GiB). Reduce
  `-c` if you hit OOM, or raise it on bigger GPUs.
- **Sampling defaults** are set server-side (`--temp 0.6 --top-p 0.95
  --top-k 20 --min-p 0.0 --presence-penalty 0.3 --repeat-penalty 1.0`); they
  apply to any request that doesn't send its own values. The mild presence
  penalty guards against the repetition loops thinking models are prone to;
  set it to 0.0 for maximum code fidelity. Don't lower the temperature much —
  near-greedy decoding makes reasoning models loop.
- **Repetition loops?** Stick to the neutral sampling defaults first. In our
  testing, enabling the DRY sampler (`--dry-multiplier 0.8`) made this model
  hallucinate — thinking models legitimately repeat phrases while reasoning,
  and DRY forces them off the correct path. If loops persist, try a mild
  `--presence-penalty 0.3` instead.
- Use `Ternary-Bonsai-27B-Q2_g64.gguf` with `:v2` and newer. The old
  `Q2_0.gguf` (g128) does **not** load with these images — it only works with
  `:v1` (PrismML fork, `QK2_0 = 128`).
- Measured: ~42 tok/s generation on an RTX 4080 Laptop GPU with v3
  (v2: ~39 tok/s, v1/g128: ~44 tok/s).

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
          "contextWindow": 153600,
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

> **Testing note:** this setup is primarily tested with the pi agent using the
> [pi-effort](https://pi.dev/packages/pi-effort) extension
> (`pi install npm:pi-effort`) with the reasoning effort set to `medium`
> (`/effort medium`). Other effort levels and clients should work but see less
> coverage.

## Building

Locally:

```bash
docker build -t mhald/ternary-bonsai-27b-gguf-llamacpp-cuda:dev .
```

The Dockerfile pins `LLAMACPP_REF` to an official llama.cpp master commit
(since v3; the CUDA Q2_0 PR #25707 is merged upstream). To move to a newer
llama.cpp, update `LLAMACPP_REF` to a newer master commit. By default the
image compiles
for a broad multi-arch CUDA set (ggml's default; Turing through
Hopper/Blackwell), so the published images run on most NVIDIA GPUs. The
container ships its own CUDA 12.8 runtime — the host only needs a reasonably
recent NVIDIA driver. For a much faster local build, narrow the target to
your GPU, e.g. `--build-arg CUDA_DOCKER_ARCH=89` (Ada / RTX 40xx).

CI: pushes to `main` publish `:latest`, tags `v*` publish the version tag to
Docker Hub (GitHub Actions, needs `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`
repo secrets).

## DSpark drafter — not recommended

The HF repo also ships a DSpark speculative-decoding drafter
(`Ternary-Bonsai-27B-dspark-Q4_1.gguf`). It only loads with `:v1`: the file
uses the PrismML fork's own format (GGUF architecture `dspark`, g128
embedding), which official llama.cpp — and therefore `:v2`/`:v3` — rejects
(upstream's DSpark implementation only supports DeepSeek-style `dflash`
drafters).

Even on `:v1` it makes no sense on consumer hardware, especially laptop GPUs
with limited VRAM: the drafter costs ~5 GB extra (~2 GB weights plus a ~3 GB
spec-decoding buffer that does not shrink with context), which collapses the
usable context window — and once layers spill to the CPU it is a net
*slowdown* (measured on a 12 GB RTX 4080 Laptop). prism-ml's 1.34× speedup
figure was measured on an H100 (80 GB). Skip the drafter unless you run `:v1`
on a large-VRAM card.

## License

Setup files: MIT. llama.cpp is MIT, the Bonsai model is Apache 2.0
(see their repositories).

## Links

- **Docker Hub (built images):**
  [mhald/ternary-bonsai-27b-gguf-llamacpp-cuda](https://hub.docker.com/r/mhald/ternary-bonsai-27b-gguf-llamacpp-cuda)
  — `:latest` (tracks `main`) · `:v3` (official master, g64) · `:v2` (master + PR #25707, g64) · `:v1` (PrismML fork, g128)
- **This repo:** https://github.com/mahald/ternary-bonsai-27b-gguf-llamacpp-cuda
- **Model:** https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
- **CUDA Q2_0 PR (merged upstream, in `:v2` and newer):** https://github.com/ggml-org/llama.cpp/pull/25707
- **llama.cpp fork (g128 kernels, in `:v1`):** https://github.com/PrismML-Eng/llama.cpp
- **Whitepaper & demos:** https://github.com/PrismML-Eng/Bonsai-demo
