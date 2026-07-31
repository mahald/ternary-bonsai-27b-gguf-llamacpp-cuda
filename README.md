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

> **Note:** the HF repo also ships a DSpark speculative-decoding drafter —
> it only works with `:v1` and is not worth it on limited-VRAM hardware; see
> [DSpark drafter — not recommended](#dspark-drafter--not-recommended).

## Quickstart

Use the compose file from this repo — it pins `-ngl`, `-c` and the KV-cache
types. Running the image with your own bare `docker run` command instead is
the classic footgun: without those arguments, current llama.cpp defaults to
`-ngl auto` with `--fit on` and will **silently keep part of a model that
doesn't fit in system RAM** — the server starts without any error and
generates correctly, just ~10× slower (all CPU cores at 100 %). See
[Troubleshooting](#troubleshooting-slow-inference-all-cpu-cores-at-100-).

```bash
# 1. Get the compose setup
git clone https://github.com/mahald/ternary-bonsai-27b-gguf-llamacpp-cuda.git
cd ternary-bonsai-27b-gguf-llamacpp-cuda

# 2. Download the model (7.59 GB, g64 packing for :v2 and newer)
mkdir -p models
curl -L -o models/Ternary-Bonsai-27B-Q2_g64.gguf \
  https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_g64.gguf

# 3. Start (needs Docker with NVIDIA container toolkit)
docker compose up -d

# 4. Test
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
      -np 3
      -c 180000
      --kv-unified
      --no-mmproj
      --cache-type-k q4_0
      --cache-type-v q4_0
      --temp 0.7
      --top-p 0.95
      --top-k 20
      --min-p 0.1
      --xtc-probability 0.8
      --xtc-threshold 0.2
      --presence-penalty 0.3
      --repeat-penalty 1.0
      --cache-ram 4096
      -lv 3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Notes:

- **Context 180000 (~180K)** with **q4_0 KV cache** and the vision projector
  disabled (`--no-mmproj`). The model supports 262K; 180000 is the practical
  limit on 12 GB when also running 3 parallel slots (185K is the max with 2
  slots; the CUDA compute buffers grow with context, so 185K + `-np 3`
  OOMs). Reduce `-c` if you hit OOM, or raise it on bigger GPUs.
- **KV cache at q4_0 for context, not q8_0**: on a 12 GB card you have to
  choose — q8_0 KV buys better attention fidelity (the model card measured its
  headline benchmarks with an FP16 cache) but only ~110K of context; q4_0
  doubles that to ~180K at a small long-range-coherence cost. This repo
  defaults to q4_0 because for long agentic sessions context is the scarcer
  resource. On a 16 GB card, q4_0 KV fits the full `-c 262144`, q8_0 fits
  ~200K.
- **`--no-mmproj`** keeps the vision projector
  (`Ternary-Bonsai-27B-mmproj-*.gguf`) from auto-loading onto the GPU, which
  frees ~1+ GB of VRAM for context. If you want image input, remove
  `--no-mmproj` (llama-server then auto-discovers the mmproj file next to the
  model, or point at it explicitly with `--mmproj`) **and reduce `-c`** (back
  toward ~150K on 12 GB) to make room for the projector.
- **`-np 3` + `--kv-unified`**: three parallel server slots on one unified
  KV-cache pool. Without `--kv-unified`, each slot would get a fixed third of
  `-c` (60000 tokens); with the unified cache, a single request can still use
  the full 180000-token context while the other slots stay available for
  concurrent requests (e.g. parallel pi subagents). This is the most slots
  that fit 12 GB — each slot needs its own compute buffers, and `-np 4`
  OOMs (`CUDA error: out of memory` at startup).

- **No `-fa on`**: flash attention defaults to `auto` in current llama.cpp,
  which already enables it wherever it is supported — setting `-fa on` is
  redundant and force-enables it even in combinations (changed KV-cache quant
  types, other hardware) where auto would fall back, which can cause issues.
  Leave it at the default.
- **`-lv 3`** raises the log verbosity so `docker logs` shows the detail
  needed to verify the setup — buffer placement, offloaded layer count, CUDA
  device table (see
  [Troubleshooting](#troubleshooting-slow-inference-all-cpu-cores-at-100-)).
- **`--cache-ram 4096`** caps the host-RAM prompt cache at 4 GiB.
- **Sampling defaults** are set server-side (`--temp 0.7 --top-p 0.95
  --top-k 20 --min-p 0.1 --xtc-probability 0.8 --xtc-threshold 0.2
  --presence-penalty 0.3 --repeat-penalty 1.0`); they apply to any request
  that doesn't send its own values. `temp 0.7`, `top-p 0.95` and `top-k 20`
  are the model card's own eval values. On top of those, `min-p 0.1` trims the
  low-probability tail, and XTC (exclude-top-choices, probability 0.8 /
  threshold 0.2) prevents the sampler from reflexively locking onto the
  highest-probability token when two candidates are close — a known quality
  boost for thinking models. The mild presence penalty guards against the
  repetition loops thinking models are prone to; set it to 0.0 for maximum
  code fidelity. Don't lower the temperature much — near-greedy decoding makes
  reasoning models loop.
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

## Parallelism expectations (one 12 GB GPU)

Bonsai is served by a single GPU, so parallelism is about *overlapping*
requests, not multiplying throughput:

- **Decode (~40 tok/s) is shared.** Three concurrent subagents each generating
  3K tokens finish no faster than ~3×3K/40 ≈ 225 s of aggregate decode time;
  more slots never makes the sum of output tokens faster.
- **Prefill is the part slots fix.** A subagent prompt of 30K tokens takes
  ~30–60 s to prefill (the new upstream Q2_0 CUDA kernels are prefill-bound at
  ~0.9–1K tok/s). With one slot, that blocks everything; with three, one
  request prefills while another decodes, and short requests overlap fully
  (measured: 3 concurrent short requests finished in 6.8 s wall vs ~15 s
  serial).
- **Requests beyond the slot count queue** (llama-server returns no error;
  they wait for a free slot). pi fan-out of more than 3 concurrent subagents
  therefore completes in waves.
- **KV pool contention**: with all 3 slots active, the shared 180K pool
  divides ~60K per slot; typical subagent contexts fit easily. If you
  regularly need more, drop to `-np 2` to reclaim the 5K back to `-c 185000`
  (or run q8_0 KV at ~110K for better attention fidelity).

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
          "contextWindow": 180000,
          "maxTokens": 16384,
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

### Subagent parallelism and timeouts (pi-subagents extension)

With the [pi-subagents](https://pi.dev/packages/pi-subagents) extension
installed, the server's 3 slots line up with a `globalConcurrencyLimit` of 2
so the parent session always keeps one slot free. This also caps top-level
parallel fan-outs at 2 concurrent subagents (they complete in waves; see
[Parallelism expectations](#parallelism-expectations-one-12-gb-gpu)).

`~/.pi/agent/extensions/subagent/config.json`:

```json
{
	"globalConcurrencyLimit": 2,
	"parallel": { "concurrency": 2 }
}
```

The builtin agents' default 30-minute run timeout can be too tight for Bonsai
on a 12 GB GPU (slow prefill + ~40 tok/s decode). Eject them to user scope
and add `timeoutMs` to the frontmatter (60 min here):

```bash
# for each agent you use: worker, reviewer, researcher, oracle, scout, ...
pi subagents eject worker          # creates ~/.pi/agent/agents/worker.md
# add to the frontmatter:
#   timeoutMs: 3600000
```

Or override per run with `timeoutMs` / `maxRuntimeMs` on the `subagent`
tool call (a per-call value always wins over the agent default).

> **Testing note:** this setup is primarily tested with the pi agent using the
> [pi-effort](https://pi.dev/packages/pi-effort) extension
> (`pi install npm:pi-effort`) with the reasoning effort set to `medium`
> (`/effort medium`). Other effort levels and clients should work but see less
> coverage.

## pi agent configuration examples

The [`examples/pi/`](examples/pi/) directory mirrors the config used and
tested with this repo — copy them into `~/.pi/agent/` as needed:

| Example file | Destination | Purpose |
|---|---|---|
| `examples/pi/models.json.example` | `~/.pi/agent/models.json` | bonsai-local provider only (`apiKey: "none"`) |
| `examples/pi/models.cloud.example` | — | bonsai-local **plus** a cloud provider with a `\<YOUR_API_KEY\>` placeholder; **never commit real keys** |
| `examples/pi/settings.json.example` | `~/.pi/agent/settings.json` | defaults (`bonsai-27b`, thinking `medium`) + the extension packages |
| `examples/pi/subagent-config.json` | `~/.pi/agent/extensions/subagent/config.json` | pi-subagents concurrency caps: 2 subagents max, 2 parallel |
| `examples/pi/agents/worker.md.example` | `~/.pi/agent/agents/worker.md` | ejected `worker` with `timeoutMs: 3600000` (60 min) in the frontmatter |

**Required extension:** the subagent config only takes effect with
[pi-subagents](https://pi.dev/packages/pi-subagents) installed:

```bash
pi install npm:pi-subagents
```

Optional but used in this setup: `npm:pi-effort` (reasoning effort control),
`npm:pi-memory` (memory files).

Apply with e.g.:

```bash
cp examples/pi/models.json.example ~/.pi/agent/models.json
cp examples/pi/subagent-config.json ~/.pi/agent/extensions/subagent/config.json
mkdir -p ~/.pi/agent/agents
cp examples/pi/agents/worker.md.example ~/.pi/agent/agents/worker.md
```

> **Security:** the examples contain no secrets. Your real `~/.pi/agent/`
> files may hold API keys (`auth.json`, `models.json` with cloud providers) —
> never copy or commit those. Treat `.pi-subagents/` (subagent transcripts)
> as sensitive too; it is gitignored in this repo.

## Troubleshooting: slow inference, all CPU cores at 100 %

If the server starts cleanly, generates correct output, but runs ~10× slower
than expected with every CPU core pegged — the model is (partly) running from
system RAM. Bonsai is a **dense** model: whatever share doesn't compute in
VRAM streams through system memory on every token, so there is no graceful
middle ground. Two llama.cpp defaults cause this silently:

- **`-ngl auto` + `--fit on`** (the defaults when you don't pass `-ngl`/`-c`):
  instead of failing on OOM, llama.cpp fills VRAM up to a margin and quietly
  keeps the remaining layers and KV cache in system RAM, auto-trimming the
  context. No error is printed. The compose file avoids this by pinning
  `-ngl 99` and `-c` — with those set, a config that doesn't fit fails
  **loudly** at startup, which is what you want.
- **mmproj auto-load**: if the vision projector
  (`Ternary-Bonsai-27B-mmproj-*.gguf`) sits next to the model file,
  llama-server auto-discovers it (`--mmproj-auto` defaults to on) and places
  it on the GPU first, taking ~1+ GB of VRAM before the text weights. The
  compose file guards against this with `--no-mmproj`; remove that flag only
  if you want image input, and reduce `-c` to give the projector its ~1+ GB
  back.

To diagnose, check `docker logs` — the compose file already runs with
`-lv 3` (verbose logging), so the log shows where the buffers land
(`CUDA0 model buffer size` vs `CPU model buffer size`, `offloaded X/Y layers
to GPU`) and the CUDA device table (`ggml_cuda_init: found N CUDA devices`).

Context sizing with flash attention (on by default): `-c 180000` with q4_0 KV,
`-np 3` and `--no-mmproj` uses ~11.8 GiB of the 12 GB card (~0.5 GiB
headroom). Measured limits on this card: q4_0 KV maxes at ~185K with 2 slots
or ~180K with 3 slots; q8_0 KV maxes at ~110K; the compute buffers grow with
context, so the cliff is steep. On 16 GB, q4_0 KV fits the full `-c 262144`,
q8_0 fits ~200K. If it OOMs, step `-c` down — the loud failure marks your
card's real maximum.

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
