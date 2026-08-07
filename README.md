# Ternary Bonsai 27B — llama.cpp CUDA server

Docker image and compose setup for serving
[prism-ml/Ternary-Bonsai-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
with CUDA and an OpenAI-compatible API.

> **`:v4` is experimental.** It no longer builds official llama.cpp but
> [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp), a research
> fork whose own README opens with *"this is a highly experimental fork of
> llama.cpp, use at your own discretion"*. What it buys is the reason for the
> switch: **the model's full 262144-token context now fits on a 12 GB card**,
> which it never did with upstream KV quantization. If you want the
> conservative setup, stay on `:v3` — it is unchanged and still works.

The switch is only about the KV cache. The fork carries the same upstream
CUDA Q2_0 backend
([PR #25707](https://github.com/ggml-org/llama.cpp/pull/25707), merged
2026-07-30), so **the model file is unchanged from `:v3`**.

Pick the model file that matches the image tag — the two Q2_0 packings are
**incompatible**:

| Image tag | llama.cpp source | KV cache | Max context on 12 GB | Model file |
|---|---|---|---|---|
| `:v4`, `:latest` | [buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) (fork; Q2_0 g64) | **VBR** (f16 → turbo1_tcq) | **262144** | `Ternary-Bonsai-27B-Q2_g64.gguf` |
| `:v3` | official master (CUDA Q2_0 merged; g64, `QK2_0=64`) | q4_0, fixed | 180000 | `Ternary-Bonsai-27B-Q2_g64.gguf` |
| `:v2` | official master + then-open PR #25707 (g64, `QK2_0=64`) | q4_0, fixed | 180000 | `Ternary-Bonsai-27B-Q2_g64.gguf` |
| `:v1` | [PrismML fork](https://github.com/PrismML-Eng/llama.cpp) (g128, `QK2_0=128`) | q4_0, fixed | 180000 | `Ternary-Bonsai-27B-Q2_0.gguf` |

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
    image: mhald/ternary-bonsai-27b-gguf-llamacpp-cuda:v4
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
      -c 262144
      -ct vbr
      --kv-unified
      --no-mmproj
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

- **Context 262144** — the model's full training length, with the vision
  projector disabled (`--no-mmproj`). This is what `:v4` is for; see
  [VBR](#vbr--the-reason-for-v4) below. Reduce `-c` if you hit OOM.
- **`-ct vbr` instead of fixed `--cache-type-k/-v`**: the KV cache is no
  longer one quantization for the whole session. It starts at **f16** and
  degrades one (layer, side) tensor at a time, cheapest-first, only when VRAM
  pressure demands it. A shallow session therefore runs at full FP16 fidelity
  — strictly better than `:v3`, which paid q4_0 from the first token — and a
  262144-token session ends up in a turbo3_tcq / turbo2_tcq mixture instead of
  not fitting at all.
- **`--no-mmproj`** keeps the vision projector
  (`Ternary-Bonsai-27B-mmproj-*.gguf`) from auto-loading onto the GPU, which
  frees ~1+ GB of VRAM for context. If you want image input, remove
  `--no-mmproj` (llama-server then auto-discovers the mmproj file next to the
  model, or point at it explicitly with `--mmproj`). Under VBR you do not
  strictly have to reduce `-c` for it — the projector simply shrinks the
  auto-derived KV budget, so the cache degrades earlier — but it costs
  quality at depth. The fork also offers `--mmproj-gpu-swap` as an
  alternative; untested with this model.
- **`-np 3` + `--kv-unified`**: three parallel server slots on one unified
  KV-cache pool. Without `--kv-unified`, each slot would get a fixed third of
  `-c` (87381 tokens); with the unified cache, a single request can still use
  the full 262144-token context while the other slots stay available for
  concurrent requests (e.g. parallel pi subagents). Dynamic VBR forces
  `--kv-unified` anyway as soon as `-np > 1`; it is spelled out here so the
  config still reads correctly if you ever drop `-ct vbr`.
- **No `-fa on`**: flash attention defaults to `auto`, which already enables
  it wherever it is supported. With turbo-typed KV the fork force-enables it
  regardless, so the flag would be redundant twice over.
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
- Measured on an RTX 4080 Laptop GPU: **~45 tok/s** generation with v4
  (v3: ~42, v2: ~39, v1/g128: ~44).

## VBR — the reason for v4

### Why the KV cache is the whole problem

Bonsai reports itself as architecture `qwen35`: 64 blocks, but
`full_attention_interval = 4`, and the other three out of every four layers
are SSM (recurrent) layers. So **only 16 layers hold a KV cache** — the
recurrent state is a constant 449 MiB no matter how long the context gets.

Those 16 layers have 4 KV heads with a key/value length of 256, i.e.
16 × 2 × 4 × 256 = **32768 values per token**. Multiply by 262144 tokens and
you get exactly 8 Gi values, which makes the arithmetic unusually clean:

> **KV cache size in GiB at the full 262144-token context = the codec's
> bits-per-value.**

| KV codec | bpv | KV @ 262144 | + 7.06 GiB weights | fits 12 GB? |
|---|---|---|---|---|
| f16 | 16.0 | 16.00 GiB | 23.1 GiB | no |
| q8_0 | 8.5 | 8.50 GiB | 15.6 GiB | no |
| q4_0 (`:v3`) | 4.5 | 4.50 GiB | 11.6 GiB | no (+ ~0.8 GiB buffers) |
| turbo4 | 4.125 | 4.13 GiB | 11.2 GiB | no |
| turbo3_tcq | 3.25 | 3.25 GiB | 10.3 GiB | tight |
| turbo2_tcq | 2.25 | 2.25 GiB | 9.3 GiB | yes |
| turbo1_tcq | 1.25 | 1.25 GiB | 8.3 GiB | yes |

That is why `:v3` stopped at 180000 tokens, and why a fixed low-bit codec is
not the answer either: you would pay 2-bit quality for the whole session just
to survive a depth you usually never reach.

### What VBR actually does

VBR (variable bit-rate) allocates the cache in a CUDA VMM pool: it reserves
the full f16-sized virtual address range (16388 MiB here) but maps physical
pages only as the context grows, against a VRAM budget it derives from
whatever is left after weights and compute buffers. The cache **starts at
f16** and, when the budget is about to be exceeded, degrades one
(layer, side) tensor at a time — transcoding on a side stream, cheapest unit
first, following a price order measured per architecture. For `qwen35` the
fork ships a baked order of **160 steps** (16 KV layers × 2 sides × 5 tiers),
which the server confirms on startup:

```
vbr_load_degrade_order: VBR degrade order: 160 baked steps (arch-matched, matrix v3)
operator(): VBR dynamic: KV VRAM budget 2853 MiB (auto, from remaining memory)
```

Explicit `-ct vbr` opens the full ladder down to `turbo1_tcq`; without it,
VBR is still on but stops at a `turbo4` floor. Explicit `-c` bypasses the
capacity estimator, which is what makes "just give me the model's full
context" work.

### Measured (RTX 4080 Laptop, 12 GB)

Single 249630-token prompt at `-c 262144 -ct vbr -np 3 --kv-unified`. The
ranges below span two separate builds of the same fork commit — one against
CUDA 12.8.1, one against 13.3.1 — because the CUDA major version turned out to
make no measurable difference. The image ships 13.3.1.

| | |
|---|---|
| VRAM after startup | 8047–8111 MiB (weights 6882, SSM state 449, compute 379) |
| VRAM at 249 k tokens | **11709 / 12282 MiB** — no OOM |
| Prefill | 864–876 s for 249630 tokens, avg **285–289 t/s** |
| Generation, shallow | **43.6–45.5 t/s** |
| Generation at 249 k depth | 18.1 t/s |
| Degrade steps fired | **112–114 of 160** |
| Final cache mixture | mostly turbo3_tcq / turbo2_tcq, a few units at turbo1_tcq |

Prefill is the cost you actually feel: it starts around 840 t/s and falls to
~145 t/s past 200 k tokens, so filling the window once takes ~15 minutes.
Depth is cheap to *hold*, not cheap to *reach*.

The first degrade fires at ~43000 tokens (where f16 exhausts the 2853 MiB
budget), and it starts at layer 63 — the last full-attention layer, i.e. the
one the price order rates least sensitive. Everything below that depth runs
on an untouched f16 cache.

Useful knobs:

- `--vbr-vram 3G` — override the auto-derived KV budget (auto leaves ~1.3 GiB
  headroom on this card; raising it trades safety margin for cache quality).
- `--vbr-floor t3` — refuse to degrade below a tier. Careful: with a floor
  above ~t2 the full 262144 window no longer fits on 12 GB.
- Run with `-v` (or the compose file's `-lv 3`) to watch `VBR degrade #…` in
  `docker logs`.

### What VBR costs you

Dynamic VBR is not free of trade-offs, and they are worth knowing before you
switch off `:v3`:

- **Context shift / self-extend and slot save-restore are disabled.** A tier
  flip would invalidate any snapshot taken before it, so the fork turns those
  off in dynamic mode (context checkpoints stay enabled on hybrid models such
  as this one). Generation stops cleanly when the context fills instead of
  sliding the window.
- **Flash attention is force-enabled**, so a backend that cannot do FA is not
  an option.
- **KV that lands on the CPU falls back to q8_0** — irrelevant here, since
  the compose file pins `-ngl 99`, but it means VBR is a GPU-only feature.
- **It is an experimental fork.** No upstream release process, no CI matrix of
  the size llama.cpp has.

## Parallelism expectations (one 12 GB GPU)

Bonsai is served by a single GPU, so parallelism is about *overlapping*
requests, not multiplying throughput:

- **Decode (~45 tok/s) is shared.** Three concurrent subagents each generating
  3K tokens finish no faster than ~3×3K/45 ≈ 200 s of aggregate decode time;
  more slots never makes the sum of output tokens faster. Decode also slows
  with depth (~18 tok/s at 249 k tokens), independently of how many slots run.
- **Prefill is the part slots fix.** A subagent prompt of 30K tokens takes
  ~30–60 s to prefill (the new upstream Q2_0 CUDA kernels are prefill-bound at
  ~0.9–1K tok/s). With one slot, that blocks everything; with three, one
  request prefills while another decodes, and short requests overlap fully
  (measured: 3 concurrent short requests finished in 6.8 s wall vs ~15 s
  serial).
- **Requests beyond the slot count queue** (llama-server returns no error;
  they wait for a free slot). pi fan-out of more than 3 concurrent subagents
  therefore completes in waves.
- **KV pool contention**: with all 3 slots active, the shared 262K pool
  divides ~87K per slot; typical subagent contexts fit easily. Note that under
  VBR the slots also share the *quality* budget — three deep sessions push the
  cache down the tier ladder faster than one does, because the degrade
  controller sees the pool's total mapped bytes, not any single slot's.

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
          "contextWindow": 262144,
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
on a 12 GB GPU (slow prefill + ~45 tok/s decode). Eject them to user scope
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

Context sizing under VBR (`:v4`) is no longer a cliff you have to find by
bisection: `-c 262144 -ct vbr -np 3 --no-mmproj` starts at 8047 MiB and grows
to 11709 MiB of the 12 GB card at full depth, because the cache trades
quality for space instead of failing. The old fixed-codec limits still apply
if you pin `--cache-type-k/-v` by hand — on this card q4_0 KV maxes at ~185K
with 2 slots or ~180K with 3, and q8_0 KV at ~110K.

## Building

### Native build (recommended for your own machine)

```bash
./build-native.sh              # -> bonsai-27b:native
```

[`Dockerfile.native`](Dockerfile.native) sets **both** knobs to native: the
CPU backend compiles with `-march=native`, and CUDA compiles for exactly your
GPU's compute capability instead of the full multi-arch set. That matters
here: the fork has ~600 CUDA translation units (384 of them TurboQuant
flash-attention instances), so a single-arch build finishes in well under an
hour where a multi-arch build takes several.

The wrapper script exists for one reason: with Docker's default `runc`
runtime the build container has **no GPU**, so CMake cannot resolve
`CMAKE_CUDA_ARCHITECTURES=native` on its own — it reports
`CMAKE_CUDA_ARCHITECTURES_NATIVE=No CUDA devices found` and would silently
build the wrong thing. `build-native.sh` reads the compute capability from
the host driver and passes it in; the Dockerfile fails loudly if you bypass
it. (If your daemon's *default* runtime is `nvidia`, plain
`docker build -f Dockerfile.native .` works too.)

Point the compose file at the resulting tag:

```yaml
image: bonsai-27b:native
```

### Portable / published build

```bash
docker build -t mhald/ternary-bonsai-27b-gguf-llamacpp-cuda:dev .
```

[`Dockerfile`](Dockerfile) is the multi-arch build behind the published
images. Three things differ from `:v3`:

- `LLAMACPP_REPO` now points at **buun-llama-cpp**, pinned via `LLAMACPP_REF`.
  To go back to official llama.cpp, override both args.
- `GGML_CUDA_FA_ALL_QUANTS=ON` is **required, not optional**. Without it the
  CMake glob only compiles the scalar turbo flash-attention instances and
  skips the TCQ ones — exactly the tiers VBR degrades into.
- `-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined` is required because
  the VBR VMM pool calls the CUDA driver API (`cuMemAddressFree` and friends);
  `libcuda.so` only exists at runtime, so the link otherwise fails with
  `undefined reference`.

The container ships its own **CUDA 13.3.1** runtime — the host needs a driver
from the 13.x series. Note that CUDA 13 dropped Maxwell, Pascal and Volta, so
the multi-arch default set now starts at Turing (7.5), unlike the CUDA 12
based `:v3` image.

CI: pushes to `main` publish `:latest`, tags `v*` publish the version tag to
Docker Hub (GitHub Actions, needs `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`
repo secrets). The workflow deliberately does **not** pin
`CUDA_DOCKER_ARCH` — ggml's default set combines real (SASS) targets for the
common GPUs with virtual (PTX) ones for the rest, so architectures without
device code still JIT on first run; a hand-pinned list of `*-real` targets
would publish an image that refuses to start on anything else. With the
fork's ~600 translation units the job takes several hours (v3 took ~1h15m),
which still fits a GitHub runner's 6-hour limit.

## DSpark drafter — not recommended

The HF repo also ships a DSpark speculative-decoding drafter
(`Ternary-Bonsai-27B-dspark-Q4_1.gguf`). It only loads with `:v1`: the file
uses the PrismML fork's own format (GGUF architecture `dspark`, g128
embedding), which official llama.cpp — and therefore `:v2`/`:v3`, as well as
the buun fork behind `:v4` — rejects
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
  — `:latest` (tracks `main`) · `:v4` (buun fork, VBR, g64) · `:v3` (official master, g64) · `:v2` (master + PR #25707, g64) · `:v1` (PrismML fork, g128)
- **This repo:** https://github.com/mahald/ternary-bonsai-27b-gguf-llamacpp-cuda
- **Model:** https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
- **llama.cpp fork used by `:v4` (TurboQuant / VBR):** https://github.com/spiritbuun/buun-llama-cpp
- **TCQ paper (trellis-coded KV cache):** https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache
- **CUDA Q2_0 PR (merged upstream, in `:v2` and newer):** https://github.com/ggml-org/llama.cpp/pull/25707
- **llama.cpp fork (g128 kernels, in `:v1`):** https://github.com/PrismML-Eng/llama.cpp
- **Whitepaper & demos:** https://github.com/PrismML-Eng/Bonsai-demo
