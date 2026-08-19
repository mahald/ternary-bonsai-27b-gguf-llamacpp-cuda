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
12 GB GPU (RTX 4080 Laptop). Comments are stripped here for readability — the
file itself carries them inline:

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
      --presence-penalty 0.0
      --repeat-penalty 1.0
      -lv 3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

> ### ⚠️ `--presence-penalty 0.0` assumes a loop breaker in your client
>
> Bonsai is a thinking model, and thinking models loop. A sampler penalty is
> the wrong layer to catch that: it sees token probabilities and nothing else,
> so it cannot tell legitimate repetition during reasoning from a real loop.
> (Same blind spot made the DRY sampler hallucinate with this model.) A guard
> at the agent layer judges observed behaviour instead — a stream cycling on
> itself, the same tool call over and over — which makes the sampler penalty
> redundant and lets `0.0` buy maximum code fidelity.
>
> This setup pairs the server with [omp](https://omp.sh), which ships both
> guards on by default and needs no extension for them.
>
> **If nothing in your client watches for loops, set `--presence-penalty` back
> to `0.3`.** `0.0` buys code fidelity, not loop immunity. Details:
> [Loop breaking](#loop-breaking-also-built-in).

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
  concurrent requests (e.g. parallel omp subagents). Dynamic VBR forces
  `--kv-unified` anyway as soon as `-np > 1`; it is spelled out here so the
  config still reads correctly if you ever drop `-ct vbr`.
- **No `-fa on`**: flash attention defaults to `auto`, which already enables
  it wherever it is supported. With turbo-typed KV the fork force-enables it
  regardless, so the flag would be redundant twice over.
- **`-lv 3`** raises the log verbosity so `docker logs` shows the detail
  needed to verify the setup — buffer placement, offloaded layer count, CUDA
  device table (see
  [Troubleshooting](#troubleshooting-slow-inference-all-cpu-cores-at-100-)).
- **No `--cache-ram`**: dynamic VBR disables the host-RAM prompt cache
  outright (`prompt cache state storage is not supported by dynamic VBR
  (KV tiers change at runtime), it will be disabled`), so the flag `:v3`
  carried would be a no-op. See [What VBR costs you](#what-vbr-costs-you).
- **Sampling defaults** are set server-side (`--temp 0.7 --top-p 0.95
  --top-k 20 --min-p 0.1 --xtc-probability 0.8 --xtc-threshold 0.2
  --presence-penalty 0.0 --repeat-penalty 1.0`); they apply to any request
  that doesn't send its own values. `temp 0.7`, `top-p 0.95` and `top-k 20`
  are the model card's own eval values. On top of those, `min-p 0.1` trims the
  low-probability tail, and XTC (exclude-top-choices, probability 0.8 /
  threshold 0.2) prevents the sampler from reflexively locking onto the
  highest-probability token when two candidates are close — a known quality
  boost for thinking models. Don't lower the temperature much — near-greedy
  decoding makes reasoning models loop.
- **`--presence-penalty 0.0`** assumes the client breaks loops. A sampler
  penalty is the wrong layer for that job: it sees token probabilities and
  nothing else, so it cannot distinguish legitimate repetition during reasoning
  from a real loop — the same blind spot that made the DRY sampler hallucinate
  with this model. An agent-level guard judges observed behaviour instead, and
  omp's [built-in loop guards](#loop-breaking-also-built-in) are what this
  setup relies on. **Set it back to 0.3 if nothing in your client watches for
  loops** — 0.0 buys code fidelity, not loop immunity.
- **Repetition loops?** Stick to the neutral sampling defaults first. In our
  testing, enabling the DRY sampler (`--dry-multiplier 0.8`) made this model
  hallucinate — thinking models legitimately repeat phrases while reasoning,
  and DRY forces them off the correct path. If loops persist, put
  `--presence-penalty 0.3` back (see above) or, better, let the client break
  loops on observed behaviour.
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
- **The host-RAM prompt cache goes with them.** The server says so on
  startup: `prompt cache state storage is not supported by dynamic VBR (KV
  tiers change at runtime), it will be disabled`. In-slot prefix reuse still
  works — a follow-up turn in the same slot does not re-prefill — but a
  conversation evicted from its slot is gone and has to be prefilled from
  scratch. Given that filling the window takes ~15 minutes, this is the
  sharpest edge of the trade: **you get the depth, but you pay full price for
  every cold session.** If your workload is many short-lived sessions rather
  than a few long ones, `:v3` with its working prompt cache may serve you
  better.
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
  they wait for a free slot). An omp fan-out wider than the slot count
  therefore completes in waves; cap it with `task.maxConcurrency` and
  `providers.maxInFlightRequests` (see
  [Matching the server's 3 slots](#matching-the-servers-3-slots)).
- **KV pool contention**: with all 3 slots active, the shared 262K pool
  divides ~87K per slot; typical subagent contexts fit easily. Note that under
  VBR the slots also share the *quality* budget — three deep sessions push the
  cache down the tier ladder faster than one does, because the degrade
  controller sees the pool's total mapped bytes, not any single slot's.

## Using with omp (oh my pi)

[omp](https://omp.sh) ([`can1357/oh-my-pi`](https://github.com/can1357/oh-my-pi))
is the client this setup is developed and tested with. Add the server as a
custom provider in `~/.omp/agent/models.yml`:

```yaml
providers:
  bonsai-local:
    baseUrl: http://127.0.0.1:8080/v1
    api: openai-completions
    apiKey: none
    models:
      - id: bonsai-27b
        name: Ternary Bonsai 27B (local)
        reasoning: true
        input: [text]
        contextWindow: 262144
        maxTokens: 16384
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
```

and make it the default model role in `~/.omp/agent/config.yml`:

```yaml
modelRoles:
  default: bonsai-local/bonsai-27b:medium
```

Keep `contextWindow` in sync with the server's `-c` value. `apiKey: none`
satisfies omp's key requirement for a keyless endpoint; `auth: none` does the
same thing. omp can also probe the server instead of being told about it —
`discovery: { type: llama.cpp }` on the provider — but an explicit model entry
is what pins `contextWindow` and `maxTokens` to the values this compose file
serves.

### Reasoning effort: built in, no extension needed

The `:medium` in the model role above *is* the reasoning-effort setting. omp
carries thinking-level control in the core, on four scopes:

| Scope | How |
|---|---|
| One model role | the `:level` suffix — `bonsai-local/bonsai-27b:medium` |
| Global default | `defaultThinkingLevel` in `config.yml` (`omp config set defaultThinkingLevel medium`) |
| One run | `--thinking <level>` |
| One turn | the `ultrathink` magic keyword, which raises that turn to the model's top supported effort |

Levels are `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, plus `auto`.

#### What the level actually does here: nothing

Worth knowing before you tune it. On an `openai-completions` provider omp puts
the level on the wire as `reasoning_effort` and nothing else — no token
budget, and `max_completion_tokens` stays at the model's `maxTokens` at every
level. Captured from omp through a logging proxy:

```json
{"model": "bonsai-27b", "reasoning_effort": "medium",
 "max_completion_tokens": 16384, "stream": true, "...": "..."}
```

llama-server then drops it. Measured against this server with deterministic
sampling (`temperature 0`, `top_k 1`, `xtc_probability 0`, fixed `seed`), the
same prompt at `reasoning_effort` `low`, `high`, and absent returns
**byte-identical** reasoning — same length, same SHA-1 — and `/apply-template`
renders the same prompt for all three. Bonsai's chat template has no effort
branch; the only reasoning switch in it is `enable_thinking`.

So `:medium` above is a label on the omp side, not a knob on the model. What
does reach the model:

| Want | How |
|---|---|
| No reasoning at all | `chat_template_kwargs: {"enable_thinking": false}` — measured: reasoning drops to zero, the answer stays correct |
| A ceiling on reasoning + answer together | `max_tokens` / `maxTokens`; reasoning tokens count against it |

Keep an explicit level anyway rather than `auto`: `defaultThinkingLevel: auto`
classifies each turn's difficulty with a *separate* model,
`providers.autoThinkingModel`, which defaults to `online` — an outbound call
this setup otherwise never makes.

### Loop breaking: also built in

omp ships two loop guards, both on by default, and they are the reason the
compose file can run `--presence-penalty 0.0`:

```yaml
model:
  loopGuard:
    enabled: true            # degenerate reasoning loops in the stream
    checkAssistantContent: true
  toolCallLoopGuard:
    enabled: true            # the same tool call, over and over
    threshold: 5
```

What each one actually does for *this* model is worth being precise about:

- **`model.loopGuard`** watches streamed deltas. Its exact suffix-cycle
  detector — a byte-identical unit repeated back-to-back — applies to every
  model, `bonsai-27b` included. On a hit the turn is terminated as a retryable
  stream stall and re-sampled, up to three attempts. Its *semantic* detectors
  (near-duplicate paragraphs by word-trigram fingerprint, and the
  "progress-lexicon stall" where filler gets reshuffled without new vocabulary)
  are gated to the Gemini, DeepSeek and Grok model families, so they do **not**
  fire for a custom provider id like `bonsai-27b`.
- **`model.toolCallLoopGuard`** is model-agnostic. It canonicalizes tool calls
  across turns and, once the same call repeats consecutively `threshold` (5)
  times, injects a corrective steer summarizing what the agent keeps doing and
  what it keeps getting back.

That is a behavioural guard — it judges what the agent did, not what the token
distribution looked like — which is the layer a sampler penalty cannot reach. A
sampler penalty sees token probabilities and nothing else, so it cannot tell
legitimate repetition during reasoning from a real loop; that is the same blind
spot that made the DRY sampler hallucinate with this model. Note this is a
rationale for the pairing, not a measured result: no loop rate was benchmarked
with the guards on and off.

**If your client has no loop breaker, set `--presence-penalty` back to `0.3`.**
`0.0` buys code fidelity, not loop immunity.

### Sampling stays server-side

omp exposes `temperature`, `topP`, `topK`, `minP`, `presencePenalty` and
`repetitionPenalty`, all defaulting to `-1`, which means "use the
provider/model default — do not send the parameter". Leave them at `-1` and
the sampling defaults pinned in the compose file apply. Set any of them and it
overrides the server for every request from that client.

### Matching the server's 3 slots

Subagents are built into omp (the `task` tool); there is no extension to
install. Two settings keep a fan-out from fighting the server for slots:

```yaml
providers:
  maxInFlightRequests:
    bonsai-local: 3
task:
  maxConcurrency: 2
```

- **`providers.maxInFlightRequests`** caps concurrent LLM HTTP requests per
  provider, and the cap is shared across every local `omp` process using the
  same config root. Pinning it to the server's `-np 3` means omp never has
  requests queued *inside* llama-server where it cannot see or cancel them.
- **`task.maxConcurrency`** (default 32 — far above what one GPU can serve)
  bounds concurrent subagents. 2 leaves the parent session a slot of its own.
  Fan-outs larger than the cap complete in waves; see
  [Parallelism expectations](#parallelism-expectations-one-12-gb-gpu).
- **No subagent timeout to work around.** `task.maxRuntimeMs` defaults to `0`,
  meaning unlimited, so slow prefill on a 12 GB card needs no config change.
  Set it only if you want a hard wall-clock ceiling.
- Idle subagent sessions are parked after `task.agentIdleTtlMs` (7 minutes),
  and each subagent gets a soft budget of `task.softRequestBudget` (200)
  provider requests before it is nudged to wrap up.

Custom subagents are markdown files in `~/.omp/agent/agents/*.md` (user scope)
or `.omp/agents/*.md` (project scope). The frontmatter takes `name`,
`description`, `model`, `thinking-level`, `spawns`, `blocking`, `advisor` and
`autoloadSkills`, so a subagent can pin its own model — a cloud one, say —
while the parent session stays on Bonsai.

### Plugins in use

omp covers reasoning effort, loop breaking, subagents, memory and LSP natively,
so the plugin list is short. For the record, what is installed on the machine
these measurements were taken on — none of it is specific to Bonsai:

| Plugin | What it does |
|---|---|
| [`pi-lens`](https://pi.dev/packages/pi-lens) | Real-time code feedback: linters, formatters, type-checking and structural analysis on top of the LSP layer |
| [`pi-agent-browser-native`](https://pi.dev/packages/pi-agent-browser-native) | Exposes agent-browser as a native tool for browser automation |
| [`@gwynnnplaine/pi-github`](https://pi.dev/packages/@gwynnnplaine/pi-github) | GitHub issues, PRs, diffs, checks and code via the `gh` CLI |
| [`@piex-dev/dap`](https://pi.dev/packages/@piex-dev/dap) | Debug Adapter Protocol — debug a program from inside the agent |
| [`@vtstech/pi-model-test`](https://pi.dev/packages/@vtstech/pi-model-test) | Model benchmark/testing extension — useful for checking what a local endpoint actually supports |
| [`@juicesharp/rpiv-voice`](https://pi.dev/packages/@juicesharp/rpiv-voice) | Voice dictation via `/voice`, on-device STT with sherpa-onnx Whisper |

A deterministic check is worth more here than on a fast cloud model: at
~45 tok/s a "write, run the type checker, read the error, fix" round-trip costs
minutes of decode, while a compiler-grade diagnostic delivered inline at edit
time costs nothing. That is the whole argument for `pi-lens` in a local-model
setup — the cheaper your tokens, the less it matters. These are npm packages
shared with the `pi` ecosystem; omp installs them into `~/.omp/plugins`.

## Configuration examples

The [`examples/omp/`](examples/omp/) directory mirrors the config used and
tested with this repo — copy the parts you need into `~/.omp/agent/`:

| Example file | Destination | Purpose |
|---|---|---|
| `examples/omp/models.yml.example` | `~/.omp/agent/models.yml` | the `bonsai-local` provider on its own |
| `examples/omp/models.cloud.yml.example` | — | `bonsai-local` **plus** a cloud provider with a `<YOUR_API_KEY>` placeholder; **never commit real keys** |
| `examples/omp/config.yml.example` | `~/.omp/agent/config.yml` | default role, effort, loop guards, and the slot-matching caps above |

```bash
cp examples/omp/models.yml.example  ~/.omp/agent/models.yml
cp examples/omp/config.yml.example  ~/.omp/agent/config.yml
```

`config.yml` is where omp writes your own settings, so merge rather than
overwrite if you already have one — every key in the example can also be set
with `omp config set <key> <value>`.

> **Security:** the examples contain no secrets. Your real `~/.omp/agent/`
> holds credentials in `agent.db` and may hold API keys in `models.yml` — never
> copy or commit those.

## Troubleshooting: slow inference, all CPU cores at 100 %

> **Less of a trap on `:v4`.** VBR needs its KV tensors on a device that
> implements the turbo codecs, so a configuration that would park the KV cache
> on the CPU now fails at startup with a clear message instead of quietly
> running slowly:
>
> ```
> failed to initialize the context: dynamic VBR (-ctk vbr) requires per-device
> KV buffers with turbo/VBR backend support, but the KV buffer type is CPU (or
> a device underneath it) without that support — offload the KV cache to
> supported GPUs or use a static KV type (f16/q8_0).
> ```
>
> Worth knowing: VBR is active **even without `-ct vbr`** (implicitly, with a
> turbo4 floor), so this applies to any `:v4` run that does not pin a static
> cache type. If you deliberately want CPU or partial-CPU inference, pass
> `-ctk f16 -ctv f16` — and then the silent-spill problem below is yours again.

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
images. Three things about it are easy to get wrong:

- `LLAMACPP_REPO` points at **buun-llama-cpp**, pinned via `LLAMACPP_REF`. To
  build against official llama.cpp instead, override both args.
- `GGML_CUDA_FA_ALL_QUANTS=ON` is **required, not optional**. Without it the
  CMake glob only compiles the scalar turbo flash-attention instances and
  skips the TCQ ones — exactly the tiers VBR degrades into.
- `-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined` is required because
  the VBR VMM pool calls the CUDA driver API (`cuMemAddressFree` and friends);
  `libcuda.so` only exists at runtime, so the link otherwise fails with
  `undefined reference`.

The container ships its own **CUDA 13.3.1** runtime — the host needs a driver
from the 13.x series. CUDA 13 dropped Maxwell, Pascal and Volta, so the
multi-arch set starts at Turing (7.5).

CI: pushes to `main` publish `:latest`, tags `v*` publish the version tag to
Docker Hub (GitHub Actions, needs `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`
repo secrets).

Since v4 the workflow pins
`CUDA_DOCKER_ARCH=75-virtual;86-real;89-real;120-real`:

| Entry | Covers |
|---|---|
| `75-virtual` | PTX floor — every GPU from Turing up can JIT |
| `86-real` | Ampere consumer (RTX 30xx) |
| `89-real` | Ada (RTX 40xx) — what this setup is tuned on |
| `120-real` | Blackwell consumer (RTX 50xx) |

Three things about that list are worth knowing before you change it:

- **It has to be pinned.** With ggml's default set the job hits the runner's
  hard 6-hour limit and gets killed mid-build — measured, twice. The fork has
  ~600 CUDA translation units against v3's ~200, and the TurboQuant
  flash-attention templates are far more expensive than average, so v3's
  ~1h15m does not extrapolate. One pass costs ~1h12m here (3 passes = 3h37m),
  so four fit with about an hour to spare.
- **It has to keep a virtual target.** PTX is forward-compatible, so the
  single `75-virtual` entry lets every GPU from Turing upwards JIT on first
  run. A list of only `*-real` targets would build faster still and publish an
  image that refuses to start on anything not explicitly compiled.
- **An architecture missing from this list is untested, not merely slower.**
  ggml selects kernel variants host-side through
  `ggml_cuda_highest_compiled_arch()`, which for an uncompiled compute
  capability returns the highest arch *below* it — 890 on a Blackwell card,
  say. The only device code actually present there is the JIT from the Turing
  PTX, so the host's choice and the available device code can disagree, and
  `fattn-mma-f16.cuh` aborts through `NO_DEVICE_CODE` instead of degrading
  gracefully.

Datacenter parts are the gap: A100 (`80`) and Hopper (`90`) are not compiled
in, because two more passes would exceed the time limit. If you run those,
build the image yourself — `./build-native.sh` compiles for exactly the card
it finds.

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
- **omp (oh my pi), the client used here:** https://omp.sh · https://github.com/can1357/oh-my-pi
