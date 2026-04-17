# TQ3_0 KV Cache — How To Use

TQ3_0 is a 3.5-bit KV cache quantization type based on the TurboQuant pipeline
(Walsh–Hadamard rotation + Lloyd-Max codebook). It compresses K/V cache memory
by ~4.6× vs F16 with near-lossless quality.

This doc covers how to enable it from the CLI on an existing GGUF model.

## Requirements

- Head dimension must be divisible by the TQ3_0 block size (32). Most modern
  models satisfy this (head dim is usually 64/128/256).
- Metal backend: the native TQ3_0 flash-attention kernels are not used; the
  graph automatically dequantizes K/V to F16 before FA and promotes FA to
  enabled when `-ctk tq3_0` is requested. No user action needed.
- CUDA backend: fused MMVQ kernel runs directly on TQ3_0 (no F16 dequant).

## CLI flags

Pass `tq3_0` to `--cache-type-k` / `--cache-type-v`:

```
-ctk tq3_0 -ctv tq3_0
```

Any tool built on `common/` (`llama-completion`, `llama-perplexity`,
`llama-gemma3-cli`, etc.) accepts these flags once the whitelist includes
TQ3_0 (see `common/arg.cpp`).

## Example — Qwen3.5 2B

Model: `Qwen3.5-2B-Q4_K_M.gguf` (head dim 256, 24 layers, 6 of which use the
full-attention KV cache path).

```bash
BIN=build-macos/bin/Release/llama-completion
MODEL="$HOME/Library/Application Support/ZaatiAI/Models/gguf/Qwen3.5-2B-Q4_K_M.gguf"

"$BIN" -m "$MODEL" \
  -c 4096 -n 32 \
  -ctk tq3_0 -ctv tq3_0 \
  -ngl 99 --temp 0 \
  -p "The capital of France is"
```

Expected log lines confirming TQ3_0 is live:

```
llama_init_from_model: TQ3_0 K cache - FA enabled with graph-side dequant to F16
llama_context: flash_attn    = enabled
llama_kv_cache:       MTL0 KV buffer size =    10.50 MiB
llama_kv_cache: size = 10.50 MiB ( 4096 cells, 6 layers, 1/1 seqs),
                K (tq3_0): 5.25 MiB, V (tq3_0): 5.25 MiB
```

## Measured memory — Qwen3.5 2B, ctx = 4096

Head-to-head on Apple M1 Pro, same prompt, greedy sampling, `-ngl 99`:

| `-ctk`/`-ctv` | Total KV | K or V | Reduction vs F16 | Quality |
|---------------|---------:|-------:|-----------------:|---------|
| `f16`         | 48.00 MiB | 24.00 MiB | 1.00× | reference |
| `q8_0`        | 25.50 MiB | 12.75 MiB | 1.88× | matches F16 |
| `q4_0`        | 13.50 MiB |  6.75 MiB | 3.56× | slight drift |
| **`tq3_0`**   | **10.50 MiB** |  **5.25 MiB** | **4.57×** | matches F16 |

Savings scale linearly with context: at 32K ctx, TQ3_0 saves ~300 MiB of KV
memory vs F16 on this model.

Bits-per-value check (K only):
`5.25 MiB / (4096 cells × 6 layers × 256 dims) × 8 bits = 3.5 bpv` ✓

## Limitations

- `llama-bench` has its own `ggml_type_from_name` allowlist and does not yet
  recognise `tq3_0`. Use `llama-completion` / `llama-perplexity` for benchmarking.
- Native Metal TQ3_0 flash-attention kernels still produce incorrect outputs;
  the graph-side F16 dequant path is used instead. This costs a small amount
  of extra memory traffic at attention time but preserves correctness.
- Head dims not divisible by 32 are rejected at context creation with an
  explicit error.
