# LLaMA Token-Generation Latency Benchmarking Framework

**CECS 530: Advanced Computer Architecture — Final Project**
**Team 8: Team Hydrazine**
**Authors:** Sahaj Maniya, Vaidik Shah
**Institution:** California State University, Long Beach
**Submission Date:** May 10, 2026

---

## Overview

This repository contains the complete benchmarking framework used in the paper:

> *"Token-Generation Latency Benchmarking in LLaMA: A Comprehensive Performance Evaluation"*

The framework measures four key LLM inference latency metrics across a full factorial sweep of prompt lengths, output lengths, and batch sizes:

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to First Token (ms) |
| **TPS** | Tokens per Second (tok/s) |
| **E2E** | End-to-End latency (ms) |
| **ITL** | Inter-Token Latency (ms) |

**Hardware tested:** Apple M5 MacBook Pro (16 GB unified memory, Metal GPU)
**Model tested:** LLaMA 3.2 3B via Ollama (Q4\_K\_M quantization)

---

## Repository Structure

```
llm-inference-benchmarks/
├── benchmark_v2.py          # Main benchmark driver
├── visualize_results.py     # Figure generation from CSV
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── results/
    └── benchmark_results.csv  # Sample output (included)
```

---

## Dependencies

### System Requirements
- **OS:** macOS 13+ (Apple Silicon) — primary target
- **Python:** 3.9 or higher
- **Ollama:** 0.5.x or higher ([install here](https://ollama.com))

### Python Packages

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `psutil` | 5.9.8 | RAM/CPU monitoring | BSD-3 |
| `matplotlib` | 3.9.0 | Figure generation | PSF |
| `pandas` | 2.2.2 | CSV processing | BSD-3 |
| `requests` | 2.32.3 | HTTP utilities | Apache-2.0 |

All other dependencies (`urllib`, `json`, `csv`, `statistics`, `threading`) are Python standard library — no additional installation required.

> **Third-party credits:** Ollama (MIT License, https://ollama.com), llama.cpp (MIT License, https://github.com/ggerganov/llama.cpp), psutil (BSD-3, https://github.com/giampaolo/psutil)

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/sahajmaniya/llm-inference-benchmarks.git
cd llm-inference-benchmarks
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Install and start Ollama

Download from [https://ollama.com](https://ollama.com), then:

```bash
# Verify Ollama is running
ollama serve
```

### Step 4 — Pull the LLaMA 3.2 model (~2 GB download, one-time)

```bash
ollama pull llama3.2
```

### Step 5 — Verify everything works (quick smoke-test, ~30 seconds)

```bash
python benchmark_v2.py \
    --backend ollama \
    --model llama3.2 \
    --prompt-lengths 64 \
    --output-lengths 64 \
    --batch-sizes 1 \
    --runs 2 --warmup 1 \
    --output results/smoke_test.csv
```

**Expected output:**
```
==============================
LLaMA Latency Benchmark — Team 8: Team Hydrazine
==============================
Backend  : ollama
Model    : llama3.2
Configs  : 1 (1P × 1O × 1B)
Runs     : 2 measured + 1 warmup
==============================

[1/1] P=64 O=64 B=1
  Warming up (1 runs)... done
  Run 1/2... TTFT=93.2ms  TPS=51.6  TPS/item=51.6  E2E=1.25s  ITL=18.34ms
  Run 2/2... TTFT=91.8ms  TPS=53.1  TPS/item=53.1  E2E=1.24s  ITL=18.12ms

[✓] Results saved to: results/smoke_test.csv
```

---

## Running the Full Paper Benchmark

This reproduces the exact 36-configuration factorial sweep from the paper (≈3.5 hours on Apple M5):

```bash
python benchmark_v2.py \
    --backend ollama \
    --model llama3.2 \
    --prompt-lengths 64 128 256 512 \
    --output-lengths 64 128 256 \
    --batch-sizes 1 2 4 \
    --runs 5 --warmup 2 \
    --output results/benchmark_results.csv
```

---

## Generating Figures

After running the benchmark, generate all 6 paper figures:

```bash
python visualize_results.py --input results/benchmark_results.csv
```

Figures are saved to `results/figures/` as high-resolution PNG files.

---

## Command-Line Reference

```
usage: benchmark_v2.py [-h] [--backend {ollama,mock}] [--model MODEL]
                       [--host HOST] [--prompt-lengths N [N ...]]
                       [--output-lengths N [N ...]] [--batch-sizes N [N ...]]
                       [--runs RUNS] [--warmup WARMUP]
                       [--temperature TEMPERATURE] [--output OUTPUT]
                       [--no-save]

Options:
  --backend     ollama (default) or mock (for testing without GPU)
  --model       Ollama model name (default: llama3.2)
  --host        Ollama server URL (default: http://localhost:11434)
  --prompt-lengths  Space-separated list of prompt token lengths
  --output-lengths  Space-separated list of output token lengths
  --batch-sizes     Space-separated list of batch sizes
  --runs        Measured runs per configuration (default: 5)
  --warmup      Warmup runs to discard (default: 2)
  --temperature Sampling temperature, 0.0 = greedy (default: 0.0)
  --output      Output CSV path (default: results/benchmark_results.csv)
  --no-save     Print to stdout only, skip CSV writing
```

---

## Output CSV Format

Each row in the output CSV represents one `(prompt_len, output_len, batch_size)` configuration with the following columns:

| Column | Description |
|--------|-------------|
| `prompt_len` | Prompt length in tokens |
| `output_len` | Output length in tokens |
| `batch_size` | Batch size |
| `n_runs` | Number of successful measured runs |
| `ttft_mean_ms` | Mean TTFT (ms) |
| `ttft_std_ms` | Std dev of TTFT (ms) |
| `ttft_p50_ms` | Median TTFT (ms) |
| `ttft_p95_ms` | 95th percentile TTFT (ms) |
| `ttft_p99_ms` | 99th percentile TTFT (ms) |
| `tps_mean` | Mean tokens/second |
| `tps_std` | Std dev of TPS |
| `tps_p50/p95/p99` | TPS percentiles |
| `e2e_mean_ms` | Mean end-to-end latency (ms) |
| `e2e_std_ms` | Std dev of E2E latency |
| `itl_mean_ms` | Mean inter-token latency (ms) |
| `itl_std_ms` | Std dev of ITL |
| `tps_per_item` | TPS normalized by batch size |
| `peak_ram_mb` | Peak RSS memory (MB), -1 if psutil unavailable |
| `mean_cpu_pct` | Mean CPU utilization (%), -1 if psutil unavailable |

---

## Testing Without GPU (Mock Backend)

For testing the harness without Ollama or Apple Silicon hardware:

```bash
python benchmark_v2.py \
    --backend mock \
    --prompt-lengths 64 128 \
    --output-lengths 64 \
    --batch-sizes 1 2 \
    --runs 3 --warmup 1 \
    --no-save
```

The mock backend returns deterministic synthetic timing data and requires no external dependencies.

---

## Reproducing Paper Results

The `results/benchmark_results.csv` file in this repository contains the original measurements from the paper (180 runs across 36 configurations on Apple M5). To verify our results independently:

1. Install Ollama on any Apple Silicon Mac (M1 or later)
2. Pull `llama3.2` (Q4\_K\_M will be selected automatically)
3. Run the full benchmark command above
4. Compare your CSV with `results/benchmark_results.csv`

> **Note:** Absolute numbers will vary by chip generation (M1/M2/M3/M4/M5 have different memory bandwidth). The key trends — flat TTFT across prompt lengths, stable TPS ceiling, linear E2E batch scaling — should hold across all Apple Silicon generations.

---

## Architecture

```
benchmark_v2.py
├── Backend (Abstract Base Class)
│   ├── OllamaBackend    ← Primary backend (HTTP streaming, Metal GPU)
│   └── MockBackend      ← Deterministic mock for testing
├── SystemMonitor        ← Background thread: RAM + CPU sampling
├── RunResult            ← Single inference call measurements
├── ConfigResult         ← Aggregated statistics for one P×O×B config
├── build_prompt()       ← Synthetic prompt generation
├── run_benchmark()      ← Full factorial sweep engine
├── save_results_csv()   ← CSV writer with relative path handling
└── print_summary()      ← Human-readable results table
```

---

## License

MIT License — see `LICENSE` file for details.

Third-party libraries retain their respective licenses as listed in the Dependencies section above.

---

## Citation

If you use this framework, please cite:

```
Maniya, S. and Shah, V. (2026). Token-Generation Latency Benchmarking
in LLaMA: A Comprehensive Performance Evaluation. CECS 530: Advanced
Computer Architecture, California State University, Long Beach.
GitHub: https://github.com/sahajmaniya/llm-inference-benchmarks
```
