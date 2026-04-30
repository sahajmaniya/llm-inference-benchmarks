"""
╔══════════════════════════════════════════════════════════════════════╗
║        LLaMA Token-Generation Latency Benchmark                      ║
║        Optimized for Apple Silicon (M1/M2/M3/M4/M5)                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  QUICK START (3 steps):                                              ║
║  1. pip install -r requirements.txt                                  ║
║  2. ollama pull llama3.2        (one-time model download)            ║
║  3. python benchmark.py                                              ║
║                                                                      ║
║  Demo (no model needed):                                             ║
║     python benchmark.py --mock                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import json
import statistics
import argparse
import csv
import os
import random
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import List
from datetime import datetime


# ══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    backend: str
    model: str
    prompt_length: int
    output_length: int
    batch_size: int
    ttft_ms: float          # Time to First Token (ms)
    tps: float              # Tokens per Second
    e2e_latency_ms: float   # End-to-End latency (ms)
    itl_ms: float           # Inter-Token Latency (ms)
    actual_tokens: int
    run_index: int


@dataclass
class AggregatedResult:
    backend: str
    model: str
    prompt_length: int
    output_length: int
    batch_size: int
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tps_mean: float
    tps_p50: float
    tps_p95: float
    e2e_mean_ms: float
    e2e_p95_ms: float
    itl_mean_ms: float
    num_runs: int


# ══════════════════════════════════════════════════════════════
#  BACKENDS
# ══════════════════════════════════════════════════════════════

class MockBackend:
    """
    Simulates realistic LLaMA latency on Apple Silicon — no model needed.
    Great for testing the benchmark pipeline end-to-end.
    """
    def load(self, model: str, device: str):
        print("  [Mock] Simulating Apple Silicon M-series performance ✓")

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float):
        results = []
        for _ in prompts:
            ttft = random.gauss(120, 20)
            tps_base = random.gauss(45, 5)
            token_timestamps = []
            t = time.perf_counter()
            t += ttft / 1000
            for _ in range(max_new_tokens):
                t += (1 / tps_base) + random.gauss(0, 0.002)
                token_timestamps.append(t)
            results.append({
                "ttft_ms": max(ttft, 10),
                "tokens": list(range(max_new_tokens)),
                "token_timestamps": token_timestamps
            })
        return results


class OllamaBackend:
    """
    Uses Ollama — easiest setup on Mac. Automatically uses Metal GPU.
    Install: https://ollama.com  then run: ollama pull llama3.2
    """
    def __init__(self):
        self.host = "http://localhost:11434"
        self.model_name = "llama3.2"

    def load(self, model: str, device: str):
        self.model_name = model
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=3)
            print(f"  [Ollama] Connected ✓  model={model}  (Metal GPU auto-enabled)")
        except Exception:
            print("\n  ✗ Ollama not running!")
            print("  Fix: open a new terminal and run:  ollama serve")
            print("  Then make sure you ran:            ollama pull", model)
            sys.exit(1)

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float):
        import urllib.request
        results = []
        for prompt in prompts:
            payload = json.dumps({
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_new_tokens,
                    "num_gpu": 999
                }
            }).encode()

            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            token_timestamps = []
            generated_tokens = []
            first_token = True
            ttft = 0.0
            gen_start = time.perf_counter()

            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    data = json.loads(line.decode())
                    tok = data.get("response", "")
                    if tok:
                        ts = time.perf_counter()
                        if first_token:
                            ttft = (ts - gen_start) * 1000
                            first_token = False
                        token_timestamps.append(ts)
                        generated_tokens.append(tok)
                    if data.get("done"):
                        break

            results.append({
                "ttft_ms": ttft,
                "tokens": generated_tokens,
                "token_timestamps": token_timestamps
            })
        return results


class HuggingFaceBackend:
    """
    Uses HuggingFace Transformers with MPS (Apple Metal GPU).
    pip install transformers torch
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self, model: str, device: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            print("  ✗ Run:  pip install transformers torch")
            sys.exit(1)

        import torch
        if torch.backends.mps.is_available():
            self.device = "mps"
            print(f"  [HF] Using Apple Metal (MPS) GPU ✓")
        else:
            self.device = "cpu"
            print(f"  [HF] Using CPU")

        print(f"  [HF] Loading {model} — this may take a minute...")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  [HF] Model loaded ✓")

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float):
        import torch
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            token_timestamps = []
            generated_ids = []
            past_kv = None
            input_ids = inputs["input_ids"]
            first_token = True
            ttft = 0.0
            gen_start = time.perf_counter()

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
                logits = out.logits[:, -1, :]
                if temperature == 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, 1)
                ts = time.perf_counter()
                if first_token:
                    ttft = (ts - gen_start) * 1000
                    first_token = False
                token_timestamps.append(ts)
                generated_ids.append(next_id.item())
                past_kv = out.past_key_values
                input_ids = next_id
                if next_id.item() == self.tokenizer.eos_token_id:
                    break

            results.append({
                "ttft_ms": ttft,
                "tokens": generated_ids,
                "token_timestamps": token_timestamps
            })
        return results


class LlamaCppBackend:
    """
    Uses llama.cpp with Metal GPU — fastest option on Mac.
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
    """
    def __init__(self):
        self.llm = None

    def load(self, model: str, device: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            print("  ✗ Run:")
            print("    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal")
            sys.exit(1)
        print(f"  [llama.cpp] Loading {model} with Metal GPU...")
        self.llm = Llama(model_path=model, n_gpu_layers=-1, verbose=False)
        print("  [llama.cpp] Model loaded ✓ (full Metal GPU offload)")

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float):
        results = []
        for prompt in prompts:
            token_timestamps = []
            generated_tokens = []
            first_token = True
            ttft = 0.0
            gen_start = time.perf_counter()
            stream = self.llm(prompt, max_tokens=max_new_tokens,
                              temperature=max(temperature, 1e-8), stream=True)
            for chunk in stream:
                ts = time.perf_counter()
                tok = chunk["choices"][0].get("text", "")
                if tok:
                    if first_token:
                        ttft = (ts - gen_start) * 1000
                        first_token = False
                    token_timestamps.append(ts)
                    generated_tokens.append(tok)
            results.append({
                "ttft_ms": ttft,
                "tokens": generated_tokens,
                "token_timestamps": token_timestamps
            })
        return results


BACKENDS = {
    "mock":     MockBackend,
    "ollama":   OllamaBackend,
    "hf":       HuggingFaceBackend,
    "llamacpp": LlamaCppBackend,
}


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def build_prompt(target_tokens: int) -> str:
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Artificial intelligence and large language models are transforming "
        "how we benchmark inference latency and throughput. "
        "Apple Silicon chips use unified memory architecture for fast on-device AI. "
    )
    repeats = math.ceil((target_tokens * 4) / len(base)) + 1
    return (base * repeats)[: target_tokens * 4]


def compute_itl(timestamps: List[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    gaps = [(timestamps[i] - timestamps[i-1]) * 1000 for i in range(1, len(timestamps))]
    return statistics.mean(gaps)


def percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, int(len(s) * p / 100) - 1)
    return s[idx]


def print_table(results: List[AggregatedResult]):
    print("\n" + "═" * 100)
    print(f"  {'BACKEND':<10} {'MODEL':<16} {'PROMPT':>7} {'OUTPUT':>7} {'BATCH':>6} "
          f"{'TTFT(ms)':>10} {'TPS':>8} {'E2E(ms)':>10} {'ITL(ms)':>10}")
    print("  " + "─" * 96)
    for r in results:
        model_short = r.model.split("/")[-1][:15]
        print(f"  {r.backend:<10} {model_short:<16} {r.prompt_length:>7} {r.output_length:>7} "
              f"{r.batch_size:>6} {r.ttft_mean_ms:>10.1f} {r.tps_mean:>8.1f} "
              f"{r.e2e_mean_ms:>10.0f} {r.itl_mean_ms:>10.2f}")
    print("═" * 100)


# ══════════════════════════════════════════════════════════════
#  CORE BENCHMARK
# ══════════════════════════════════════════════════════════════

def run_benchmark(args) -> List[AggregatedResult]:
    backend = BACKENDS[args.backend]()
    backend.load(args.model, args.device)

    raw: List[RunResult] = []
    configs = [
        (pl, ol, bs)
        for pl in args.prompt_lengths
        for ol in args.output_lengths
        for bs in args.batch_sizes
    ]
    total = len(configs) * args.num_runs
    done = 0

    for prompt_len, output_len, batch_size in configs:
        prompts = [build_prompt(prompt_len) for _ in range(batch_size)]
        label = f"prompt={prompt_len:>4}t  output={output_len:>4}t  batch={batch_size}"

        print(f"\n  ┌─ {label}")
        print(f"  │  warmup ({args.warmup} runs)...", end="", flush=True)
        for _ in range(args.warmup):
            backend.generate(prompts, output_len, args.temperature)
        print(" done")

        for run_i in range(args.num_runs):
            t0 = time.perf_counter()
            outputs = backend.generate(prompts, output_len, args.temperature)
            t1 = time.perf_counter()

            ttft_avg     = statistics.mean(o["ttft_ms"] for o in outputs)
            itl_avg      = statistics.mean(compute_itl(o["token_timestamps"]) for o in outputs)
            total_tokens = sum(len(o["tokens"]) for o in outputs)
            e2e_ms       = (t1 - t0) * 1000
            tps          = total_tokens / (e2e_ms / 1000) if e2e_ms > 0 else 0

            raw.append(RunResult(
                backend=args.backend, model=args.model,
                prompt_length=prompt_len, output_length=output_len, batch_size=batch_size,
                ttft_ms=ttft_avg, tps=tps, e2e_latency_ms=e2e_ms,
                itl_ms=itl_avg, actual_tokens=total_tokens, run_index=run_i
            ))
            done += 1
            pct = done / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  │  [{bar}] {pct:5.1f}%  run {run_i+1}/{args.num_runs}  "
                  f"TTFT={ttft_avg:6.0f}ms  TPS={tps:5.1f}  E2E={e2e_ms:6.0f}ms")

    return aggregate(raw)


def aggregate(raw: List[RunResult]) -> List[AggregatedResult]:
    from itertools import groupby
    key = lambda r: (r.backend, r.model, r.prompt_length, r.output_length, r.batch_size)
    aggregated = []
    for gk, grp in groupby(sorted(raw, key=key), key=key):
        runs = list(grp)
        ttft = [r.ttft_ms for r in runs]
        tps  = [r.tps for r in runs]
        e2e  = [r.e2e_latency_ms for r in runs]
        itl  = [r.itl_ms for r in runs]
        aggregated.append(AggregatedResult(
            backend=gk[0], model=gk[1],
            prompt_length=gk[2], output_length=gk[3], batch_size=gk[4],
            ttft_mean_ms=statistics.mean(ttft), ttft_p50_ms=percentile(ttft, 50),
            ttft_p95_ms=percentile(ttft, 95),   ttft_p99_ms=percentile(ttft, 99),
            tps_mean=statistics.mean(tps),      tps_p50=percentile(tps, 50),
            tps_p95=percentile(tps, 95),
            e2e_mean_ms=statistics.mean(e2e),   e2e_p95_ms=percentile(e2e, 95),
            itl_mean_ms=statistics.mean(itl),   num_runs=len(runs)
        ))
    return aggregated


def save_results(results: List[AggregatedResult], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(out_dir, f"benchmark_{ts}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_path = os.path.join(out_dir, f"benchmark_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        w.writeheader()
        w.writerows(asdict(r) for r in results)

    print(f"\n  💾  Saved → {json_path}")
    print(f"  💾  Saved → {csv_path}")
    return json_path, csv_path


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="LLaMA Token-Generation Latency Benchmark — Apple Silicon Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --mock
  python benchmark.py --backend ollama --model llama3.2
  python benchmark.py --backend ollama --model llama3.2 --prompt-lengths 64 256 512 --num-runs 10
  python benchmark.py --backend hf --model meta-llama/Llama-3.2-1B
  python benchmark.py --backend llamacpp --model ./models/llama-3.2-1b.gguf
        """
    )
    p.add_argument("--mock",            action="store_true",  help="Run demo simulation (no model needed)")
    p.add_argument("--backend",         default="ollama",     choices=["ollama", "hf", "llamacpp", "mock"])
    p.add_argument("--model",           default="llama3.2",   help="Model name or path")
    p.add_argument("--prompt-lengths",  nargs="+", type=int,  default=[64, 128, 256, 512])
    p.add_argument("--output-lengths",  nargs="+", type=int,  default=[64, 128, 256])
    p.add_argument("--batch-sizes",     nargs="+", type=int,  default=[1, 2, 4])
    p.add_argument("--num-runs",        type=int,  default=5, help="Measured runs per config")
    p.add_argument("--warmup",          type=int,  default=2, help="Warmup runs (discarded)")
    p.add_argument("--temperature",     type=float,default=0.0)
    p.add_argument("--device",          default="auto",       help="auto | cpu | mps")
    p.add_argument("--out-dir",         default="results",    help="Output folder for JSON/CSV")
    args = p.parse_args()

    if args.mock:
        args.backend = "mock"
        args.model   = "mock-llama-m5"

    total_configs = len(args.prompt_lengths) * len(args.output_lengths) * len(args.batch_sizes)

    print("""
╔══════════════════════════════════════════════════════╗
║  LLaMA Latency Benchmark  •  Apple Silicon Edition   ║
╚══════════════════════════════════════════════════════╝""")
    print(f"  Backend  : {args.backend}")
    print(f"  Model    : {args.model}")
    print(f"  Prompts  : {args.prompt_lengths} tokens")
    print(f"  Outputs  : {args.output_lengths} tokens")
    print(f"  Batches  : {args.batch_sizes}")
    print(f"  Runs     : {args.num_runs} measured + {args.warmup} warmup")
    print(f"  Configs  : {total_configs} combinations\n")

    results = run_benchmark(args)
    print_table(results)
    save_results(results, args.out_dir)
    print("\n  ✓ Done! Check the 'results/' folder for JSON and CSV output.\n")


if __name__ == "__main__":
    main()
