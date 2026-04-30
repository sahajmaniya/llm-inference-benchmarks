"""
╔══════════════════════════════════════════════════════════════════════╗
║   LLaMA Token-Generation Latency Benchmark  v2.0                    ║
║   Apple Silicon Edition (M1/M2/M3/M4/M5)                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  NEW in v2.0:                                                        ║
║   ✦ RAM usage tracking (peak MB during inference)                   ║
║   ✦ CPU utilization % per run                                       ║
║   ✦ Model load time measurement                                     ║
║   ✦ Std deviation on all metrics                                    ║
║   ✦ Normalized TPS per batch item                                   ║
║   ✦ Multi-model comparison (--models flag)                          ║
║   ✦ Rich summary report printed at end                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  QUICK START:                                                        ║
║    python benchmark_v2.py --mock                                    ║
║    python benchmark_v2.py --models llama3.2 llama3.2:1b             ║
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
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False
    print("  ⚠  psutil not found — RAM/CPU tracking disabled. Run: pip install psutil")


# ══════════════════════════════════════════════════════════════
#  SYSTEM MONITOR  (background thread)
# ══════════════════════════════════════════════════════════════

class SystemMonitor:
    """Samples RAM and CPU usage in a background thread during a run."""
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._ram_samples: List[float] = []
        self._cpu_samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid()) if PSUTIL_OK else None

    def start(self):
        self._ram_samples.clear()
        self._cpu_samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if not self._ram_samples:
            return {"peak_ram_mb": 0.0, "mean_cpu_pct": 0.0}
        return {
            "peak_ram_mb": max(self._ram_samples),
            "mean_cpu_pct": statistics.mean(self._cpu_samples) if self._cpu_samples else 0.0
        }

    def _sample(self):
        if not PSUTIL_OK:
            return
        while self._running:
            try:
                mem = self._proc.memory_info().rss / (1024 * 1024)
                cpu = psutil.cpu_percent(interval=None)
                self._ram_samples.append(mem)
                self._cpu_samples.append(cpu)
            except Exception:
                pass
            time.sleep(self.interval)


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
    ttft_ms: float
    tps: float
    tps_per_item: float          # TPS normalized per batch item
    e2e_latency_ms: float
    itl_ms: float
    actual_tokens: int
    peak_ram_mb: float           # NEW: peak RAM during run
    mean_cpu_pct: float          # NEW: mean CPU % during run
    run_index: int


@dataclass
class AggregatedResult:
    backend: str
    model: str
    prompt_length: int
    output_length: int
    batch_size: int
    # TTFT
    ttft_mean_ms: float
    ttft_std_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    # TPS
    tps_mean: float
    tps_std: float
    tps_p50: float
    tps_p95: float
    tps_per_item_mean: float     # normalized
    # E2E
    e2e_mean_ms: float
    e2e_std_ms: float
    e2e_p95_ms: float
    # ITL
    itl_mean_ms: float
    itl_std_ms: float
    # System
    peak_ram_mb_mean: float
    mean_cpu_pct_mean: float
    num_runs: int


# ══════════════════════════════════════════════════════════════
#  BACKENDS
# ══════════════════════════════════════════════════════════════

class MockBackend:
    """Simulates realistic M5 LLaMA latency — no model needed."""
    def load(self, model: str, device: str) -> float:
        t0 = time.perf_counter()
        time.sleep(0.05)  # simulate load
        print("  [Mock] Simulating Apple Silicon M5 performance ✓")
        return (time.perf_counter() - t0) * 1000

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float):
        results = []
        for _ in prompts:
            ttft = random.gauss(80, 8)
            tps_base = random.gauss(55, 3)
            token_timestamps = []
            t = time.perf_counter() + ttft / 1000
            for _ in range(max_new_tokens):
                t += (1 / tps_base) + random.gauss(0, 0.001)
                token_timestamps.append(t)
            results.append({
                "ttft_ms": max(ttft, 10),
                "tokens": list(range(max_new_tokens)),
                "token_timestamps": token_timestamps
            })
        return results


class OllamaBackend:
    """Ollama backend — Metal GPU auto-enabled on Apple Silicon."""
    def __init__(self):
        self.host = "http://localhost:11434"
        self.model_name = "llama3.2"

    def load(self, model: str, device: str) -> float:
        import urllib.request
        self.model_name = model
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=3)
        except Exception:
            print("\n  ✗ Ollama not running! Run:  ollama serve")
            sys.exit(1)

        # Measure model load time with a tiny prompt
        print(f"  [Ollama] Warming up {model}...", end="", flush=True)
        t0 = time.perf_counter()
        payload = json.dumps({
            "model": model, "prompt": "hi", "stream": False,
            "options": {"num_predict": 1, "num_gpu": 999}
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=120)
        except Exception:
            pass
        load_ms = (time.perf_counter() - t0) * 1000
        print(f" done  ({load_ms:.0f}ms load time)")
        return load_ms

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float):
        import urllib.request
        results = []
        for prompt in prompts:
            payload = json.dumps({
                "model": self.model_name, "prompt": prompt, "stream": True,
                "options": {"temperature": temperature, "num_predict": max_new_tokens, "num_gpu": 999}
            }).encode()
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            token_timestamps, generated_tokens = [], []
            first_token, ttft = True, 0.0
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
            results.append({"ttft_ms": ttft, "tokens": generated_tokens, "token_timestamps": token_timestamps})
        return results


class HuggingFaceBackend:
    """HuggingFace Transformers with MPS (Apple Metal)."""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self, model: str, device: str) -> float:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            print("  ✗ Run: pip install transformers torch"); sys.exit(1)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  [HF] Loading {model} on {self.device}...")
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto").to(self.device)
        self.model.eval()
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"  [HF] Loaded ✓  ({load_ms:.0f}ms)")
        return load_ms

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float):
        import torch
        results = []
        for prompt in prompts:
            inputs = {k: v.to(self.device) for k, v in self.tokenizer(prompt, return_tensors="pt").items()}
            token_timestamps, generated_ids = [], []
            past_kv, input_ids = None, inputs["input_ids"]
            first_token, ttft = True, 0.0
            gen_start = time.perf_counter()
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
                logits = out.logits[:, -1, :]
                next_id = logits.argmax(dim=-1, keepdim=True) if temperature == 0 else \
                          torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
                ts = time.perf_counter()
                if first_token:
                    ttft = (ts - gen_start) * 1000; first_token = False
                token_timestamps.append(ts); generated_ids.append(next_id.item())
                past_kv = out.past_key_values; input_ids = next_id
                if next_id.item() == self.tokenizer.eos_token_id:
                    break
            results.append({"ttft_ms": ttft, "tokens": generated_ids, "token_timestamps": token_timestamps})
        return results


class LlamaCppBackend:
    """llama.cpp with full Metal GPU offload."""
    def __init__(self):
        self.llm = None

    def load(self, model: str, device: str) -> float:
        try:
            from llama_cpp import Llama
        except ImportError:
            print("  ✗ Run: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal")
            sys.exit(1)
        print(f"  [llama.cpp] Loading {model}...")
        t0 = time.perf_counter()
        self.llm = Llama(model_path=model, n_gpu_layers=-1, verbose=False)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"  [llama.cpp] Loaded ✓  ({load_ms:.0f}ms)")
        return load_ms

    def generate(self, prompts: List[str], max_new_tokens: int, temperature: float):
        results = []
        for prompt in prompts:
            token_timestamps, generated_tokens = [], []
            first_token, ttft = True, 0.0
            gen_start = time.perf_counter()
            for chunk in self.llm(prompt, max_tokens=max_new_tokens, temperature=max(temperature, 1e-8), stream=True):
                ts = time.perf_counter()
                tok = chunk["choices"][0].get("text", "")
                if tok:
                    if first_token:
                        ttft = (ts - gen_start) * 1000; first_token = False
                    token_timestamps.append(ts); generated_tokens.append(tok)
            results.append({"ttft_ms": ttft, "tokens": generated_tokens, "token_timestamps": token_timestamps})
        return results


BACKENDS = {"mock": MockBackend, "ollama": OllamaBackend, "hf": HuggingFaceBackend, "llamacpp": LlamaCppBackend}


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def build_prompt(target_tokens: int) -> str:
    base = ("The quick brown fox jumps over the lazy dog. "
            "Artificial intelligence and large language models are transforming "
            "how we benchmark inference latency and throughput on Apple Silicon. ")
    return (base * (math.ceil(target_tokens * 4 / len(base)) + 2))[: target_tokens * 4]


def compute_itl(timestamps: List[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    return statistics.mean((timestamps[i] - timestamps[i-1]) * 1000 for i in range(1, len(timestamps)))


def pct(data: List[float], p: int) -> float:
    if not data: return 0.0
    s = sorted(data)
    return s[max(0, int(len(s) * p / 100) - 1)]


def std(data: List[float]) -> float:
    return statistics.stdev(data) if len(data) > 1 else 0.0


# ══════════════════════════════════════════════════════════════
#  CORE BENCHMARK
# ══════════════════════════════════════════════════════════════

def run_one_model(args, model: str) -> tuple:
    """Run full benchmark for one model. Returns (load_ms, aggregated_results)."""
    backend = BACKENDS[args.backend]()
    monitor = SystemMonitor()

    print(f"\n{'─'*60}")
    print(f"  MODEL: {model}")
    print(f"{'─'*60}")
    load_ms = backend.load(model, args.device)

    configs = [(pl, ol, bs) for pl in args.prompt_lengths
               for ol in args.output_lengths for bs in args.batch_sizes]
    total = len(configs) * args.num_runs
    done  = 0
    raw: List[RunResult] = []

    for prompt_len, output_len, batch_size in configs:
        prompts = [build_prompt(prompt_len) for _ in range(batch_size)]

        print(f"\n  ┌─ prompt={prompt_len:>4}t  output={output_len:>4}t  batch={batch_size}")
        print(f"  │  warmup ({args.warmup})...", end="", flush=True)
        for _ in range(args.warmup):
            backend.generate(prompts, output_len, args.temperature)
        print(" done")

        for run_i in range(args.num_runs):
            monitor.start()
            t0 = time.perf_counter()
            outputs = backend.generate(prompts, output_len, args.temperature)
            t1 = time.perf_counter()
            sys_stats = monitor.stop()

            ttft_avg     = statistics.mean(o["ttft_ms"] for o in outputs)
            itl_avg      = statistics.mean(compute_itl(o["token_timestamps"]) for o in outputs)
            total_tokens = sum(len(o["tokens"]) for o in outputs)
            e2e_ms       = (t1 - t0) * 1000
            tps          = total_tokens / (e2e_ms / 1000) if e2e_ms > 0 else 0
            tps_per_item = tps / batch_size

            raw.append(RunResult(
                backend=args.backend, model=model,
                prompt_length=prompt_len, output_length=output_len, batch_size=batch_size,
                ttft_ms=ttft_avg, tps=tps, tps_per_item=tps_per_item,
                e2e_latency_ms=e2e_ms, itl_ms=itl_avg,
                actual_tokens=total_tokens,
                peak_ram_mb=sys_stats["peak_ram_mb"],
                mean_cpu_pct=sys_stats["mean_cpu_pct"],
                run_index=run_i
            ))
            done += 1
            pct_done = done / total * 100
            bar = "█" * int(pct_done / 5) + "░" * (20 - int(pct_done / 5))
            ram_str = f"  RAM={sys_stats['peak_ram_mb']:.0f}MB" if PSUTIL_OK else ""
            cpu_str = f"  CPU={sys_stats['mean_cpu_pct']:.0f}%" if PSUTIL_OK else ""
            print(f"  │  [{bar}] {pct_done:5.1f}%  "
                  f"TTFT={ttft_avg:6.0f}ms  TPS={tps:5.1f}  "
                  f"E2E={e2e_ms:6.0f}ms{ram_str}{cpu_str}")

    return load_ms, aggregate(raw)


def aggregate(raw: List[RunResult]) -> List[AggregatedResult]:
    from itertools import groupby
    key = lambda r: (r.backend, r.model, r.prompt_length, r.output_length, r.batch_size)
    results = []
    for gk, grp in groupby(sorted(raw, key=key), key=key):
        runs = list(grp)
        ttft  = [r.ttft_ms for r in runs]
        tps_  = [r.tps for r in runs]
        tpi   = [r.tps_per_item for r in runs]
        e2e   = [r.e2e_latency_ms for r in runs]
        itl   = [r.itl_ms for r in runs]
        ram   = [r.peak_ram_mb for r in runs]
        cpu   = [r.mean_cpu_pct for r in runs]
        results.append(AggregatedResult(
            backend=gk[0], model=gk[1],
            prompt_length=gk[2], output_length=gk[3], batch_size=gk[4],
            ttft_mean_ms=statistics.mean(ttft), ttft_std_ms=std(ttft),
            ttft_p50_ms=pct(ttft,50), ttft_p95_ms=pct(ttft,95), ttft_p99_ms=pct(ttft,99),
            tps_mean=statistics.mean(tps_), tps_std=std(tps_),
            tps_p50=pct(tps_,50), tps_p95=pct(tps_,95),
            tps_per_item_mean=statistics.mean(tpi),
            e2e_mean_ms=statistics.mean(e2e), e2e_std_ms=std(e2e), e2e_p95_ms=pct(e2e,95),
            itl_mean_ms=statistics.mean(itl), itl_std_ms=std(itl),
            peak_ram_mb_mean=statistics.mean(ram), mean_cpu_pct_mean=statistics.mean(cpu),
            num_runs=len(runs)
        ))
    return results


# ══════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════

def print_table(results: List[AggregatedResult], load_times: dict):
    W = 120
    print("\n" + "═" * W)
    print("  RESULTS SUMMARY")
    print("═" * W)

    # Model load times
    if load_times:
        print("\n  Model Load Times:")
        for model, ms in load_times.items():
            print(f"    {model:<30} {ms:>8.0f} ms")

    print(f"\n  {'MODEL':<18} {'PROMPT':>7} {'OUTPUT':>7} {'BATCH':>6} "
          f"{'TTFT±σ(ms)':>14} {'P95-TTFT':>9} "
          f"{'TPS±σ':>10} {'TPS/item':>9} "
          f"{'E2E(ms)':>9} {'ITL(ms)':>9} "
          f"{'RAM(MB)':>8} {'CPU%':>6}")
    print("  " + "─" * (W - 2))

    prev_model = None
    for r in results:
        model_short = r.model.split("/")[-1][:17]
        if model_short != prev_model:
            if prev_model is not None:
                print()
            prev_model = model_short

        ttft_str = f"{r.ttft_mean_ms:.1f}±{r.ttft_std_ms:.1f}"
        tps_str  = f"{r.tps_mean:.1f}±{r.tps_std:.1f}"
        ram_str  = f"{r.peak_ram_mb_mean:.0f}" if PSUTIL_OK else "N/A"
        cpu_str  = f"{r.mean_cpu_pct_mean:.0f}" if PSUTIL_OK else "N/A"

        print(f"  {model_short:<18} {r.prompt_length:>7} {r.output_length:>7} {r.batch_size:>6} "
              f"{ttft_str:>14} {r.ttft_p95_ms:>9.1f} "
              f"{tps_str:>10} {r.tps_per_item_mean:>9.1f} "
              f"{r.e2e_mean_ms:>9.0f} {r.itl_mean_ms:>9.2f} "
              f"{ram_str:>8} {cpu_str:>6}")

    print("═" * W)

    # Key insights
    print("\n  KEY INSIGHTS:")
    all_ttft = [r.ttft_mean_ms for r in results]
    all_tps  = [r.tps_mean for r in results]
    all_ram  = [r.peak_ram_mb_mean for r in results]
    print(f"    • TTFT range  : {min(all_ttft):.1f}ms – {max(all_ttft):.1f}ms")
    print(f"    • TPS range   : {min(all_tps):.1f} – {max(all_tps):.1f} tokens/sec")
    if PSUTIL_OK:
        print(f"    • Peak RAM    : {max(all_ram):.0f} MB")

    # Batch scaling insight
    b1 = [r for r in results if r.batch_size == 1]
    b4 = [r for r in results if r.batch_size == 4]
    if b1 and b4:
        avg_tps_b1 = statistics.mean(r.tps_mean for r in b1)
        avg_tps_b4 = statistics.mean(r.tps_mean for r in b4)
        print(f"    • Batch 1→4 TPS scaling: {avg_tps_b1:.1f} → {avg_tps_b4:.1f} "
              f"({avg_tps_b4/avg_tps_b1:.2f}x)")

    print()


def save_results(all_results: List[AggregatedResult], load_times: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON (includes load times)
    json_path = os.path.join(out_dir, f"benchmark_v2_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": ts,
                "load_times_ms": load_times,
                "psutil_available": PSUTIL_OK
            },
            "results": [asdict(r) for r in all_results]
        }, f, indent=2)

    # CSV
    csv_path = os.path.join(out_dir, f"benchmark_v2_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=asdict(all_results[0]).keys())
        w.writeheader()
        w.writerows(asdict(r) for r in all_results)

    print(f"  💾  JSON → {json_path}")
    print(f"  💾  CSV  → {csv_path}")
    return json_path, csv_path


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="LLaMA Latency Benchmark v2.0 — Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_v2.py --mock
  python benchmark_v2.py --models llama3.2
  python benchmark_v2.py --models llama3.2 llama3.2:1b
  python benchmark_v2.py --models llama3.2 --prompt-lengths 64 256 512 --num-runs 10
        """
    )
    p.add_argument("--mock",           action="store_true",  help="Simulation mode (no model)")
    p.add_argument("--backend",        default="ollama",     choices=["ollama","hf","llamacpp","mock"])
    p.add_argument("--models",         nargs="+",            default=["llama3.2"], help="One or more models")
    p.add_argument("--prompt-lengths", nargs="+", type=int,  default=[64, 128, 256, 512])
    p.add_argument("--output-lengths", nargs="+", type=int,  default=[64, 128, 256])
    p.add_argument("--batch-sizes",    nargs="+", type=int,  default=[1, 2, 4])
    p.add_argument("--num-runs",       type=int,  default=5)
    p.add_argument("--warmup",         type=int,  default=2)
    p.add_argument("--temperature",    type=float,default=0.0)
    p.add_argument("--device",         default="auto")
    p.add_argument("--out-dir",        default="results")
    args = p.parse_args()

    if args.mock:
        args.backend = "mock"
        args.models  = ["mock-llama3.2-3b", "mock-llama3.2-1b"]

    total_configs = len(args.prompt_lengths) * len(args.output_lengths) * len(args.batch_sizes)

    print("""
╔══════════════════════════════════════════════════════╗
║  LLaMA Latency Benchmark v2.0 • Apple Silicon        ║
╚══════════════════════════════════════════════════════╝""")
    print(f"  Backend  : {args.backend}")
    print(f"  Models   : {args.models}")
    print(f"  Prompts  : {args.prompt_lengths} tokens")
    print(f"  Outputs  : {args.output_lengths} tokens")
    print(f"  Batches  : {args.batch_sizes}")
    print(f"  Runs     : {args.num_runs} measured + {args.warmup} warmup")
    print(f"  Configs  : {total_configs} × {len(args.models)} model(s) = "
          f"{total_configs * len(args.models)} total")
    print(f"  RAM/CPU  : {'✓ enabled (psutil)' if PSUTIL_OK else '✗ disabled (pip install psutil)'}")

    all_results: List[AggregatedResult] = []
    load_times: dict = {}

    for model in args.models:
        load_ms, model_results = run_one_model(args, model)
        load_times[model] = load_ms
        all_results.extend(model_results)

    print_table(all_results, load_times)
    save_results(all_results, load_times, args.out_dir)
    print("\n  ✓ Done! Run visualize_results.py to generate charts.\n")


if __name__ == "__main__":
    main()
