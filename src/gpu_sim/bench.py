"""Phase 1 profiling harness: count a workload's CUDA driver calls with CUPTI and gate it on the BRIEF invariants.

Workload contract (any framework):
  1. Wrap only the timed loop in cudaProfilerStart()/cudaProfilerStop(); the CUPTI counter
     (cucount.c, injected via CUDA_INJECTION64_PATH) counts only between those calls.
  2. Print one line `BENCH_RESULT {"steps": N, "seconds": S, "peak_vram_bytes": B}` to stdout.
     steps: batched steps (Stage 1: generations). peak_vram_bytes: peak device memory *used*
     (NVML, incl. CUDA context and framework pools), not the framework's allocator counter.
     JAX workloads set XLA_PYTHON_CLIENT_PREALLOCATE=false.
     Stage 2 adds `"env_steps": E`: env-steps summed across all towns (steps x towns).
  3. Pack telemetry into one device buffer, so one readback = one device-to-host copy.

Why CUPTI and not nsys: nsys records no CUDA data on WSL2 + Pascal (DECISIONS.md, Rejected).
Needs gcc and a CUDA 12.x toolkit with CUPTI (CUDA_HOME, default /usr/local/cuda).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

VRAM_LIMIT_BYTES = 7.0e9  # BRIEF C3 (tests/test_bench.py asserts these equal the BRIEF)
LAUNCHES_PER_10K_MAX = 100  # BRIEF C2: at most 1 host launch per 100 steps (provisional; BACKLOG steps-fused sweep)
TELEMETRY_INTERVAL = 100_000  # BRIEF C1: one host copy and one host sync per this many steps
RESULT_PREFIX = "BENCH_RESULT "

# Driver API names, after stripping version (_v2) and per-thread-stream (_ptds/_ptsz) suffixes.
LAUNCH_PREFIXES = ("cuLaunchKernel", "cuLaunchCooperativeKernel", "cuGraphLaunch")
HOST_COPY_MARKERS = ("HtoD", "DtoH", "HtoA", "AtoH")
# cuMemcpy/cuMemcpyAsync take unified-address pointers, so the direction is unknown here.
# Count them as host copies to be safe: a false alarm beats a missed host round-trip.
GENERIC_COPIES = {"cuMemcpy", "cuMemcpyAsync"}
SYNC_PREFIXES = ("cuCtxSynchronize", "cuStreamSynchronize", "cuEventSynchronize")

CUCOUNT_SRC = Path(__file__).with_name("cucount.c")
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gpu_sim"
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))


@dataclass
class Report:
    steps: int
    seconds: float
    peak_vram_bytes: int
    host_memcpys: int
    launches: int
    host_syncs: int
    env_steps: int = 0  # Stage 2 only: summed across towns (BRIEF Targets)

    @property
    def steps_per_hour(self) -> float:
        return self.steps / self.seconds * 3600

    @property
    def launches_per_10k(self) -> float:
        return self.launches / self.steps * 10_000

    def failures(self) -> list[str]:
        allowed_copies = self.steps // TELEMETRY_INTERVAL
        checks = [
            (self.launches > 0, "no launches captured: did the workload call cudaProfilerStart/Stop?"),
            (self.host_memcpys <= allowed_copies, f"host memcpys {self.host_memcpys} > {allowed_copies}"),
            (self.host_syncs <= allowed_copies, f"host syncs {self.host_syncs} > {allowed_copies}"),
            (self.launches_per_10k <= LAUNCHES_PER_10K_MAX, f"launches/10k {self.launches_per_10k:.1f}"),
            (self.peak_vram_bytes < VRAM_LIMIT_BYTES, f"peak VRAM {self.peak_vram_bytes / 1e9:.2f} GB"),
        ]
        return [message for passed, message in checks if not passed]

    def summary(self) -> dict:
        derived = {"steps_per_hour": self.steps_per_hour, "launches_per_10k": self.launches_per_10k}
        derived["env_steps_per_hour"] = self.env_steps / self.seconds * 3600
        return asdict(self) | derived | {"failures": self.failures()}


def parse_result(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        raise ValueError(f"workload printed no `{RESULT_PREFIX.strip()}` line")
    return json.loads(lines[-1].removeprefix(RESULT_PREFIX))


def classify(api_counts: dict[str, int]) -> dict[str, int]:
    """Fold the counter's {driverApiName: calls} into the gated totals."""
    totals = {"launches": 0, "host_memcpys": 0, "host_syncs": 0}
    for name, calls in api_counts.items():
        base = re.sub(r"_(v\d+|ptds|ptsz)", "", name)
        if base.startswith(LAUNCH_PREFIXES):
            totals["launches"] += calls
        elif base in GENERIC_COPIES or any(marker in base for marker in HOST_COPY_MARKERS):
            totals["host_memcpys"] += calls
        elif base.startswith(SYNC_PREFIXES):
            totals["host_syncs"] += calls
    return totals


def build_counter() -> Path:
    """Compile cucount.c into the cache dir, unless the cached library is newer than the source."""
    lib = CACHE_DIR / "libcucount.so"
    if lib.exists() and lib.stat().st_mtime >= CUCOUNT_SRC.stat().st_mtime:
        return lib
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    libdir = CUDA_HOME / "targets/x86_64-linux/lib"
    cmd = ["gcc", "-shared", "-fPIC", "-O2", "-o", str(lib), str(CUCOUNT_SRC), f"-I{CUDA_HOME}/include"]
    subprocess.run([*cmd, f"-L{libdir}", "-lcupti", f"-Wl,-rpath,{libdir}"], check=True)
    return lib


def profile(command: list[str], out_dir: Path) -> Report:
    counts_path = out_dir / "counts.json"
    counts_path.unlink(missing_ok=True)
    env = os.environ | {"CUDA_INJECTION64_PATH": str(build_counter()), "CUCOUNT_OUT": str(counts_path)}
    run = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    result = parse_result(run.stdout)
    if not counts_path.exists():
        raise RuntimeError("CUPTI counter wrote no counts: the workload never initialised CUDA")
    return Report(
        steps=int(result["steps"]),
        seconds=float(result["seconds"]),
        peak_vram_bytes=int(result["peak_vram_bytes"]),
        env_steps=int(result.get("env_steps", 0)),
        **classify(json.loads(counts_path.read_text())),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu-sim bench", description=__doc__.splitlines()[0])
    parser.add_argument("--keep", type=Path, help="directory to keep the raw counts.json in (default: temp)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="workload command, after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the workload command after --")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = args.keep or Path(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = profile(command, out_dir)
    print(json.dumps(report.summary(), indent=2))
    return 1 if report.failures() else 0


if __name__ == "__main__":
    sys.exit(main())
