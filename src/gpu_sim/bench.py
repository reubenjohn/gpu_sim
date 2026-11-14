"""Phase 1 profiling harness: run a workload under nsys and gate it on the BRIEF invariants.

Workload contract (any framework):
  1. Wrap only the timed loop in cudaProfilerStart()/cudaProfilerStop(); nsys runs with
     `--capture-range=cudaProfilerApi`, so everything nsys reports belongs to the timed loop.
  2. Print one line `BENCH_RESULT {"steps": N, "seconds": S, "peak_vram_bytes": B}` to stdout.
     steps: batched steps (Stage 1: generations). peak_vram_bytes: peak device memory *used*
     (NVML, incl. CUDA context and framework pools), not the framework's allocator counter.
     JAX workloads set XLA_PYTHON_CLIENT_PREALLOCATE=false.
  3. Pack telemetry into one device buffer, so one readback = one device-to-host copy.

The counting is nsys's own (`nsys stats` reports as CSV); this module only reads the totals.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

VRAM_LIMIT_BYTES = 7.0e9  # BRIEF C3 (tests/test_bench.py asserts these equal the BRIEF)
LAUNCHES_PER_10K_MAX = 100  # BRIEF C2: at most 1 host launch per 100 steps (proposed; Q5)
TELEMETRY_INTERVAL = 100_000  # BRIEF C1: one host copy and one host sync per this many steps
SYNC_APIS = {"cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"}
SYNC_APIS |= {"cuCtxSynchronize", "cuStreamSynchronize", "cuEventSynchronize"}
HOST_COPIES = ("Host-to-Device", "Device-to-Host")
LAUNCH_APIS = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"}
LAUNCH_APIS |= {"cudaGraphLaunch", "cuGraphLaunch", "cudaLaunchCooperativeKernel"}
RESULT_PREFIX = "BENCH_RESULT "


@dataclass
class Report:
    steps: int
    seconds: float
    peak_vram_bytes: int
    host_memcpys: int
    launches: int
    host_syncs: int

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
        return asdict(self) | derived | {"failures": self.failures()}


def parse_result(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if not lines:
        raise ValueError(f"workload printed no `{RESULT_PREFIX.strip()}` line")
    return json.loads(lines[-1].removeprefix(RESULT_PREFIX))


def rows(report_csv: str, column: str) -> list[dict]:
    """CSV rows from the first line whose fields include `column` (skips nsys's preamble lines,
    whether or not the header is quoted)."""
    lines = report_csv.splitlines()
    start = next((i for i, line in enumerate(lines) if column in next(csv.reader([line]), [])), len(lines))
    return list(csv.DictReader(lines[start:]))


def count_host_memcpys(mem_size_csv: str) -> int:
    """Rows of the `cuda_gpu_mem_size_sum` report: Count per memory operation."""
    return sum(
        int(row["Count"]) for row in rows(mem_size_csv, "Operation") if any(k in row["Operation"] for k in HOST_COPIES)
    )


def count_api_calls(api_sum_csv: str, names: set[str]) -> int:
    """Rows of the `cuda_api_sum` report: total calls to the `names` APIs (version suffix dropped)."""
    return sum(
        int(row["Num Calls"]) for row in rows(api_sum_csv, "Name") if re.sub(r"_v\d+$", "", row["Name"]) in names
    )


def nsys_report(rep: Path, report: str) -> str:
    cmd = ["nsys", "stats", "--report", report, "--format", "csv", "--output", "-", "--force-export=true", str(rep)]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout  # rows() skips the preamble


def profile(command: list[str], out_dir: Path) -> Report:
    rep = out_dir / "profile"
    nsys = ["nsys", "profile", "--capture-range=cudaProfilerApi", "--capture-range-end=stop"]  # default kills the app
    nsys += ["--trace=cuda,nvtx", "--force-overwrite=true"]
    run = subprocess.run([*nsys, "-o", str(rep), *command], check=True, capture_output=True, text=True)
    result = parse_result(run.stdout)
    rep = rep.with_suffix(".nsys-rep")
    api_sum = nsys_report(rep, "cuda_api_sum")
    return Report(
        steps=int(result["steps"]),
        seconds=float(result["seconds"]),
        peak_vram_bytes=int(result["peak_vram_bytes"]),
        host_memcpys=count_host_memcpys(nsys_report(rep, "cuda_gpu_mem_size_sum")),
        launches=count_api_calls(api_sum, LAUNCH_APIS),
        host_syncs=count_api_calls(api_sum, SYNC_APIS),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu-sim bench", description=__doc__.splitlines()[0])
    parser.add_argument("--keep", type=Path, help="directory to keep the .nsys-rep in (default: temp)")
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
