import json
import os
import pathlib
import shutil
import subprocess

import pytest

import gpu_sim
from gpu_sim import bench


def test_parse_result_last_line_wins():
    stdout = 'warmup\nBENCH_RESULT {"steps": 1}\nnoise\nBENCH_RESULT {"steps": 2, "seconds": 3.5}\n'
    assert bench.parse_result(stdout) == {"steps": 2, "seconds": 3.5}


def test_parse_result_missing_line_raises():
    with pytest.raises(ValueError, match="BENCH_RESULT"):
        bench.parse_result("no result here\n")


# Real CUPTI counter output for tests/fixtures/probe.cu (1000 launches, 3 H2D, 2 D2H, 4 syncs).
PROBE_COUNTS = {
    "cuModuleGetFunction": 1,
    "cuStreamSynchronize": 4,
    "cuMemcpyHtoDAsync_v2": 3,
    "cuMemcpyDtoHAsync_v2": 2,
    "cuLaunchKernel": 1000,
    "cuProfilerStart": 1,
    "cuCtxPushCurrent_v2": 1,
    "cuCtxPopCurrent_v2": 1,
    "cuLibraryLoadData": 1,
    "cuLibraryGetModule": 1,
}


def test_classify_probe_counts():
    assert bench.classify(PROBE_COUNTS) == {"launches": 1000, "host_memcpys": 5, "host_syncs": 4}


def test_classify_launch_variants_and_suffixes():
    counts = {"cuLaunchKernelEx": 2, "cuLaunchCooperativeKernel_ptsz": 3, "cuGraphLaunch": 5, "cuLaunchHostFunc": 7}
    assert bench.classify(counts)["launches"] == 10


def test_classify_host_copies_skip_device_to_device_and_memset():
    counts = {"cuMemcpyHtoA_v2": 1, "cuMemcpyAtoH_v2": 1, "cuMemcpyDtoD_v2": 50, "cuMemsetD32Async": 9}
    assert bench.classify(counts)["host_memcpys"] == 2


def test_classify_counts_direction_unknown_copies_as_host():
    assert bench.classify({"cuMemcpy": 1, "cuMemcpyAsync_ptsz": 2, "cuMemcpyPeer": 4})["host_memcpys"] == 3


def test_classify_syncs_ignore_event_record_and_stream_wait():
    counts = {"cuCtxSynchronize": 1, "cuEventSynchronize": 2, "cuStreamSynchronize_ptsz": 3, "cuEventRecord": 208}
    counts |= {"cuStreamWaitEvent": 107}
    assert bench.classify(counts)["host_syncs"] == 6


def test_classify_empty_counts_is_all_zero():
    assert bench.classify({}) == {"launches": 0, "host_memcpys": 0, "host_syncs": 0}


def report(**overrides) -> bench.Report:
    fields = {
        "steps": 1_000_000,
        "seconds": 10.0,
        "peak_vram_bytes": 1_000_000_000,
        "host_memcpys": 0,
        "launches": 1,
        "host_syncs": 0,
    }
    return bench.Report(**fields | overrides)


def test_report_passes_and_derives_rates():
    r = report()
    assert r.failures() == []
    assert r.steps_per_hour == 360_000_000
    assert r.launches_per_10k == pytest.approx(0.01)
    assert r.summary()["failures"] == []
    assert r.summary()["env_steps_per_hour"] == 0


def test_summary_reports_summed_env_steps_per_hour():
    assert report(env_steps=1_000_000_000).summary()["env_steps_per_hour"] == 360_000_000_000


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"host_memcpys": 11}, "host memcpys 11 > 10"),
        ({"steps": 99_999, "host_memcpys": 1}, "host memcpys 1 > 0"),
        ({"launches": 10_001}, "launches/10k"),
        ({"peak_vram_bytes": 7_000_000_000}, "peak VRAM 7.00 GB"),
    ],
    ids=["memcpys", "memcpys-below-interval", "launches", "vram"],
)
def test_report_failure_modes(overrides, message):
    failures = report(**overrides).failures()
    assert len(failures) == 1 and message in failures[0]


def test_telemetry_allowance_is_steps_over_interval():
    assert report(steps=1_000_000, host_memcpys=10).failures() == []
    assert report(steps=1_999_999, host_memcpys=19).failures() == []


def test_cli_unknown_subcommand_returns_2(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["gpu-sim", "nope"])
    assert gpu_sim.main() == 2
    assert "usage: gpu-sim" in capsys.readouterr().err


def test_bench_without_command_errors(capsys):
    with pytest.raises(SystemExit) as exc:
        bench.main([])
    assert exc.value.code == 2
    assert "workload command" in capsys.readouterr().err


NVCC = shutil.which("nvcc") or str(bench.CUDA_HOME / "bin/nvcc")
HAS_GPU = (
    shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
)


@pytest.mark.gpu
@pytest.mark.skipif(
    not (pathlib.Path(NVCC).exists() and shutil.which("gcc") and HAS_GPU), reason="needs nvcc, gcc, GPU"
)
def test_bench_end_to_end_on_probe(tmp_path, capsys):
    probe = tmp_path / "probe"
    source = pathlib.Path(__file__).parent / "fixtures/probe.cu"
    env = os.environ | {"NVCC_CCBIN": os.environ.get("NVCC_CCBIN", "g++-14")}
    subprocess.run([NVCC, "-arch=sm_61", "-o", str(probe), str(source)], check=True, env=env)
    assert bench.main(["--keep", str(tmp_path), "--", str(probe)]) == 1  # 1000 launches/1000 steps fails C2
    summary = json.loads(capsys.readouterr().out)
    assert (summary["launches"], summary["host_memcpys"], summary["host_syncs"]) == (1000, 5, 4)
    assert (tmp_path / "counts.json").exists()


def test_report_fails_when_capture_range_never_fired():
    assert any("no launches captured" in f for f in report(launches=0).failures())


def test_report_fails_on_host_syncs_beyond_telemetry_allowance():
    assert report(steps=1_000_000, host_syncs=10).failures() == []
    assert "host syncs 11 > 10" in report(steps=1_000_000, host_syncs=11).failures()


def test_thresholds_match_brief():
    """bench.py constants must equal BRIEF C1-C3; change both together."""
    brief = (pathlib.Path(__file__).parents[1] / "docs/BRIEF.md").read_text()
    assert "steps // 100_000" in brief and bench.TELEMETRY_INTERVAL == 100_000
    assert "≤ 100 host-side" in brief and bench.LAUNCHES_PER_10K_MAX == 100
    assert "< 7.0e9 bytes" in brief and bench.VRAM_LIMIT_BYTES == 7.0e9
