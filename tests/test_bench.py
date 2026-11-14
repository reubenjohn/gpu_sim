import pathlib
import shutil
import subprocess

import pytest

import gpu_sim
from gpu_sim import bench

PREAMBLE = (
    "Generating SQLite file /tmp/profile.sqlite from /tmp/profile.nsys-rep\n"
    "Processing [/tmp/profile.sqlite] with [/opt/nvidia/nsys/reports/{report}.py]... \n"
)
MEM_HEADER = '"Total (MB)","Count","Avg (MB)","Med (MB)","Min (MB)","Max (MB)","StdDev (MB)","Operation"\n'
MEM_ROWS = (
    '"4.194","3","1.398","1.398","1.398","1.398","0.000","[CUDA memcpy Host-to-Device]"\n'
    '"0.004","2","0.002","0.002","0.002","0.002","0.000","[CUDA memcpy Device-to-Host]"\n'
    '"800.000","50","16.000","16.000","16.000","16.000","0.000","[CUDA memcpy Device-to-Device]"\n'
    '"64.000","7","9.143","8.000","8.000","16.000","3.024","[CUDA memset]"\n'
)
API_HEADER = (
    '"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"\n'
)
API_ROWS = (
    '"60.0","6000","10","600.0","600.0","500","700","50.0","cudaLaunchKernel"\n'
    '"20.0","2000","4","500.0","500.0","500","500","0.0","cuLaunchKernel"\n'
    '"15.0","1500","5","300.0","300.0","300","300","0.0","cudaGraphLaunch_v10000"\n'
    '"5.0","500","100","5.0","5.0","5","5","0.0","cudaMemcpyAsync"\n'
)


def nsys_output(header: str, body: str, report: str, quoted: bool = True) -> str:
    """What `nsys stats --format csv --output -` prints: preamble lines, then the table."""
    table = header + body
    return PREAMBLE.format(report=report) + (table if quoted else table.replace('"', ""))


def test_parse_result_last_line_wins():
    stdout = 'warmup\nBENCH_RESULT {"steps": 1}\nnoise\nBENCH_RESULT {"steps": 2, "seconds": 3.5}\n'
    assert bench.parse_result(stdout) == {"steps": 2, "seconds": 3.5}


def test_parse_result_missing_line_raises():
    with pytest.raises(ValueError, match="BENCH_RESULT"):
        bench.parse_result("no result here\n")


@pytest.mark.parametrize("quoted", [True, False], ids=["quoted", "unquoted"])
def test_count_host_memcpys_ignores_device_to_device_and_memset(quoted):
    assert bench.count_host_memcpys(nsys_output(MEM_HEADER, MEM_ROWS, "cuda_gpu_mem_size_sum", quoted)) == 5


@pytest.mark.parametrize("quoted", [True, False], ids=["quoted", "unquoted"])
def test_count_launches_strips_version_suffix_and_skips_non_launch_apis(quoted):
    assert bench.count_api_calls(nsys_output(API_HEADER, API_ROWS, "cuda_api_sum", quoted), bench.LAUNCH_APIS) == 19


def test_counts_are_zero_when_nsys_skips_the_report():
    skipped = "SKIPPED: /tmp/profile.sqlite does not contain CUDA trace data.\n"
    assert bench.count_host_memcpys(skipped) == 0
    assert bench.count_api_calls(skipped, bench.LAUNCH_APIS) == 0


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


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("nsys") is None, reason="nsys not installed")
def test_nsys_available():
    """Placeholder for the end-to-end bench run until a reference workload exists."""
    assert subprocess.run(["nsys", "--version"], check=True, capture_output=True, text=True).stdout


def test_report_fails_when_capture_range_never_fired():
    assert any("no launches captured" in f for f in report(launches=0).failures())


def test_report_fails_on_host_syncs_beyond_telemetry_allowance():
    assert report(steps=1_000_000, host_syncs=10).failures() == []
    assert "host syncs 11 > 10" in report(steps=1_000_000, host_syncs=11).failures()


def test_count_api_calls_counts_syncs():
    csv_text = '"Num Calls","Name"\n"7","cudaStreamSynchronize"\n"2","cuCtxSynchronize_v2000"\n"5","cudaLaunchKernel"\n'
    assert bench.count_api_calls(csv_text, bench.SYNC_APIS) == 9


def test_thresholds_match_brief():
    """bench.py constants must equal BRIEF C1-C3; change both together."""
    brief = (pathlib.Path(__file__).parents[1] / "docs/BRIEF.md").read_text()
    assert "steps // 100_000" in brief and bench.TELEMETRY_INTERVAL == 100_000
    assert "≤ 100 host-side" in brief and bench.LAUNCHES_PER_10K_MAX == 100
    assert "< 7.0e9 bytes" in brief and bench.VRAM_LIMIT_BYTES == 7.0e9
