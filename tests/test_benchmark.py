from progression_memory.benchmarks.synthetic import run_synthetic_benchmark


def test_adaptive_benchmark_beats_fixed_window_boundary_detection() -> None:
    results = {result.system: result for result in run_synthetic_benchmark(seed=7)}
    fixed = results["fixed_14d"]
    adaptive = results["adaptive_linear"]

    assert adaptive.f1 >= 0.9
    assert adaptive.f1 > fixed.f1
    assert adaptive.segment_count < fixed.segment_count
