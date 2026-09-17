"""A real-data acceptance test that must fail if the reliability claim is unmet.

This is intentionally separate from implementation tests. Do not weaken the
thresholds to make a release green; report the measured failure/inconclusiveness.
"""
import pytest

from stylometry.benchmark import run_benchmark
from stylometry.benchmark_data import load_benchmark
from stylometry.benchmark_suite import DEFAULT_MANIFESTS, REPO


@pytest.mark.empirical
def test_real_reference_author_reliability(tmp_path):
    works = [work for manifest in DEFAULT_MANIFESTS
             for work in load_benchmark(manifest, REPO / 'data/raw/benchmarks', download=False)]
    report = run_benchmark(works, tmp_path / 'empirical')
    failures = [f"{run['language']}/{run['sample_length']}/{run.get('ablation', 'unavailable')}: "
                + ', '.join(f"{name}={gate['status']}" for name, gate in run.get('gates', {}).items()
                            if gate['status'] != 'passed')
                for run in report['runs'] if run['status'] != 'passed']
    assert report['status'] == 'passed', (
        'Real-author reliability is not established. Do not equate passing implementation tests with attribution validation.\n'
        + '\n'.join(failures) + f"\nFull benchmark: {tmp_path / 'empirical/report.md'}"
    )
