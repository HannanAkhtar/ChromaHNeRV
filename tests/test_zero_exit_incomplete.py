import pytest

from run_uvg_hnerv_suite import require_complete_run


def test_zero_exit_without_complete_artifacts_is_failure(tmp_path):
    with pytest.raises(RuntimeError, match="without complete final artifacts"):
        require_complete_run(tmp_path / "run", 150)
