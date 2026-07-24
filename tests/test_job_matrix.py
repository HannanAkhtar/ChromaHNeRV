from collections import Counter

from uvg_utils import build_uvg_jobs


def test_default_job_matrix_has_280_unique_jobs():
    jobs = build_uvg_jobs()
    assert len(jobs) == 280
    assert len({job["run_id"] for job in jobs}) == 280
    counts = Counter((job["sequence"], job["modelsize"]) for job in jobs)
    assert set(counts.values()) == {10}
    for job in jobs:
        if job["split_stage"] is None:
            assert job["branch_width"] is None
        else:
            assert job["branch_width"] in (8, 4)
