from uvg_utils import backup_artifact_names


def test_final_backup_artifact_list_is_complete():
    expected = {
        "model_latest.pth", "model_best.pth", "epoch150.pth",
        "epoch150.csv", "completion.json", "config.json", "command.txt",
        "environment.txt", "git_commit.txt", "rank0.txt", "quant_vid.pth",
        "img_decoder.pth", "/logs/run.log",
    }
    assert expected <= set(backup_artifact_names(
        150, final=True, launcher_log="/logs/run.log",
        latest_csv="epoch150.csv"))


def test_intermediate_backup_includes_best_and_metadata():
    expected = {
        "model_latest.pth", "model_best.pth", "rank0.txt", "config.json",
        "command.txt", "environment.txt", "git_commit.txt",
    }
    assert expected <= set(backup_artifact_names(150))
