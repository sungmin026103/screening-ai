import pickle

import pytest

import projects


def test_save_project_state_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "touch_project", lambda slug: None)
    projects.save_project_state("p", "k", {"a": 1})
    path = projects.project_path("p") / "state" / "k.pkl"
    assert pickle.loads(path.read_bytes()) == {"a": 1}          # same pickle format
    assert projects.load_project_state("p", "k") == {"a": 1}

    class Bad:
        def __reduce__(self):
            raise RuntimeError("cannot pickle")
    with pytest.raises(RuntimeError):
        projects.save_project_state("p", "k", {"a": Bad()})
    assert projects.load_project_state("p", "k") == {"a": 1}   # old file intact after a failed write
    assert [f.name for f in path.parent.iterdir()] == ["k.pkl"]  # no temp file left behind
