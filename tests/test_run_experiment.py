import copy
import json
import yaml
import run_experiment as rx

BASE = yaml.safe_load(open("config.yaml"))


def _cfg(**llm):
    cfg = copy.deepcopy(BASE)
    cfg["llm"].update(llm)
    cfg["experiment"].update(n_graphs=4, p_out_values=[0.004], qa_n=10)
    return cfg


def test_planted_backend_end_to_end_and_outputs(tmp_path):
    cfg = _cfg(backend="planted")
    res = rx.run(cfg, tmp_path, log=lambda *_: None)
    for name in ("results.json", "summary.md", "config_used.yaml", "env.json"):
        assert (tmp_path / name).exists()
    assert len(list((tmp_path / "graphs").glob("*.json"))) >= 3
    a = res["0.004"]
    assert a["retention"]["l2:diff"]["mean"] < 0           # planted effect shows up
    assert set(a["interventions"]) >= {"lift_topology", "lift_control", "did"}
    assert "default_vs_topology" in a["qa"]
    text = (tmp_path / "summary.md").read_text()
    assert "L2 primary" in text and "Multi-hop QA" in text and "Holm" in text
    assert json.loads((tmp_path / "results.json").read_text()).keys() == res.keys()
    assert "config_sha256" in json.loads((tmp_path / "env.json").read_text())


def test_resume_does_not_recompute_finished_graphs(tmp_path):
    cfg = _cfg(backend="mock")
    cfg["experiment"]["n_graphs"] = 2
    rx.run(cfg, tmp_path, log=lambda *_: None)
    f = next((tmp_path / "graphs").glob("*.json"))
    before = f.stat().st_mtime_ns
    calls = []
    rx.run(cfg, tmp_path, log=calls.append)
    assert f.stat().st_mtime_ns == before and not any("done" in str(c) for c in calls)


def test_template_backend_runs_default_condition_only(tmp_path):
    cfg = _cfg(backend="template")
    cfg["experiment"]["n_graphs"] = 2
    res = rx.run(cfg, tmp_path, log=lambda *_: None)
    a = res["0.004"]
    assert "interventions" not in a and list(a["qa"]["accuracy"]) == ["default"]
    assert a["qa"]["accuracy"]["default"] == 1.0           # template keeps every input fact


def test_near_limit_prompts_are_counted(tmp_path):
    cfg = _cfg(backend="mock")
    cfg["experiment"].update(n_graphs=2, context_limit=50)
    res = rx.run(cfg, tmp_path, log=lambda *_: None)
    assert res["0.004"]["n_near_limit"] > 0


def test_disconnected_seed_is_skipped_not_fatal(tmp_path, monkeypatch):
    real = rx.build_graph
    def flaky(g):
        if g["seed"] == 1:
            raise ValueError("graph is not connected")
        return real(g)
    monkeypatch.setattr(rx, "build_graph", flaky)
    cfg = _cfg(backend="mock")
    cfg["experiment"]["n_graphs"] = 3
    res = rx.run(cfg, tmp_path, log=lambda *_: None)
    assert res["0.004"]["n_graphs"] == 2


def test_build_llm_rejects_unknown_backend():
    import pytest
    with pytest.raises(ValueError):
        rx.build_llm({"backend": "gpt"}, {}, 0)
