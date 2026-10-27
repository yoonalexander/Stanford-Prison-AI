from pathlib import Path

from streamlit.testing.v1 import AppTest

from stanford_ai.models import ExperimentConfig
from stanford_ai.simulation.runner import Runner
from stanford_ai.storage.sqlite import Store

APP = Path(__file__).parents[1] / "src/stanford_ai/dashboard/app.py"


def test_empty_dashboard_renders_without_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("STANFORD_AI_DB", str(tmp_path / "empty.sqlite"))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert len(app.tabs) == 6
    assert app.title[0].value == "Authority, under observation."


def test_saved_run_profiles_charts_replay_and_comparison_render(tmp_path, monkeypatch):
    path = tmp_path / "saved.sqlite"
    monkeypatch.setenv("STANFORD_AI_DB", str(path))
    with Store(path) as store:
        run_id = Runner(ExperimentConfig(days=1), store).run()
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert app.session_state["selected_run"] == run_id
    next_button = next(b for b in app.button if b.label == "Next event")
    next_button.click().run(timeout=20)
    assert not app.exception
    assert app.slider[0].value == 1
    next(b for b in app.button if b.label == "Next event").click().run(timeout=20)
    assert app.slider[0].value == 2
    app.multiselect[0].select(run_id).run(timeout=20)
    assert not app.exception


def test_setup_starts_background_mock_experiment(tmp_path, monkeypatch):
    import time
    path = tmp_path / "started.sqlite"
    monkeypatch.setenv("STANFORD_AI_DB", str(path))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    next(b for b in app.button if b.label == "Start experiment").click().run(timeout=20)
    for _ in range(50):
        with Store(path) as store:
            runs = store.runs()
            if runs and runs[0]["status"] == "completed":
                break
        time.sleep(.1)
    assert runs and runs[0]["status"] == "completed"
    app.run(timeout=20)
    assert not app.exception
