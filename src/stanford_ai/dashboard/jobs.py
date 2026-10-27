"""One local background simulation at a time. Worker never calls Streamlit APIs."""

import threading
from dataclasses import dataclass, field

from stanford_ai.simulation.runner import Runner
from stanford_ai.storage.sqlite import Store


@dataclass
class Job:
    run_id: str
    stop: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    error: str | None = None


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.current: Job | None = None

    def start(self, config, path) -> Job:
        with self.lock:
            if self.current and not self.current.done.is_set():
                raise ValueError("A local experiment is already running. Stop it or wait for completion.")
            # Construct runner on worker-owned SQLite connection in the worker.
            import uuid
            job = Job(str(uuid.uuid4()))
            self.current = job

            def work():
                try:
                    with Store(path) as store:
                        runner = Runner(config, store)
                        runner.run_id = job.run_id
                        runner.run(should_stop=job.stop.is_set)
                except Exception as exc:
                    job.error = type(exc).__name__ + ": inspect local terminal for details"
                    import traceback
                    traceback.print_exc()
                finally:
                    job.done.set()

            threading.Thread(target=work, name=f"experiment-{job.run_id}", daemon=True).start()
            return job
