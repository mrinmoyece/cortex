"""The Celery layer, which was at 0% coverage.

Worker configuration is the kind of code that is never tested and always
wrong in the same three ways: acks that drop work on a crash, prefetch that
starves other workers, and a beat schedule that silently disagrees with the
documentation.
"""

from __future__ import annotations

import pytest

from cortex.workers.celery_app import celery_app


class TestCeleryConfiguration:
    def test_acks_late_so_a_crashed_worker_does_not_lose_the_task(self):
        """With acks_early (the default) a worker that dies mid-task has
        already acknowledged it, and the run vanishes silently."""
        assert celery_app.conf.task_acks_late is True

    def test_prefetch_of_one_so_a_long_task_does_not_starve_the_queue(self):
        """Agent runs are minutes long and wildly variable. The default
        prefetch of 4 lets one worker grab four long runs while others idle."""
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_hard_timeout_exceeds_the_soft_one(self):
        """If the hard limit fires first the task is killed before its
        cleanup handler can run, and the soft limit is decoration."""
        assert celery_app.conf.task_time_limit > celery_app.conf.task_soft_time_limit

    def test_broker_and_backend_are_configured(self):
        assert celery_app.conf.broker_url
        assert celery_app.conf.result_backend


class TestBeatSchedule:
    def test_exactly_the_two_documented_periodic_tasks(self):
        schedule = celery_app.conf.beat_schedule
        assert len(schedule) == 2, f"expected 2 periodic tasks, found {sorted(schedule)}"

    def test_eval_regression_runs_every_six_hours(self):
        """Crontab, not an interval - `crontab(minute=0, hour="*/6")` fires
        at fixed wall-clock hours, so the schedule does not drift every time
        a worker restarts. Asserting on the fields rather than a seconds
        count is what makes that distinction visible."""
        entry = next(v for v in celery_app.conf.beat_schedule.values() if "eval" in v["task"])
        assert entry["schedule"].hour == {0, 6, 12, 18}
        assert entry["schedule"].minute == {0}

    def test_memory_consolidation_runs_hourly(self):
        entry = next(v for v in celery_app.conf.beat_schedule.values() if "memory" in v["task"])
        assert len(entry["schedule"].hour) == 24, "should fire every hour"
        assert entry["schedule"].minute == {30}, "offset from the hour avoids the eval spike"

    def test_scheduled_tasks_go_to_dedicated_queues(self):
        """Maintenance work sharing the agent queue would let a consolidation
        pass block user runs behind it."""
        queues = {v["options"]["queue"] for v in celery_app.conf.beat_schedule.values()}
        assert queues == {"eval", "maintenance"}

    def test_every_scheduled_task_actually_exists(self):
        """A beat schedule pointing at a task name that was renamed fails
        once an hour, in a worker log nobody reads."""
        for entry in celery_app.conf.beat_schedule.values():
            assert entry["task"] in celery_app.tasks, f"{entry['task']} is not registered"


class TestTaskRegistration:
    @pytest.mark.parametrize(
        "name",
        [
            "cortex.workers.celery_app.run_agent_task",
            "cortex.workers.celery_app.run_eval_regression",
            "cortex.workers.celery_app.consolidate_stale_memory",
            "cortex.workers.celery_app.ingest_document_task",
        ],
    )
    def test_task_is_registered_under_its_explicit_name(self, name):
        """Names are pinned explicitly rather than inferred, so moving the
        module does not silently orphan every queued task."""
        assert name in celery_app.tasks

    def test_agent_task_retries_rather_than_dropping_work(self):
        task = celery_app.tasks["cortex.workers.celery_app.run_agent_task"]
        assert task.max_retries is not None and task.max_retries >= 1
        assert task.queue == "agent"
