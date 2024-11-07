import argparse
from typing import Any

import wandb
from loguru import logger
from tqdm import tqdm
from wandb.apis.public import Run


def get_runs_for_sweep(*sweep_ids: str) -> list[Run]:
    """Get run IDs for the provided sweep ids."""
    api = wandb.Api()
    runs = api.runs("lewidi/cluster")
    logger.info("Getting runs from sweeps")
    sweep_runs = [
        run for run in tqdm(runs, desc="Getting runs") if getattr(run, "sweep", None) is not None
    ]
    logger.info("Getting relevant runs")
    runs = [
        run
        for run in tqdm(sweep_runs, desc="Getting sweep runs")
        if run.sweep.id in sweep_ids and run.state == "finished"
    ]
    return runs


def get_runs_for_group(group: str = "Best Models") -> list[Run]:
    """Get run IDs for the provided group."""
    api = wandb.Api()
    runs = api.runs("lewidi/cluster", include_sweeps=False)
    logger.info("Getting runs from group")
    group_runs = [
        run for run in tqdm(runs, desc="Getting runs") if getattr(run, "group", None) == group
    ]
    return group_runs


def get_best_run_id_from_sweep(sweep_id: str) -> str:
    """Get the best run ID from a sweep."""
    runs_for_sweep = get_runs_for_sweep(sweep_id)
    if not runs_for_sweep:
        raise ValueError(f"No runs found for sweep {sweep_id}")

    best_run = sorted(runs_for_sweep, key=lambda run: run.summary["val_loss"])[0]
    return best_run.id


def get_args_from_run_id(run_id: str) -> argparse.Namespace:
    """Get the checkpoint path from the run id."""
    run = wandb.Api().run(f"lewidi/cluster/{run_id}")
    return argparse.Namespace(**run.config)


def add_config_entry_to_run(run_id: str, key: str, value: Any) -> None:
    """Add a config entry to a run."""
    run = wandb.Api().run(f"lewidi/cluster/{run_id}")
    run.config.update({key: value})
    run.update()
