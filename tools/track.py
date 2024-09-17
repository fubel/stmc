import json
import os
from subprocess import PIPE, run

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from qqdm import format_str, qqdm

import wandb
from src.datasets.dataset import create_dataloader
from src.tracker.encoder import create_encoder
from src.tracker.solver import create_solver
from src.tracker.tracker import create_tracker
from src.utils.evaluate import evaluate_tracker
from src.utils.iotools import ResultsWriter


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.device == "cpu" or not torch.cuda.is_available():
        raise ValueError("This code runs on CUDA only. Please set device to 'cuda'.")
    else:
        device = torch.device(cfg.device)
        logger.info(f"🚀 Using device: {device}")

    cfg.tracker.matching.distance_weight = 1 - cfg.tracker.matching.rescale_weight

    # create output directories
    output_path = os.path.join(cfg.output_path)
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, cfg.dataset.name)
    logger.info(f"📂 Writing to output path: {output_path}")

    # Initialize wandb and tensorboard
    if cfg.logging.wandb.enable:
        wandb.init(project=cfg.logging.wandb.project)
        wandb.config.update(OmegaConf.to_container(cfg))
        if cfg.logging.wandb.tags is not None:
            wandb.run.tags = cfg.logging.wandb.tags

    # Initialize solver
    solver_opts = create_solver(cfg.solver.backend)
    logger.info(f"✨ Initialized solver, using backend: {cfg.solver.backend}")

    # Initialize dataset and dataloader
    dataloader = create_dataloader(cfg)
    logger.info("✨ Created dataloader.")

    # Initialize encoder
    encoder = create_encoder(cfg.encoder, device)
    logger.info("✨ Created encoder.")

    tracker = create_tracker(cfg, solver_opts, encoder, len(dataloader.dataset.camera_names), device)
    logger.info("✨ Initialized tracker.")

    results_writer = ResultsWriter(
        output_path=output_path,
        cfg=cfg,
        normalization=dataloader.dataset._norm_factors,
        camera_names=dataloader.dataset.camera_names,
    )

    tw = qqdm(range(len(dataloader)), desc=format_str("bold", "Description"))
    for i, batch in enumerate(dataloader):
        results, _ = tracker.step(batch)
        results_writer.add(results)
        stats = tracker.stats
        tw.set_infos(stats)
        tw.update()

        if cfg.logging.wandb.enable:
            _stats_str_to_float = {k: float(v) for k, v in stats.items()}
            wandb.log(_stats_str_to_float, step=i)

    logger.info(f"🕒 Cumulative execution time of tracker {tracker.cumulative_execution_time * 10}")
    logger.info(f"🕒 Average time per frame {tracker.cumulative_execution_time / tracker.frame}")

    results_writer.save()

    logger.info("🚀 Tracking completed.")
    logger.info(
        f"📈 Results saved to {results_writer.results_file}. "
        "Use the official evaluation script of the dataset for evaluation."
    )


if __name__ == "__main__":
    main()
