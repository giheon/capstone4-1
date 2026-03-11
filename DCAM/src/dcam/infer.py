"""Inference / export entry point."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf

from dcam.data.builders import build_dataloaders
from dcam.engine.evaluate import export_inference_artifacts
from dcam.models.dcam import build_dcam_model
from dcam.utils.io import ensure_dir, save_json
from dcam.utils.logging import get_logger
from dcam.utils.seed import seed_everything

logger = get_logger(__name__)
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.checkpoint_path is None:
        raise ValueError("Set checkpoint_path=... for inference")

    seed_everything(int(cfg.seed))
    logger.info("\n%s", OmegaConf.to_yaml(cfg))

    _, _, eval_loader, metadata = build_dataloaders(cfg.data)
    model = build_dcam_model(cfg=cfg, metadata=metadata)

    fabric = Fabric(
        accelerator=cfg.trainer.device,
        precision=cfg.trainer.precision,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        num_nodes=cfg.trainer.num_nodes,
    )
    fabric.launch()

    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = fabric.setup(model)
    eval_loader = fabric.setup_dataloaders(eval_loader)

    T = int(checkpoint.get("T", cfg.trainer.curriculum.t_start))
    output_dir = ensure_dir(Path(cfg.output_dir) / "inference")
    save_json(output_dir / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))

    export_inference_artifacts(
        fabric=fabric,
        model=model,
        dataloader=eval_loader,
        T=T,
        output_dir=output_dir,
        dataset_type=metadata.dataset_type,
        num_channels=metadata.num_channels,
    )
    logger.info("Saved inference artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
