"""Train entry point for DCAM."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf

from dcam.data.builders import build_dataloaders
from dcam.engine.diagnostics import run_clustering_diagnostics
from dcam.engine.evaluate import export_inference_artifacts
from dcam.engine.pretrain import pretrain_autoencoder
from dcam.engine.train import CurriculumState, train_dcam
from dcam.models.dcam import build_dcam_model
from dcam.utils.io import ensure_dir, save_json
from dcam.utils.logging import get_logger
from dcam.utils.seed import seed_everything
from dcam.utils.wandb import finish_wandb_run, init_wandb_run

logger = get_logger(__name__)
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _build_fabric(cfg: DictConfig) -> Fabric:
    return Fabric(
        accelerator=cfg.trainer.device,
        precision=cfg.trainer.precision,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        num_nodes=cfg.trainer.num_nodes,
    )


def _build_dcam_optimizers(model, cfg: DictConfig) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    opt_e = torch.optim.Adam(model.ae.encoder_parameters(), lr=float(cfg.trainer.optim.lr_e))
    opt_d = torch.optim.Adam(model.ae.decoder_parameters(), lr=float(cfg.trainer.optim.lr_d))
    opt_rho = torch.optim.Adam([model.rho], lr=float(cfg.trainer.optim.lr_rho))
    return opt_e, opt_d, opt_rho


def _build_plateau_schedulers(opt_e, opt_d, opt_rho, cfg: DictConfig):
    scheduler_kwargs = dict(
        mode="min",
        factor=float(cfg.trainer.scheduler.factor),
        patience=int(cfg.trainer.scheduler.patience),
        min_lr=float(cfg.trainer.scheduler.min_lr),
        threshold=float(cfg.trainer.scheduler.threshold),
    )
    return (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt_e, **scheduler_kwargs),
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, **scheduler_kwargs),
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt_rho, **scheduler_kwargs),
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    logger.info("\n%s", OmegaConf.to_yaml(cfg))
    wandb_run = init_wandb_run(cfg)

    try:
        train_loader, val_loader, eval_loader, metadata = build_dataloaders(cfg.data)
        model = build_dcam_model(cfg=cfg, metadata=metadata)
        fabric = _build_fabric(cfg)
        fabric.launch()

        if bool(cfg.compile_model) and hasattr(torch, "compile"):
            model = torch.compile(model)

        opt_pre = torch.optim.Adam(model.ae.parameters(), lr=float(cfg.trainer.optim.lr_d))

        model, opt_pre = fabric.setup(model, opt_pre)
        train_loader, val_loader, eval_loader = fabric.setup_dataloaders(train_loader, val_loader, eval_loader)

        output_dir = Path(cfg.output_dir)
        ensure_dir(output_dir)
        save_json(output_dir / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))

        logger.info("Starting AE pretraining")
        pretrain_autoencoder(
            fabric=fabric,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=opt_pre,
            epochs=int(cfg.trainer.pretrain.epochs),
            logger=logger,
            wandb_run=wandb_run,
        )

        init_batch = next(iter(train_loader))
        model.initialize_rho_from_batch(fabric.to_device(init_batch["x"]))

        opt_e, opt_d, opt_rho = _build_dcam_optimizers(model=model, cfg=cfg)
        opt_e, opt_d, opt_rho = fabric.setup_optimizers(opt_e, opt_d, opt_rho)
        sch_e, sch_d, sch_rho = _build_plateau_schedulers(opt_e=opt_e, opt_d=opt_d, opt_rho=opt_rho, cfg=cfg)

        curriculum_state = CurriculumState(T=int(cfg.trainer.curriculum.t_start))
        logger.info("Saving initial diagnostics before DCAM training")
        run_clustering_diagnostics(
            fabric=fabric,
            model=model,
            dataloader=eval_loader,
            T=curriculum_state.T,
            epoch=0,
            total_epochs=int(cfg.trainer.dcam.epochs),
            output_dir=output_dir,
            logging_cfg=cfg.trainer.logging,
            wandb_run=wandb_run,
        )

        logger.info("Starting DCAM training")
        best_payload = train_dcam(
            fabric=fabric,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_loader=eval_loader,
            opt_e=opt_e,
            opt_d=opt_d,
            opt_rho=opt_rho,
            sch_e=sch_e,
            sch_d=sch_d,
            sch_rho=sch_rho,
            epochs=int(cfg.trainer.dcam.epochs),
            curriculum_state=curriculum_state,
            curriculum_cfg=cfg.trainer.curriculum,
            logging_cfg=cfg.trainer.logging,
            output_dir=output_dir,
            logger=logger,
            wandb_run=wandb_run,
        )
        save_json(output_dir / "best_payload.json", best_payload)

        logger.info("Exporting inference artifacts from best current weights")
        export_inference_artifacts(
            fabric=fabric,
            model=model,
            dataloader=eval_loader,
            T=int(best_payload.get("T", cfg.trainer.curriculum.t_start)),
            output_dir=output_dir / "artifacts",
            dataset_type=metadata.dataset_type,
            num_channels=metadata.num_channels,
        )
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
