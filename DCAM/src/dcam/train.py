"""Train entry point for DCAM."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf

from dcam.data.builders import build_dataloaders
from dcam.engine.evaluate import export_inference_artifacts
from dcam.engine.pretrain import pretrain_autoencoder
from dcam.engine.train import CurriculumState, train_dcam
from dcam.models.dcam import build_dcam_model
from dcam.utils.io import ensure_dir, save_json
from dcam.utils.logging import get_logger
from dcam.utils.seed import seed_everything

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    logger.info("\n%s", OmegaConf.to_yaml(cfg))

    train_loader, val_loader, eval_loader, metadata = build_dataloaders(cfg.data)
    model = build_dcam_model(cfg=cfg, metadata=metadata)

    fabric = Fabric(
        accelerator=cfg.trainer.device,
        precision=cfg.trainer.precision,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        num_nodes=cfg.trainer.num_nodes,
    )
    fabric.launch()

    if bool(cfg.compile_model) and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt_pre = torch.optim.Adam(model.ae.parameters(), lr=float(cfg.trainer.optim.lr_d))
    opt_e = torch.optim.Adam(model.ae.encoder_parameters(), lr=float(cfg.trainer.optim.lr_e))
    opt_d = torch.optim.Adam(model.ae.decoder_parameters(), lr=float(cfg.trainer.optim.lr_d))
    opt_rho = torch.optim.Adam([model.rho], lr=float(cfg.trainer.optim.lr_rho))

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
    )

    # rho init from the first train batch, matching Algorithm 1.
    init_batch = next(iter(train_loader))
    model.initialize_rho_from_batch(fabric.to_device(init_batch["x"]))

    # Dedicated DCAM optimizers and schedulers.
    opt_e = torch.optim.Adam(model.ae.encoder_parameters(), lr=float(cfg.trainer.optim.lr_e))
    opt_d = torch.optim.Adam(model.ae.decoder_parameters(), lr=float(cfg.trainer.optim.lr_d))
    opt_rho = torch.optim.Adam([model.rho], lr=float(cfg.trainer.optim.lr_rho))

    opt_e, opt_d, opt_rho = fabric.setup_optimizers(opt_e, opt_d, opt_rho)

    scheduler_kwargs = dict(
        mode="min",
        factor=float(cfg.trainer.scheduler.factor),
        patience=int(cfg.trainer.scheduler.patience),
        min_lr=float(cfg.trainer.scheduler.min_lr),
        threshold=float(cfg.trainer.scheduler.threshold),
    )
    sch_e = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_e, **scheduler_kwargs)
    sch_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, **scheduler_kwargs)
    sch_rho = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_rho, **scheduler_kwargs)

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
        curriculum_state=CurriculumState(T=int(cfg.trainer.curriculum.t_start)),
        scheduler_cfg=cfg.trainer.scheduler,
        curriculum_cfg=cfg.trainer.curriculum,
        logging_cfg=cfg.trainer.logging,
        output_dir=output_dir,
        logger=logger,
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


if __name__ == "__main__":
    main()
