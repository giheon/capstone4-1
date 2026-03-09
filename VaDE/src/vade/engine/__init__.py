"""Training and evaluation engines."""

from vade.engine.diagnostics import (
    DiagnosticsConfig,
    append_assignment_history,
    run_diagnostics,
    save_assignment_heatmaps,
    save_init_final_comparison,
)
from vade.engine.evaluate import encode_dataset, predict_gamma
from vade.engine.pretrain import (
    get_author_pretrain_weight_path,
    initialize_from_pretraining,
    load_author_pretrained_autoencoder,
    pretrain_autoencoder,
)
from vade.engine.train import train_vade

__all__ = [
    "append_assignment_history",
    "DiagnosticsConfig",
    "encode_dataset",
    "get_author_pretrain_weight_path",
    "initialize_from_pretraining",
    "load_author_pretrained_autoencoder",
    "predict_gamma",
    "pretrain_autoencoder",
    "run_diagnostics",
    "save_assignment_heatmaps",
    "save_init_final_comparison",
    "train_vade",
]
