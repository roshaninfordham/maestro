import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Literal, Optional

import dacite
import lightning
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor

from maestro.trainer.common.callbacks import SaveCheckpoint
from maestro.trainer.common.datasets.core import create_data_loaders, resolve_dataset_path
from maestro.trainer.common.metrics import BaseMetric, MetricsTracker, parse_metrics, save_metric_plots
from maestro.trainer.common.training import MaestroTrainer
from maestro.trainer.common.utils.device import device_is_available, parse_device_spec
from maestro.trainer.common.utils.path import create_new_run_directory
from maestro.trainer.common.utils.seed import ensure_reproducibility
from maestro.trainer.logger import get_maestro_logger
from maestro.trainer.models.gemma_3.checkpoints import (
    DEFAULT_GEMMA3_MODEL_ID,
    DEFAULT_GEMMA3_MODEL_REVISION,
    OptimizationStrategy,
    load_model,
    save_model,
)
from maestro.trainer.models.gemma_3.inference import predict_with_inputs
from maestro.trainer.models.gemma_3.loaders import evaluation_collate_fn, train_collate_fn

logger = get_maestro_logger()


@dataclass()
class Gemma3Configuration:
    """
    Configuration for training the Gemma3 model.

    Attributes:
        dataset (str):
            Local path or Roboflow identifier. If not found locally, it will be resolved (and downloaded) automatically.
        model_id (str):
            Identifier for the Gemma3 model.
        revision (str):
            Model revision to use.
        device (str | torch.device):
            Device to run training on. Can be a ``torch.device`` or a string such as
            "auto", "cpu", "cuda", or "mps". If "auto", the code will pick the best
            available device.
        optimization_strategy (Literal["lora", "qlora", "freeze", "none"]):
            Strategy for optimizing the model parameters.
        cache_dir (Optional[str]):
            Directory to cache the model weights locally.
        epochs (int):
            Number of training epochs.
        lr (float):
            Learning rate for training.
        batch_size (int):
            Training batch size.
        accumulate_grad_batches (int):
            Number of batches to accumulate before performing a gradient update.
        val_batch_size (Optional[int]):
            Validation batch size. If None, defaults to the training batch size.
        num_workers (int):
            Number of workers for data loading.
        val_num_workers (Optional[int]):
            Number of workers for validation data loading. If None, defaults to num_workers.
        output_dir (str):
            Directory to store training outputs.
        metrics (list[BaseMetric] | list[str]):
            Metrics to track during training. Can be a list of metric objects or metric names.
        max_new_tokens (int):
            Maximum number of new tokens generated during inference.
        random_seed (Optional[int]):
            Random seed for ensuring reproducibility. If None, no seeding is applied.
        peft_advanced_params (Optional[dict]):
            Custom LoRA configuration . If None, default configuration is applied.
    """

    dataset: str
    model_id: str = DEFAULT_GEMMA3_MODEL_ID
    revision: str = DEFAULT_GEMMA3_MODEL_REVISION
    device: str | torch.device = "auto"
    optimization_strategy: Literal["lora", "qlora", "freeze", "none"] = "lora"
    cache_dir: Optional[str] = None
    epochs: int = 10
    lr: float = 1e-5
    batch_size: int = 4
    accumulate_grad_batches: int = 8
    val_batch_size: Optional[int] = None
    num_workers: int = 0
    val_num_workers: Optional[int] = None
    output_dir: str = "./training/gemma_3"
    metrics: list[BaseMetric] | list[str] = field(default_factory=list)
    max_new_tokens: int = 512
    random_seed: Optional[int] = None
    peft_advanced_params: Optional[dict] = None

    def __post_init__(self):
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size

        if self.val_num_workers is None:
            self.val_num_workers = self.num_workers

        if isinstance(self.metrics, list) and all(isinstance(m, str) for m in self.metrics):
            self.metrics = parse_metrics(self.metrics)

        self.device = parse_device_spec(self.device)
        if not device_is_available(self.device):
            raise ValueError(f"Requested device '{self.device}' is not available.")


class Gemma3Trainer(MaestroTrainer):
    """
    Trainer for fine-tuning the Gemma-3 model.

    Attributes:
        processor (Gemma3Processor): Tokenizer and processor for model inputs.
        model (Gemma3ForConditionalGeneration): Pre-trained Gemma-3 model.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        config (Gemma3Configuration): Configuration object containing training parameters.
    """

    def __init__(
        self,
        processor: Gemma3Processor,
        model: Gemma3ForConditionalGeneration,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: Gemma3Configuration,
    ):
        super().__init__(processor, model, train_loader, valid_loader)
        self.config = config

        # TODO: Redesign metric tracking system
        self.train_metrics_tracker = MetricsTracker.init(metrics=["loss"])
        metrics = ["loss"]
        for metric in config.metrics:
            if isinstance(metric, BaseMetric):
                metrics += metric.describe()  # ensure mypy understands it's BaseMetric
        self.valid_metrics_tracker = MetricsTracker.init(metrics=metrics)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, pixel_values, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=self.config.batch_size)
        self.train_metrics_tracker.register("loss", epoch=self.current_epoch, step=batch_idx, value=loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, prefixes, suffixes = batch
        generated_suffixes = predict_with_inputs(
            model=self.model,
            processor=self.processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            device=self.config.device,
            max_new_tokens=self.config.max_new_tokens,
        )

        if batch_idx == 0:
            logger.info(f"sample valid prefix: {prefixes[0]}")
            logger.info(f"sample valid suffix: {suffixes[0]}")
            logger.info(f"sample generated suffix: {generated_suffixes[0]}")

        for metric in self.config.metrics:
            result = metric.compute(predictions=generated_suffixes, targets=suffixes)
            for key, value in result.items():
                self.valid_metrics_tracker.register(
                    metric=key,
                    epoch=self.current_epoch,
                    step=batch_idx,
                    value=value,
                )
                self.log(key, value, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def on_fit_end(self) -> None:
        save_metrics_path = os.path.join(self.config.output_dir, "metrics")
        save_metric_plots(
            training_tracker=self.train_metrics_tracker,
            validation_tracker=self.valid_metrics_tracker,
            output_dir=save_metrics_path,
        )


def train(config: Gemma3Configuration | dict) -> None:
    if isinstance(config, dict):
        config = dacite.from_dict(data_class=Gemma3Configuration, data=config)
    assert isinstance(config, Gemma3Configuration)  # ensure mypy understands it's not a dict

    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)
    run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    config = replace(config, output_dir=run_dir)

    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),
        peft_advanced_params=config.peft_advanced_params,
        cache_dir=config.cache_dir,
    )
    dataset_location = resolve_dataset_path(config.dataset)
    if dataset_location is None:
        return
    train_loader, valid_loader, test_loader = create_data_loaders(
        dataset_location=dataset_location,
        train_batch_size=config.batch_size,
        train_collect_fn=partial(train_collate_fn, processor=processor, max_length=config.max_new_tokens),
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor),
        test_num_workers=config.val_num_workers,
    )

    _, train_entry = train_loader.dataset[0]
    logger.info(f"sample train prefix: {train_entry['prefix']}")
    logger.info(f"sample train suffix: {train_entry['suffix']}")

    pl_module = Gemma3Trainer(
        processor=processor, model=model, train_loader=train_loader, valid_loader=valid_loader, config=config
    )
    save_checkpoints_path = os.path.join(config.output_dir, "checkpoints")
    save_checkpoint_callback = SaveCheckpoint(result_path=save_checkpoints_path, save_model_callback=save_model)
    trainer = lightning.Trainer(
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        check_val_every_n_epoch=1,
        limit_val_batches=1,
        log_every_n_steps=10,
        callbacks=[save_checkpoint_callback],
    )
    trainer.fit(pl_module)
