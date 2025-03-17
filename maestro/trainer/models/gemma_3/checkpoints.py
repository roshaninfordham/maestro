import os
from enum import Enum
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor

from maestro.trainer.common.utils.device import parse_device_spec
from maestro.trainer.logger import get_maestro_logger

DEFAULT_GEMMA3_MODEL_ID = "google/gemma-3-4b-pt"  # `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`
DEFAULT_GEMMA3_MODEL_REVISION = "refs/heads/main"
DEFAULT_GEMMA3_PEFT_PARAMS = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
    "task_type": "CAUSAL_LM",
}
logger = get_maestro_logger()


class OptimizationStrategy(Enum):
    """Enumeration for optimization strategies."""

    LORA = "lora"
    FREEZE = "freeze"
    NONE = "none"


def load_model(
    model_id_or_path: str = DEFAULT_GEMMA3_MODEL_ID,
    revision: str = DEFAULT_GEMMA3_MODEL_REVISION,
    device: str | torch.device = "auto",
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.NONE,
    peft_advanced_params: Optional[dict] = None,
    cache_dir: Optional[str] = None,
) -> tuple[Gemma3Processor, Gemma3ForConditionalGeneration]:
    """Loads a Gemma 3 model and its associated processor.

    Args:
        model_id_or_path (str): The identifier or path of the Gemma 3 model to load.
        revision (str): The specific model revision to use.
        device (torch.device): The device to load the model onto.
        optimization_strategy (OptimizationStrategy): The optimization strategy to apply to the model.
        peft_advanced_params: custom lora configuration
        cache_dir (Optional[str]): Directory to cache the downloaded model files.

    Returns:
        tuple(AutoProcessor, AutoModelForImageTextToText):
            A tuple containing the loaded processor and model.

    Raises:
        ValueError: If the model or processor cannot be loaded.
    """
    device = parse_device_spec(device)
    processor = Gemma3Processor.from_pretrained(model_id_or_path, trust_remote_code=True, revision=revision)

    if optimization_strategy == OptimizationStrategy.LORA:
        default_params = DEFAULT_GEMMA3_PEFT_PARAMS
        if peft_advanced_params is not None:
            default_params.update(peft_advanced_params)
            try:
                config = LoraConfig(**default_params)
                logger.info("Successfully created LoraConfig")
            except TypeError:
                logger.exception("Invalid parameters for LoraConfig")
                raise
        else:
            logger.info("No LoRA parameters provided. Using default configuration.")
            config = LoraConfig(**default_params)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        model = get_peft_model(model, config).to(device)
        model.print_trainable_parameters()
    else:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id_or_path,
            revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
        ).to(device)

        if optimization_strategy == OptimizationStrategy.FREEZE:
            for param in model.vision_tower.parameters():
                param.is_trainable = False

    return processor, model


def save_model(
    target_dir: str,
    processor: Gemma3Processor,
    model: Gemma3ForConditionalGeneration,
) -> None:
    os.makedirs(target_dir, exist_ok=True)
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)
