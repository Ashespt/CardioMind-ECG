from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    PreTrainedModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)
from transformers.utils import ModelOutput

from .data import LABEL_NAMES
from .output_space import exclude_sr


@dataclass
class ECGMultiTaskOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    classification_logits: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    classification_loss: Optional[torch.Tensor] = None
    base_classification_logits: Optional[torch.Tensor] = None
    clear_fusion_gate: Optional[torch.Tensor] = None


class ClearCLSGatedFusion(nn.Module):
    """Per-class calibrated CLEAR residual with a context-dependent reliability gate."""

    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.context_gate = nn.Linear(hidden_size, num_labels)
        self.clear_logit_scale = nn.Parameter(torch.ones(num_labels))
        self.clear_logit_bias = nn.Parameter(torch.zeros(num_labels))
        self.gate_logit_scale = nn.Parameter(torch.zeros(num_labels))
        self.gate_bias = nn.Parameter(torch.zeros(num_labels))
        self.residual_scale = nn.Parameter(torch.zeros(num_labels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)
        nn.init.ones_(self.clear_logit_scale)
        nn.init.zeros_(self.clear_logit_bias)
        nn.init.zeros_(self.gate_logit_scale)
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.residual_scale)

    def forward(
        self,
        base_logits: torch.Tensor,
        pooled: torch.Tensor,
        clear_probabilities: torch.Tensor,
        clear_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        working_dtype = base_logits.dtype
        probabilities = clear_probabilities.to(base_logits.device, torch.float32)
        clear_logits = torch.logit(probabilities.clamp(1e-5, 1.0 - 1e-5)).to(working_dtype)
        calibrated = (
            clear_logits * self.clear_logit_scale.to(working_dtype)
            + self.clear_logit_bias.to(working_dtype)
        )
        gate_logits = (
            self.context_gate(pooled.to(self.context_gate.weight.dtype)).to(working_dtype)
            + clear_logits * self.gate_logit_scale.to(working_dtype)
            + self.gate_bias.to(working_dtype)
        )
        gate = torch.sigmoid(gate_logits)
        if clear_mask is None:
            mask = torch.ones_like(clear_logits)
        else:
            mask = clear_mask.to(base_logits.device, working_dtype)
        correction = mask * gate * self.residual_scale.to(working_dtype) * calibrated
        return base_logits + correction, gate * mask


class ECGMultiTaskModel(PreTrainedModel):
    config_class = Qwen3VLConfig
    base_model_prefix = "vlm"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _tied_weights_keys = {
        "vlm.lm_head.weight": "vlm.model.language_model.embed_tokens.weight"
    }

    def __init__(self, config: Qwen3VLConfig, vlm: Optional[Qwen3VLForConditionalGeneration] = None):
        super().__init__(config)
        self.vlm = vlm if vlm is not None else Qwen3VLForConditionalGeneration(config)
        hidden_size = config.text_config.hidden_size
        num_labels = int(getattr(config, "ecg_num_labels", len(LABEL_NAMES)))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(float(getattr(config, "ecg_classifier_dropout", 0.1))),
            nn.Linear(hidden_size, num_labels),
        )
        self.clear_fusion = (
            ClearCLSGatedFusion(hidden_size, num_labels)
            if bool(getattr(config, "ecg_clear_late_fusion", False))
            else None
        )
        self.classification_loss_weight = float(
            getattr(config, "ecg_classification_loss_weight", 1.0)
        )
        self.lm_loss_weight = float(getattr(config, "ecg_lm_loss_weight", 1.0))
        self.classification_pos_weight = getattr(config, "ecg_classification_pos_weight", None)
        config.ecg_num_labels = num_labels
        config.ecg_label_names = list(getattr(config, "ecg_label_names", LABEL_NAMES))
        self.post_init()
        if self.clear_fusion is not None:
            # A new late-fusion checkpoint must initially reproduce the base logits.
            self.clear_fusion.reset_parameters()

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.vlm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.vlm.get_output_embeddings()

    def set_output_embeddings(self, value):
        return self.vlm.set_output_embeddings(value)

    def enable_input_require_grads(self):
        return self.vlm.enable_input_require_grads()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        return self.vlm.gradient_checkpointing_disable()

    def generate(self, *args, **kwargs):
        return self.vlm.generate(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        class_targets=None,
        classification_positions=None,
        clear_probabilities=None,
        clear_mask=None,
        **kwargs,
    ) -> ECGMultiTaskOutput:
        kwargs.pop("num_items_in_batch", None)
        model_outputs = self.vlm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )
        hidden = model_outputs.last_hidden_state
        if classification_positions is None:
            classification_positions = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_indices, classification_positions.to(hidden.device)]
        classifier_dtype = next(self.classifier.parameters()).dtype
        base_class_logits = self.classifier(pooled.to(classifier_dtype))
        class_logits = base_class_logits
        clear_fusion_gate = None
        if self.clear_fusion is not None and clear_probabilities is not None:
            class_logits, clear_fusion_gate = self.clear_fusion(
                base_class_logits, pooled, clear_probabilities, clear_mask
            )

        lm_loss = None
        if labels is not None and self.lm_loss_weight != 0.0:
            shifted_labels = labels[:, 1:].contiguous()
            active = shifted_labels.ne(-100)
            if active.any():
                active_hidden = hidden[:, :-1, :][active]
                token_logits = self.vlm.lm_head(active_hidden).float()
                lm_loss = F.cross_entropy(token_logits, shifted_labels[active])

        classification_loss = None
        if class_targets is not None:
            pos_weight = None
            if self.classification_pos_weight is not None:
                pos_weight = class_logits.new_tensor(self.classification_pos_weight).float()
            classification_loss = F.binary_cross_entropy_with_logits(
                class_logits.float(), class_targets.float(), pos_weight=pos_weight
            )

        loss = None
        if lm_loss is not None and classification_loss is not None:
            loss = (
                self.lm_loss_weight * lm_loss
                + self.classification_loss_weight * classification_loss
            )
        elif lm_loss is not None:
            loss = self.lm_loss_weight * lm_loss
        elif classification_loss is not None:
            loss = self.classification_loss_weight * classification_loss

        return ECGMultiTaskOutput(
            loss=loss,
            logits=exclude_sr(class_logits),
            classification_logits=exclude_sr(class_logits),
            lm_loss=lm_loss,
            classification_loss=classification_loss,
            base_classification_logits=exclude_sr(base_class_logits),
            clear_fusion_gate=exclude_sr(clear_fusion_gate) if clear_fusion_gate is not None else None,
        )


class ECGMultiTaskQwen25Model(ECGMultiTaskModel):
    config_class = Qwen2_5_VLConfig

    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        vlm: Optional[Qwen2_5_VLForConditionalGeneration] = None,
    ):
        if vlm is None:
            vlm = Qwen2_5_VLForConditionalGeneration(config)
        super().__init__(config, vlm=vlm)


def model_classes(model_type: str):
    if model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration, ECGMultiTaskQwen25Model
    if model_type == "qwen3_vl":
        return Qwen3VLForConditionalGeneration, ECGMultiTaskModel
    raise ValueError(
        f"Unsupported model_type {model_type!r}; expected 'qwen2_5_vl' or 'qwen3_vl'"
    )
