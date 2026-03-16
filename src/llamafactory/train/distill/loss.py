from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from ...extras.logging import get_logger

logger = get_logger(__name__)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing_extensions import override

from ...hparams import FinetuningArguments, ModelArguments


class DistillLossInputs(BaseModel):
    labels: torch.Tensor = None
    outputs_student: CausalLMOutputWithPast
    outputs_teachers: List[CausalLMOutputWithPast]

    class Config:
        arbitrary_types_allowed = True

    def get_teachers_logits(self):
        """Extract logits from teacher outputs."""
        vocab = self.get_student_logits().shape[2]
        all_teacher_logits = []
        for i in range(len(self.outputs_teachers)):
            outputs_teachers_logits = self.outputs_teachers[i].logits[..., :vocab]
            all_teacher_logits.append(outputs_teachers_logits)
        avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        return avg_teacher_logits

    def get_student_logits(self) -> torch.FloatTensor:
        """Extract logits from student outputs."""
        return self.outputs_student.logits

    def get_teacher_hidden_state(self) -> Tuple[torch.FloatTensor, ...]:
        """Extract hidden states from teacher outputs."""
        if len(self.outputs_teachers) > 1:
            logger.warning(
                "Multi-teachers do not support hidden-states-based layerwise distill. We will use first teacher."
            )

        first_teacher_outputs = self.outputs_teachers[0]
        assert first_teacher_outputs.hidden_states is not None, "teacher_hidden_states should not be None"
        return first_teacher_outputs.hidden_states

    def get_student_hidden_state(self) -> Tuple[torch.FloatTensor, ...]:
        """Extract hidden states from student outputs."""
        assert self.outputs_student.hidden_states is not None, "student_hidden_states should not be None"
        return self.outputs_student.hidden_states

    def get_student_loss(self) -> torch.FloatTensor:
        assert self.outputs_student.loss is not None, "student_loss should not be None"
        return self.outputs_student.loss


class DistillationLoss(ABC):
    """An abstract class for various types of distillation losses."""

    require_adaptaion_layer: bool = False
    loss_type: Optional[str] = None

    def __init__(
        self,
        adaptation_layer: Optional[torch.nn.Module] = None,
        temperature: float = 2.0,
        max_length: int = 8196,
        hidden_states_alpha: float = 0.9,
        distill_alpha: float = 0.1,
    ):
        self.temperature = temperature
        self.max_length = max_length
        self.adaptation_layer = adaptation_layer
        self.hidden_states_alpha = hidden_states_alpha
        self.distill_alpha = distill_alpha

    @abstractmethod
    def compute_disitill_loss(self, inputs: DistillLossInputs):
        """Compute the loss between student and teacher outputs.

        :return: Computed loss.
        """
        pass

    def fuse_loss(self, distill_loss, student_loss):
        return self.distill_alpha * student_loss + (1.0 - self.distill_alpha) * distill_loss

    def __call__(self, inputs: DistillLossInputs):
        distill_loss = self.compute_disitill_loss(inputs)
        student_loss = inputs.get_student_loss()
        return self.fuse_loss(distill_loss, student_loss)


class LogitsKLDivergenceLoss(DistillationLoss):
    """Kullback-Leibler Divergence Loss for Knowledge Distillation."""

    loss_type = "logits-kld"

    @override
    def compute_disitill_loss(self, inputs: DistillLossInputs):
        # Compute KLD
        logits_kld_loss = (
            F.kl_div(
                F.log_softmax(inputs.get_student_logits() / self.temperature, dim=-1),
                F.softmax(inputs.get_teachers_logits() / self.temperature, dim=-1),
                reduction="batchmean",
            )
            * (self.temperature**2)
            / self.max_length
        )

        return logits_kld_loss


class LogitsKLDivergenceV2Loss(DistillationLoss):
    """Kullback-Leibler Divergence v2 Loss for Knowledge Distillation."""

    loss_type = "logits-kld-v2"

    @override
    def compute_disitill_loss(self, inputs: DistillLossInputs):
        # Compute KLD
        teacher_probs = F.softmax(inputs.get_teachers_logits(), dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(inputs.get_student_logits())
        logprobs = F.log_softmax(inputs.get_student_logits(), dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)

        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (inputs.labels != -100).int()
        logits_kld_v2_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return logits_kld_v2_loss


class LogitsReversedKLDivergenceLoss(DistillationLoss):
    """Reversed Kullback-Leibler Divergence Loss for Knowledge Distillation."""

    loss_type = "logits-reversed_kld"

    @override
    def compute_disitill_loss(self, inputs: DistillLossInputs):
        # Compute reverse KLD
        probs = F.softmax(inputs.get_student_logits(), dim=-1, dtype=torch.float32)
        teacher_inf_mask = torch.isinf(inputs.get_teachers_logits())
        teacher_logprobs = F.log_softmax(inputs.get_teachers_logits(), dim=-1, dtype=torch.float32)
        # print(f"probs: {probs.shape},  teacher_logprobs: {teacher_logprobs.shape}, teacher_inf_mask: {teacher_inf_mask.shape}")

        vocab = inputs.get_student_logits().shape[2]
        teacher_prod_probs = torch.masked_fill(
            probs[..., :vocab] * teacher_logprobs[..., :vocab], teacher_inf_mask[..., :vocab], 0
        )

        inf_mask = torch.isinf(inputs.get_student_logits())
        logprobs = F.log_softmax(inputs.get_student_logits(), dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(probs * logprobs, inf_mask, 0)

        x = torch.sum(prod_probs[..., :vocab] - teacher_prod_probs, dim=-1).view(-1)
        mask = (inputs.labels != -100).int()  # [B, 512]
        logits_reversed_kld_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return logits_reversed_kld_loss


class HiddenStatesKLDivergenceLoss(DistillationLoss):
    """Layer-wise KLD Distillation Loss for Knowledge Distillation."""

    loss_type = "hidden-states-layerwise-kld"
    require_adaptaion_layer = True

    @override
    def compute_disitill_loss(self, inputs: DistillLossInputs):
        teacher_hidden_states = inputs.get_teacher_hidden_state()
        student_hidden_states = inputs.get_student_hidden_state()

        assert self.adaptation_layer is not None, "adaptation_layer is required"
        adaptation_layer = self.adaptation_layer.to(student_hidden_states[0].device)
        adapted_student_hidden_states = adaptation_layer(student_hidden_states)

        total_loss_kd = 0
        for student_hidden_idx, teacher_idx in adaptation_layer.layer_mapping.items():
            teacher_hidden = teacher_hidden_states[teacher_idx]

            if adapted_student_hidden_states[student_hidden_idx].shape != teacher_hidden.shape:
                raise ValueError(
                    f"Shape mismatch: student {adapted_student_hidden_states[student_hidden_idx].shape} vs teacher {teacher_hidden.shape}"
                )

            teacher_probs = F.softmax(teacher_hidden / self.temperature, dim=-1)

            loss_kd = F.kl_div(
                F.log_softmax(adapted_student_hidden_states[student_hidden_idx] / self.temperature, dim=-1),
                teacher_probs,
                reduction="batchmean",
            ) * (self.temperature**2)

            total_loss_kd += loss_kd

        avg_loss_kd = total_loss_kd / len(adaptation_layer.layer_mapping)
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        layerwise_loss_kld = avg_loss_kd / hidden_dim

        return layerwise_loss_kld

    @override
    def fuse_loss(self, distill_loss, student_loss):
        return student_loss + self.hidden_states_alpha * distill_loss


class HiddenStatesMSELoss(DistillationLoss):
    """Layer-wise Distillation MSE Loss for Knowledge Distillation."""

    loss_type = "hidden-states-layerwise-mse"
    require_adaptaion_layer = True

    @override
    def compute_disitill_loss(self, inputs: DistillLossInputs):
        teacher_hidden_states = inputs.get_teacher_hidden_state()
        student_hidden_states = inputs.get_student_hidden_state()

        assert self.adaptation_layer is not None, "adaptation_layer is required"
        adaptation_layer = self.adaptation_layer.to(student_hidden_states[0].device)
        adapted_student_hidden_states = adaptation_layer(student_hidden_states)

        # teacher_layer_num = len(teacher_hidden_states)
        student_layer_num = len(student_hidden_states)

        layerwise_loss_mse = 0.0
        for student_hidden_idx, teacher_idx in adaptation_layer.layer_mapping.items():
            teacher_hidden = teacher_hidden_states[teacher_idx]

            if adapted_student_hidden_states[student_hidden_idx].shape != teacher_hidden.shape:
                raise ValueError(
                    f"Shape mismatch: student {adapted_student_hidden_states[student_hidden_idx].shape} vs teacher {teacher_hidden.shape}"
                )

            layerwise_loss_mse += F.mse_loss(
                adapted_student_hidden_states[student_hidden_idx].float().clone(),
                teacher_hidden.detach().float().clone(),
                reduction="mean",
            )
        layerwise_loss_mse = layerwise_loss_mse / student_layer_num

        return layerwise_loss_mse

    @override
    def fuse_loss(self, distill_loss, student_loss):
        return student_loss + self.hidden_states_alpha * distill_loss


class LogitsAndHiddenStatesKLDivergenceLoss(DistillationLoss):
    """Layer-wise Distillation Loss for Knowledge Distillation."""

    loss_type = "logits-and-hidden-states-kld"
    require_adaptaion_layer = (
        LogitsKLDivergenceLoss.require_adaptaion_layer | HiddenStatesKLDivergenceLoss.require_adaptaion_layer
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logits_loss_fn = LogitsKLDivergenceLoss(*args, **kwargs)
        self.hidden_states_loss_fn = HiddenStatesKLDivergenceLoss(*args, **kwargs)

    @override
    def compute_disitill_loss(self, inputs: DistillLossInputs):
        logits_loss = self.logits_loss_fn.compute_disitill_loss(inputs)
        hidden_states_loss = self.hidden_states_loss_fn.compute_disitill_loss(inputs)
        return (1.0 - self.distill_alpha) * logits_loss + self.hidden_states_alpha * hidden_states_loss

    @override
    def fuse_loss(self, distill_loss, student_loss):
        return self.distill_alpha * student_loss + distill_loss


class DistillLossFactory:
    LOSS_CLASS: List[Type[DistillationLoss]] = [
        LogitsKLDivergenceLoss,
        LogitsKLDivergenceV2Loss,
        LogitsReversedKLDivergenceLoss,
        HiddenStatesKLDivergenceLoss,
        HiddenStatesMSELoss,
        LogitsAndHiddenStatesKLDivergenceLoss,
    ]

    @classmethod
    def create(cls, loss_type: str, **kwargs) -> DistillationLoss:
        for loss_cls in cls.LOSS_CLASS:
            if loss_cls.loss_type == loss_type:
                return loss_cls(**kwargs)

        raise ValueError(f"Distillation loss type '{loss_type}' is not supported.")

    @classmethod
    def from_args(cls, finetuning_args: "FinetuningArguments", model_args: "ModelArguments"):
        logger.info(f"Distillation method: {finetuning_args.distill_loss_type}")
        return cls.create(
            finetuning_args.distill_loss_type,
            temperature=finetuning_args.distill_temperature,
            max_length=model_args.model_max_length,
            hidden_states_alpha=finetuning_args.hidden_states_alpha,
            distill_alpha=finetuning_args.distill_alpha,
        )
