from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from ...extras.logging import get_logger
from ...hparams import FinetuningArguments, ModelArguments
from ...model import load_model, load_tokenizer
from ..sft.trainer import CustomSeq2SeqTrainer
from .loss import DistillationLoss, DistillLossFactory, DistillLossInputs
from .utils import disable_deepspeed_zero3


logger = get_logger(__name__)


class AdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList(
            [torch.nn.Linear(student_dim, teacher_dim, dtype=dtype) for _ in range(num_student_layers)]
        )
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers):
        return {i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1)) for i in range(num_student_layers)}

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states


class DistillationTrainer(CustomSeq2SeqTrainer):
    def __init__(self, finetuning_args: "FinetuningArguments", model_args: "ModelArguments" = None, *args, **kwargs) -> None:
        model_args = kwargs.pop("model_args", model_args)
        # CustomSeq2SeqTrainer expects (finetuning_args, processor, model_args=...); we pass processor=None
        super().__init__(finetuning_args, None, model_args=model_args, *args, **kwargs)
        self.teacher_parallel = finetuning_args.teacher_parallel
        self.output_hidden_states = False

        self.distill_loss: DistillationLoss = DistillLossFactory.from_args(finetuning_args, model_args)
        self.teacher_models = self._init_teacher_model(finetuning_args, model_args)

    def _init_teacher_model(
        self, finetuning_args: "FinetuningArguments", model_args: "ModelArguments"
    ) -> List["PreTrainedModel"]:
        teachers = []
        for teacher_path in finetuning_args.teacher_models:
            logger.info(f"loading teacher model, path: {teacher_path}")

            teacher_model_args = ModelArguments(model_name_or_path=teacher_path)
            teacher_model_args.flash_attn = model_args.flash_attn
            teacher_model_args.compute_dtype = model_args.compute_dtype
            teacher_model_args.enable_liger_kernel = model_args.enable_liger_kernel
            teacher_model_args.enable_liger_fused_ce = model_args.enable_liger_fused_ce

            if finetuning_args.teacher_parallel == "no":
                tokenizer = load_tokenizer(teacher_model_args)["tokenizer"]
                model = load_model(
                    tokenizer, teacher_model_args, finetuning_args, is_trainable=False, add_valuehead=False
                )
                # NOTE(wuhuan): adapt deepspeed zero3 stage, in case of weight shape mismatch
                model = Accelerator().prepare(model)
                model.eval()
            elif finetuning_args.teacher_parallel == "pp":
                if (rank := dist.get_rank()) == 0:
                    # disable zero3.init, use HF's pipeline parallel initialization.
                    with disable_deepspeed_zero3(self.args):
                        model = AutoModelForCausalLM.from_pretrained(
                            teacher_path,
                            device_map="auto",
                            torch_dtype=model_args.compute_dtype,
                        )
                        model.eval()
                else:
                    logger.info(f"skip load teacher model for rank {rank}")
                    continue
            else:
                raise NotImplementedError(f"{finetuning_args.teacher_parallel=} is not supported.")

            logger.info("teacher model is loaded")
            teachers.append(model)

            if self.distill_loss.require_adaptaion_layer:
                logger.info(
                    "Multi-teachers do not support hidden-states-based layerwise distill. We will use first teacher."
                )
                device = Accelerator().device
                self.output_hidden_states = True

                teacher_config = AutoConfig.from_pretrained(teacher_path)
                student_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
                adaptation_layer = AdaptationLayer(
                    student_config.hidden_size,
                    teacher_config.hidden_size,
                    student_config.num_hidden_layers,
                    teacher_config.num_hidden_layers,
                    dtype=torch.bfloat16,
                ).to(device)
                self.distill_loss.adaptation_layer = Accelerator().prepare(adaptation_layer)
                break

        return teachers

    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """Compute loss of distillation training.

        Parameters
        ----------
        model : PreTrainedModel
            student model
        inputs : Dict[str, Any]
            input data
        return_outputs : bool
        """

        # compute student output
        outputs_student = model(**inputs, output_hidden_states=self.output_hidden_states)

        @torch.inference_mode()
        def infer_teacher(_inputs):
            all_teacher_outputs = []
            for teacher in self.teacher_models:
                outputs_teacher = teacher(**_inputs, output_hidden_states=self.output_hidden_states)
                all_teacher_outputs.append(outputs_teacher)
            return all_teacher_outputs

        if self.teacher_parallel == "pp":
            # FIXME "pp" mode hasn't been tested. Default is "no".
            raise NotImplementedError("pp mode hasn't been implemented yet, please use 'no'")
            # device = inputs["input_ids"].device
            # micro_seq_len, vocab = outputs_student.logits.shape[1:]
            # global_max_len = get_global_max_seq_len(inputs["input_ids"])
            # gathered_batch_encoding = gather_inputs(inputs, global_max_len)
            # logits_shape = [inputs["input_ids"].shape[0], global_max_len, vocab]
            # if dist.get_rank() == 0:
            #     outputs_teachers = infer_teacher(gathered_batch_encoding)
            #     outputs_teachers = scatter_outputs(outputs_teachers, logits_shape, device)
            # else:
            #     outputs_teachers = scatter_outputs(None, logits_shape, device)
            # outputs_teachers = outputs_teachers[:, :micro_seq_len]
        else:
            outputs_teachers = infer_teacher(inputs)

        loss = self.distill_loss(
            DistillLossInputs(
                outputs_student=outputs_student,
                outputs_teachers=outputs_teachers,
                labels=inputs["labels"],
            )
        )
        return (loss, outputs_student) if return_outputs else loss
