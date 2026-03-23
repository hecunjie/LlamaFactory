# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_jieba_available, is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() == 2:
        return logits  # already argmaxed (e.g. from latent chain eval)

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeExactMatch:
    r"""Compute exact match accuracy for the answer portion.

    In non-generate mode, uses the label mask to extract only the answer
    tokens from predictions (accounting for autoregressive shift).
    Supports numerical comparison for math tasks — extracts the last number
    from both predicted and ground truth answers for robust matching.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            if len(self.score_dict["exact_match"]) > 0:
                result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            else:
                result = {"exact_match": 0.0}

        self.score_dict = {"exact_match": []}
        return result

    def __post_init__(self):
        self._dump()

    @staticmethod
    def _extract_last_number(text: str) -> Optional[float]:
        """Extract the last number from text for numerical comparison."""
        text = text.replace(",", "")
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None

    @staticmethod
    def _compare_answers(pred: str, label: str) -> bool:
        """Compare predicted and ground truth answers.

        First tries exact string match, then numerical comparison.
        """
        if pred == label:
            return True

        # Try numerical comparison for math tasks
        pred_num = ComputeExactMatch._extract_last_number(pred)
        label_num = ComputeExactMatch._extract_last_number(label)
        if pred_num is not None and label_num is not None:
            return abs(pred_num - label_num) < 1e-6

        return False

    @staticmethod
    def _normalize_extracted_answer(text: str) -> str:
        """Normalize extracted answer text for robust exact-match."""
        ans = text.strip()
        ans = ans.strip(" \t\r\n\"'`")
        # Handle trailing JSON-like noise such as: ... "}" or ... }
        ans = re.sub(r'[\"\'}\]\s]+$', "", ans).strip()

        # If answer has explanatory parenthesis, prefer core value.
        # e.g. "6 (buses)" -> "6"
        paren_match = re.match(r"^\s*([^\(\)]+?)\s*\([^)]*\)\s*$", ans)
        if paren_match:
            ans = paren_match.group(1).strip()

        return ans

    @staticmethod
    def _extract_boxed_content(text: str) -> Optional[str]:
        r"""Extract content from the last \boxed{...} block, supports nested braces."""
        key = r"\boxed{"
        start = text.rfind(key)
        if start == -1:
            return None

        i = start + len(key)
        depth = 1
        buf = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(buf).strip()
                buf.append(ch)
            else:
                buf.append(ch)
            i += 1
        return None

    @staticmethod
    def _extract_answer_from_output(text: str) -> str:
        r"""Extract final answer from model output.

        Priority:
        1) last \boxed{...}
        2) after last "####"
        3) after last "The answer is ..."
        4) full stripped text
        """
        text = text.strip()

        boxed = ComputeExactMatch._extract_boxed_content(text)
        if boxed:
            return ComputeExactMatch._normalize_extracted_answer(boxed)

        if "####" in text:
            return ComputeExactMatch._normalize_extracted_answer(text.split("####")[-1].strip())

        answer_is_matches = re.findall(r"The answer is\s*(.+)", text, flags=re.IGNORECASE)
        if answer_is_matches:
            return ComputeExactMatch._normalize_extracted_answer(answer_is_matches[-1].strip())

        return ComputeExactMatch._normalize_extracted_answer(text)

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # Detect generate mode: in generate mode, preds are full generated
        # sequences (including prompt padding) and labels are separately stored.
        # In non-generate mode, preds are argmax logits with same shape as labels.
        is_generate_mode = (preds.shape != labels.shape)

        if is_generate_mode:
            # ---- Generate mode: decode full text, split by #### ----
            # Replace IGNORE_INDEX and pad tokens for clean decoding
            preds_clean = np.where(
                preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id
            )
            labels_clean = np.where(
                labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id
            )

            for i in range(len(preds_clean)):
                decoded_pred = self.tokenizer.decode(
                    preds_clean[i], skip_special_tokens=True
                ).strip()
                decoded_label = self.tokenizer.decode(
                    labels_clean[i], skip_special_tokens=True
                ).strip()

                pred_answer = self._extract_answer_from_output(decoded_pred)
                label_answer = self._extract_answer_from_output(decoded_label)

                match = self._compare_answers(pred_answer, label_answer)
                self.score_dict["exact_match"].append(1.0 if match else 0.0)

                # Debug: log first 3 samples per eval call to diagnose
                if i < 3:
                    import logging as _logging
                    _logger = _logging.getLogger(__name__)
                    _has_delim_pred = "####" in decoded_pred
                    _has_delim_label = "####" in decoded_label
                    _has_boxed_pred = r"\boxed{" in decoded_pred
                    _has_boxed_label = r"\boxed{" in decoded_label
                    _logger.info(
                        f"[exact_match generate] sample={i} match={match}\n"
                        f"  pred_answer='{pred_answer}' (has_boxed={_has_boxed_pred}, has_####={_has_delim_pred})\n"
                        f"  label_answer='{label_answer}' (has_boxed={_has_boxed_label}, has_####={_has_delim_label})\n"
                        f"  decoded_pred[:200]='{decoded_pred[:200]}'\n"
                        f"  decoded_label[:200]='{decoded_label[:200]}'"
                    )
        else:
            # ---- Non-generate mode: logits argmax comparison ----
            for i in range(len(preds)):
                # Handle autoregressive shift: preds[j] predicts labels[j+1]
                pred = preds[i, :-1]
                label = labels[i, 1:]

                # Only keep answer positions where labels are valid (not IGNORE_INDEX).
                label_mask = label != IGNORE_INDEX
                pred_answer_ids = pred[label_mask]
                label_answer_ids = label[label_mask]

                if len(label_answer_ids) == 0:
                    self.score_dict["exact_match"].append(0.0)
                    continue

                # Decode to strings
                decoded_pred = self.tokenizer.decode(
                    pred_answer_ids, skip_special_tokens=True
                ).strip()
                decoded_label = self.tokenizer.decode(
                    label_answer_ids, skip_special_tokens=True
                ).strip()

                pred_answer = self._extract_answer_from_output(decoded_pred)
                label_answer = self._extract_answer_from_output(decoded_label)

                match = self._compare_answers(pred_answer, label_answer)
                self.score_dict["exact_match"].append(1.0 if match else 0.0)

        if compute_result:
            return self._dump()

@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
