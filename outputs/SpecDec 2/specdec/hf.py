"""Optional Hugging Face / PyTorch adapter for real transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .distributions import Distribution


class MissingDependencyError(RuntimeError):
    pass


@dataclass
class HuggingFaceCausalLM:
    """Causal LM adapter implementing the LanguageModel protocol.

    Dependencies are imported lazily so the core package remains testable
    without torch/transformers installed.
    """

    model_name: str
    device: str = "cpu"
    dtype: str | None = None

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - exercised in integration envs.
            raise MissingDependencyError(
                "Install the 'hf' extra to use HuggingFaceCausalLM: "
                "pip install -e '.[hf]'"
            ) from exc

        torch_dtype = None
        if self.dtype is not None:
            torch_dtype = getattr(torch, self.dtype)

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()
        self._vocab_size = int(self.model.config.vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("prompt must encode to at least one token")
        return [int(token) for token in ids]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids))

    def next_token_distribution(self, prefix: Sequence[int]) -> Distribution:
        return self.batch_next_token_distributions(prefix, [])[0]

    def batch_next_token_distributions(
        self, prefix: Sequence[int], continuation: Sequence[int]
    ) -> list[Distribution]:
        if not prefix:
            raise ValueError("causal LM scoring requires a non-empty prefix")
        ids = list(prefix) + list(continuation)
        torch = self._torch
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids).logits[0]
            start = len(prefix) - 1
            stop = len(prefix) + len(continuation)
            selected = logits[start:stop]
            probs = torch.softmax(selected.float(), dim=-1).cpu().tolist()
        return [[float(x) for x in row] for row in probs]
