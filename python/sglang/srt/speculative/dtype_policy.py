from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class SpeculativeDtypePolicy:
    """Resolved dtype ownership for hidden-state speculative decoding.

    Target hidden states are produced by the target model, then consumed inside
    the draft model. Keep each model's dtype unchanged and cast only at that
    ownership boundary.
    """

    target_hidden_dtype: Optional[torch.dtype]
    draft_hidden_dtype: Optional[torch.dtype]
    cast_target_hidden_states: bool = True
    logger: Optional[logging.Logger] = field(default=None, repr=False, compare=False)
    algorithm: str = ""
    _cast_warning_emitted: bool = field(default=False, init=False, repr=False)

    @property
    def needs_target_hidden_cast(self) -> bool:
        return (
            self.cast_target_hidden_states
            and self.target_hidden_dtype is not None
            and self.draft_hidden_dtype is not None
            and self.target_hidden_dtype != self.draft_hidden_dtype
        )

    def prepare_target_hidden_for_draft(
        self, hidden_states: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if (
            not self.cast_target_hidden_states
            or hidden_states is None
            or self.draft_hidden_dtype is None
        ):
            return hidden_states
        if hidden_states.dtype == self.draft_hidden_dtype:
            return hidden_states
        self._warn_target_hidden_cast(hidden_states.dtype)
        return hidden_states.to(dtype=self.draft_hidden_dtype)

    def _warn_target_hidden_cast(self, source_dtype: torch.dtype) -> None:
        if self.logger is None or self._cast_warning_emitted:
            return
        self._cast_warning_emitted = True
        self.logger.warning(
            f"Speculative decoding dtype mismatch for {self.algorithm}: casting "
            f"target hidden states from {source_dtype} to {self.draft_hidden_dtype} "
            "before draft forward."
        )


def build_speculative_dtype_policy(
    *,
    target_dtype: Optional[torch.dtype],
    draft_dtype: Optional[torch.dtype],
    logger: logging.Logger,
    algorithm: str,
    cast_target_hidden_states: bool = True,
) -> SpeculativeDtypePolicy:
    return SpeculativeDtypePolicy(
        target_hidden_dtype=target_dtype,
        draft_hidden_dtype=draft_dtype,
        cast_target_hidden_states=cast_target_hidden_states,
        logger=logger,
        algorithm=algorithm,
    )


def align_draft_input_embeds(
    embeds: torch.Tensor, hidden_states: Optional[torch.Tensor]
) -> torch.Tensor:
    """Align shared target embedding output at model-specific draft fusion sites."""
    if hidden_states is None or embeds.dtype == hidden_states.dtype:
        return embeds
    return embeds.to(dtype=hidden_states.dtype)
