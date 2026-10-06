from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StopDecision:
    should_stop: bool
    reason: str
    best_fid: float
    epochs_without_improvement: int


@dataclass
class StopAgentConfig:
    patience: int = 5
    min_improvement: float = 0.5
    warmup_epochs: int = 3
    target_fid: Optional[float] = None


@dataclass
class StopAgent:
    config: StopAgentConfig = field(default_factory=StopAgentConfig)
    best_fid: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def update(self, epoch: int, fid: float) -> StopDecision:
        if fid < 0:
            raise ValueError(f"FID must be non-negative, got {fid}")

        if epoch <= self.config.warmup_epochs:
            self._update_best(epoch, fid, allow_small=True)
            return StopDecision(
                should_stop=False,
                reason="Warmup period",
                best_fid=self.best_fid,
                epochs_without_improvement=self.epochs_without_improvement,
            )

        improved = (self.best_fid - fid) >= self.config.min_improvement
        if improved:
            self._update_best(epoch, fid, allow_small=False)
            reason = "FID improved"
        else:
            self.epochs_without_improvement += 1
            reason = "No significant FID improvement"

        if self.config.target_fid is not None and fid <= self.config.target_fid:
            return StopDecision(
                should_stop=True,
                reason=f"Target FID reached ({fid:.4f} <= {self.config.target_fid:.4f})",
                best_fid=self.best_fid,
                epochs_without_improvement=self.epochs_without_improvement,
            )

        if self.epochs_without_improvement >= self.config.patience:
            return StopDecision(
                should_stop=True,
                reason=f"Early stop: no FID improvement for {self.config.patience} epochs",
                best_fid=self.best_fid,
                epochs_without_improvement=self.epochs_without_improvement,
            )

        return StopDecision(
            should_stop=False,
            reason=reason,
            best_fid=self.best_fid,
            epochs_without_improvement=self.epochs_without_improvement,
        )

    def reset(self) -> None:
        self.best_fid = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def state_dict(self) -> Dict[str, float | int]:
        return {
            "best_fid": self.best_fid,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

    def _update_best(self, epoch: int, fid: float, allow_small: bool) -> None:
        if allow_small:
            should_update = fid < self.best_fid
        else:
            should_update = (self.best_fid - fid) >= self.config.min_improvement

        if should_update:
            self.best_fid = fid
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
