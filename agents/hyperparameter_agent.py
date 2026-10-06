from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch.optim as optim


@dataclass
class HyperparameterAgentConfig:
    lr_min: float = 1e-5
    lr_max: float = 5e-4
    lr_decay_factor: float = 0.5
    lr_grow_factor: float = 1.1
    batch_size_min: int = 32
    batch_size_max: int = 256
    batch_size_step: int = 32
    imbalance_ratio: float = 1.5
    severe_imbalance_ratio: float = 2.0
    cooldown_steps: int = 3
    history_window: int = 5


@dataclass
class HyperparameterAction:
    learning_rate: float
    batch_size: int
    reason: str
    changed: bool


@dataclass
class HyperparameterAgent:
    config: HyperparameterAgentConfig = field(default_factory=HyperparameterAgentConfig)
    current_lr: float = 2e-4
    current_batch_size: int = 128
    _step: int = 0
    _last_change_step: int = -1000
    _g_history: List[float] = field(default_factory=list)
    _d_history: List[float] = field(default_factory=list)

    def update(
        self,
        generator_loss: float,
        discriminator_loss: float,
    ) -> HyperparameterAction:
        self._step += 1
        self._append_history(generator_loss, discriminator_loss)

        if not self._enough_history():
            return HyperparameterAction(
                learning_rate=self.current_lr,
                batch_size=self.current_batch_size,
                reason="Warmup history collection",
                changed=False,
            )

        if self._in_cooldown():
            return HyperparameterAction(
                learning_rate=self.current_lr,
                batch_size=self.current_batch_size,
                reason="Cooldown active",
                changed=False,
            )

        g_avg = sum(self._g_history) / len(self._g_history)
        d_avg = sum(self._d_history) / len(self._d_history)
        ratio = (g_avg + 1e-8) / (d_avg + 1e-8)

        reason = "Balanced losses"
        changed = False

        if ratio > self.config.severe_imbalance_ratio:
            self._decrease_lr()
            self._decrease_batch_size()
            reason = "Generator loss much higher than discriminator; stabilizing updates"
            changed = True
        elif ratio > self.config.imbalance_ratio:
            self._decrease_lr()
            reason = "Generator lagging; reducing learning rate"
            changed = True
        elif ratio < 1.0 / self.config.severe_imbalance_ratio:
            self._increase_lr()
            self._increase_batch_size()
            reason = "Discriminator loss much higher than generator; strengthening updates"
            changed = True
        elif ratio < 1.0 / self.config.imbalance_ratio:
            self._increase_lr()
            reason = "Discriminator lagging; increasing learning rate"
            changed = True

        if changed:
            self._last_change_step = self._step

        return HyperparameterAction(
            learning_rate=self.current_lr,
            batch_size=self.current_batch_size,
            reason=reason,
            changed=changed,
        )

    def apply_learning_rate(
        self,
        optimizer_g: optim.Optimizer,
        optimizer_d: Optional[optim.Optimizer] = None,
    ) -> None:
        self._set_optimizer_lr(optimizer_g, self.current_lr)
        if optimizer_d is not None:
            self._set_optimizer_lr(optimizer_d, self.current_lr)

    def state_dict(self) -> Dict[str, float | int]:
        return {
            "current_lr": self.current_lr,
            "current_batch_size": self.current_batch_size,
            "step": self._step,
            "last_change_step": self._last_change_step,
        }

    def _append_history(self, g_loss: float, d_loss: float) -> None:
        self._g_history.append(float(g_loss))
        self._d_history.append(float(d_loss))
        window = self.config.history_window
        if len(self._g_history) > window:
            self._g_history = self._g_history[-window:]
            self._d_history = self._d_history[-window:]

    def _enough_history(self) -> bool:
        return len(self._g_history) >= self.config.history_window

    def _in_cooldown(self) -> bool:
        return (self._step - self._last_change_step) <= self.config.cooldown_steps

    def _decrease_lr(self) -> None:
        self.current_lr = max(self.config.lr_min, self.current_lr * self.config.lr_decay_factor)

    def _increase_lr(self) -> None:
        self.current_lr = min(self.config.lr_max, self.current_lr * self.config.lr_grow_factor)

    def _decrease_batch_size(self) -> None:
        self.current_batch_size = max(
            self.config.batch_size_min, self.current_batch_size - self.config.batch_size_step
        )

    def _increase_batch_size(self) -> None:
        self.current_batch_size = min(
            self.config.batch_size_max, self.current_batch_size + self.config.batch_size_step
        )

    @staticmethod
    def _set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
