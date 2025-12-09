import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.slot_attn import MultiHeadSTEVESA


@dataclass
class SlotJEPAOutput:
    teacher_slots: torch.Tensor
    teacher_attn: torch.Tensor
    student_slots: torch.Tensor
    student_attn: torch.Tensor


class SlotJEPATeacherStudent(nn.Module):
    """
    Wrapper coordinating a student slot-attention module with its EMA teacher.

    The teacher branch receives the full feature map and produces slot
    assignments. These assignments guide the student (which attends only to
    unmasked tokens) by providing an attention override during the first
    iteration. Slot noise is shared across branches to keep the priors aligned.
    """

    def __init__(
        self,
        student: MultiHeadSTEVESA,
        *,
        momentum: float = 0.996,
        ema_warmup_steps: int = 0,
        guided_grad_substitute: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(student, MultiHeadSTEVESA):
            raise TypeError("student must be an instance of MultiHeadSTEVESA.")
        if not (0.0 < momentum <= 1.0):
            raise ValueError("momentum must be in (0, 1].")

        self.student = student
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.register_buffer("_ema_step", torch.tensor(0, dtype=torch.long), persistent=False)
        self.momentum = float(momentum)
        self.ema_warmup_steps = int(max(0, ema_warmup_steps))
        self.guided_grad_substitute = bool(guided_grad_substitute)

    @torch.no_grad()
    def reset_teacher(self) -> None:
        """Synchronise teacher weights with the student."""
        self.teacher.load_state_dict(self.student.state_dict())
        self._ema_step.zero_()

    def _current_momentum(self, step: Optional[int] = None) -> float:
        if self.ema_warmup_steps <= 0:
            return self.momentum
        if step is None:
            step = int(self._ema_step.item())
        warmup_progress = min(float(step) / float(max(1, self.ema_warmup_steps)), 1.0)
        return float(self.momentum * warmup_progress)

    @torch.no_grad()
    def update_teacher(self, step: Optional[int] = None, momentum_override: Optional[float] = None) -> None:
        """
        Momentum update for teacher parameters. Should be called after each
        optimiser step.
        """
        momentum = float(momentum_override) if momentum_override is not None else self._current_momentum(step)
        if momentum <= 0.0:
            self.reset_teacher()
            return

        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        self._ema_step.add_(1)

    def sample_slot_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.student.num_slots,
            self.student.slot_size,
            device=device,
            dtype=dtype,
        )

    def _sample_slot_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.sample_slot_noise(batch_size, device, dtype)

    def forward(
        self,
        teacher_inputs: torch.Tensor,
        student_inputs: torch.Tensor,
        *,
        valid_token_mask: Optional[torch.Tensor] = None,
        slot_noise: Optional[torch.Tensor] = None,
        teacher_iterations: Optional[int] = None,
        student_iterations: Optional[int] = None,
    ) -> SlotJEPAOutput:
        """
        Args:
            teacher_inputs: Feature tensor for the teacher branch [B, C, H, W].
            student_inputs: Feature tensor for the student branch (masked tokens).
            valid_token_mask: Optional boolean tensor [B, N] marking tokens that
                remain visible to the student. When provided, tokens masked out
                (False) do not contribute to the student's updates.
            slot_noise: Optional noise tensor shared across teacher/student.
            teacher_iterations / student_iterations: Override the default number
                of slot-attention iterations per branch.
        """
        if teacher_inputs.shape != student_inputs.shape:
            raise ValueError("teacher_inputs and student_inputs must share the same shape.")

        B = teacher_inputs.shape[0]
        device = teacher_inputs.device
        dtype = teacher_inputs.dtype
        if slot_noise is None:
            slot_noise = self._sample_slot_noise(B, device=device, dtype=dtype)
        else:
            if slot_noise.shape != (B, self.student.num_slots, self.student.slot_size):
                raise ValueError(
                    "slot_noise must have shape [B, num_slots, slot_size]; "
                    f"received {tuple(slot_noise.shape)}"
                )

        with torch.no_grad():
            teacher_slots, teacher_attn = self.teacher.forward_slots(
                teacher_inputs,
                slot_noise=slot_noise,
                num_iterations=teacher_iterations,
            )

        student_slots, student_attn = self.student.forward_slots(
            student_inputs,
            slot_noise=slot_noise,
            attn_override=teacher_attn.detach(),
            valid_token_mask=valid_token_mask,
            guided_grad_substitute=self.guided_grad_substitute,
            num_iterations=student_iterations,
        )

        return SlotJEPAOutput(
            teacher_slots=teacher_slots.detach(),
            teacher_attn=teacher_attn.detach(),
            student_slots=student_slots,
            student_attn=student_attn,
        )
