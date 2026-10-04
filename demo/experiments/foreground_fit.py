"""Tiny diagnostic optimizer. It never starts the campaign trainer's 80-epoch loop."""

from __future__ import annotations

import time

import torch

from demo.models.foreground_objective import evaluation_metrics, smoke_hand_objective
from demo.models.hand_objective import hand_step_loss

STEP_CAP = 120
SECOND_CAP = 120
EVAL_STEPS = (0, 20, 60, 120)


def optimize_arm(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    objective: str,
    steps: int,
    seconds: float,
    seed: int = 1234,
) -> dict:
    """Run one arm until ``steps`` updates or ``seconds`` elapse, whichever is first."""
    if steps < 0 or steps > STEP_CAP or seconds <= 0 or seconds > SECOND_CAP:
        raise ValueError("fit arm exceeds the 120-step or 120-second cap")
    if objective not in {"legacy", "corrected"}:
        raise ValueError("objective must be legacy or corrected")
    if not batches:
        raise ValueError("empty batches")
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    history = []
    started = time.perf_counter()
    completed = 0

    def evaluate(step: int) -> dict:
        model.eval()
        totals = []
        with torch.no_grad():
            for batch in batches:
                pred = model(batch["input"])
                metrics = evaluation_metrics(pred[:, :3], pred[:, 3:4], batch["source_rgb"], batch["target_alpha"])
                totals.append(metrics)
        keys = totals[0].keys()
        mean = {key: sum(item[key] for item in totals) / len(totals) for key in keys}
        if not all(value == value for value in mean.values()):
            raise ValueError("nonfinite evaluation")
        mean["step"] = step
        mean["elapsed_s"] = time.perf_counter() - started
        return mean

    while True:
        if completed in EVAL_STEPS or completed == steps:
            history.append(evaluate(completed))
        if completed >= steps or time.perf_counter() - started >= seconds:
            break
        batch = batches[completed % len(batches)]
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(batch["input"])
        if objective == "legacy":
            target = torch.cat((batch["source_rgb"], batch["target_alpha"]), dim=1)
            loss, _, _ = hand_step_loss(pred, target, None)
        else:
            loss, _parts = smoke_hand_objective(pred, batch["source_rgb"], batch["target_alpha"])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise ValueError("nonfinite gradient")
        opt.step()
        completed += 1
    return {
        "objective": objective,
        "completed_steps": completed,
        "elapsed_s": time.perf_counter() - started,
        "device": str(device),
        "history": history,
        "legacy_ablation": objective == "legacy",
    }
