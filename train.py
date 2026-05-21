from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

from model import FSCDAN, dann_alpha


def _move_batch(batch, device):
    xc, xp, xt, y, aux = batch
    return (
        xc.to(device),
        xp.to(device),
        xt.to(device),
        y.to(device),
        None if aux is None else aux.to(device),
    )


def _align_batch_size(src, tgt):
    n = min(src[0].shape[0], tgt[0].shape[0])
    src = tuple(x[:n] if isinstance(x, torch.Tensor) else x for x in src)
    tgt = tuple(x[:n] if isinstance(x, torch.Tensor) else x for x in tgt)
    return src, tgt


def _domain_weight(epoch: int, alpha: float, domain_lambda: float, domain_warmup_epochs: int) -> float:
    warmup = max(int(domain_warmup_epochs), 0)
    if warmup > 0:
        scale = min(float(epoch + 1) / float(warmup), 1.0)
    else:
        scale = 1.0
    return float(domain_lambda) * scale * float(alpha)


def evaluate(model: FSCDAN, loader, device, classification_threshold: float) -> Dict[str, float]:
    model.eval()
    device = torch.device(device)
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            xc, xp, xt, y, aux = _move_batch(batch, device)
            logits = model((xc, xp, xt, aux))
            prob = torch.sigmoid(logits)
            preds.append(prob.cpu().numpy())
            targets.append(y.cpu().numpy())

    pred = np.concatenate(preds, axis=0).reshape(-1)
    true = np.concatenate(targets, axis=0).reshape(-1)
    pred_label = (pred >= classification_threshold).astype(np.int32)
    true_label = (true >= 0.5).astype(np.int32)

    return {
        "f1_macro": f1_score(true_label, pred_label, average="macro", zero_division=0),
        "precision": precision_score(true_label, pred_label, zero_division=0),
        "recall": recall_score(true_label, pred_label, zero_division=0),
    }


def train_fscdan(
    model: FSCDAN,
    source_loader,
    target_loader,
    val_loader,
    device,
    optimizer,
    num_epochs: int,
    early_stop_patience: int,
    classification_threshold: float,
    domain_lambda: float,
    domain_warmup_epochs: int,
    target_supervision_weight: float,
) -> Dict[str, object]:
    device = torch.device(device)
    model = model.to(device)

    best_score = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        tgt_iter = iter(target_loader)
        alpha = dann_alpha(epoch, num_epochs)
        domain_weight = _domain_weight(epoch, alpha, domain_lambda, domain_warmup_epochs)

        for src_batch in source_loader:
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_loader)
                tgt_batch = next(tgt_iter)

            src = _move_batch(src_batch, device)
            tgt = _move_batch(tgt_batch, device)
            src, tgt = _align_batch_size(src, tgt)
            src_xc, src_xp, src_xt, src_y, src_aux = src
            tgt_xc, tgt_xp, tgt_xt, tgt_y, tgt_aux = tgt

            src_pred, tgt_pred, src_domain, tgt_domain = model(
                (src_xc, src_xp, src_xt, src_aux),
                (tgt_xc, tgt_xp, tgt_xt, tgt_aux),
                alpha=alpha,
            )

            task_loss_src = F.binary_cross_entropy_with_logits(src_pred, src_y)
            task_loss_tgt = F.binary_cross_entropy_with_logits(tgt_pred, tgt_y)
            src_domain_y = torch.ones_like(src_domain)
            tgt_domain_y = torch.zeros_like(tgt_domain)
            domain_loss = (
                F.binary_cross_entropy_with_logits(src_domain, src_domain_y)
                + F.binary_cross_entropy_with_logits(tgt_domain, tgt_domain_y)
            ) / 2.0
            loss = (
                task_loss_src
                + target_supervision_weight * task_loss_tgt
                + domain_weight * domain_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        metrics = evaluate(model, val_loader, device, classification_threshold)
        score = metrics["f1_macro"]
        print(
            f"Epoch {epoch + 1:03d} | loss={total_loss / max(len(source_loader), 1):.4f} "
            f"| F1-Macro={score:.4f} | Precision={metrics['precision']:.4f} "
            f"| Recall={metrics['recall']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}. Best F1-Macro={best_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_f1_macro": best_score, "state_dict": best_state}
