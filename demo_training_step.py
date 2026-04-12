"""
demo_training_step.py

A basic (fast) sanity check that PolyTensor can run through:
  - F.linear
  - F.cross_entropy (label_smoothing supported)
  - softmax / logsumexp / exp / log
and can be used to propagate a polynomial through a single manual training step.

Run:
  python demo_training_step.py
"""

import torch
import torch.nn.functional as F

from polytensor import PolyTensor


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)


def step_linear_classifier(W: torch.Tensor, b: torch.Tensor, X: torch.Tensor, y: torch.Tensor, lr: float, label_smoothing: float):
    """
    One GD-like step for a linear classifier with cross-entropy loss.
    Gradient is computed manually (no autograd).
    """
    logits = F.linear(X, W, b)
    probs = torch.softmax(logits, dim=1)

    # label smoothing in the gradient:
    # y_smooth = (1-eps) * one_hot + eps * uniform
    C = logits.shape[1]
    oh = one_hot(y, C)
    if label_smoothing != 0.0:
        oh = (1.0 - label_smoothing) * oh + label_smoothing * (1.0 / C)

    grad_logits = (probs - oh) / logits.shape[0]  # mean reduction over batch
    grad_W = grad_logits.T @ X
    grad_b = grad_logits.sum(dim=0)

    return W - lr * grad_W, b - lr * grad_b


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tiny synthetic batch
    N, D, C = 8, 5, 4
    X = torch.randn(N, D, device=device)
    y = torch.randint(0, C, (N,), device=device)

    lr = 0.3
    label_smoothing = 0.2
    deg = 2

    W0 = torch.randn(C, D, device=device) * 0.1
    b0 = torch.randn(C, device=device) * 0.1

    # direction (tangent) we differentiate along
    dW = torch.randn_like(W0) * 0.01
    db = torch.randn_like(b0) * 0.01

    W = PolyTensor.from_primal(W0, degree=deg, tangent=dW)
    b = PolyTensor.from_primal(b0, degree=deg, tangent=db)

    logits = F.linear(X, W, b)
    loss = F.cross_entropy(logits, y, reduction="mean", label_smoothing=label_smoothing)

    # manual gradient as a PolyTensor (depends on W,b through logits)
    probs = torch.softmax(logits, dim=1)
    oh = one_hot(y, C).to(device=device)
    if label_smoothing != 0.0:
        oh = (1.0 - label_smoothing) * oh + label_smoothing * (1.0 / C)
    grad_logits = (probs - oh) / N
    grad_W = grad_logits.T @ X
    grad_b = grad_logits.sum(dim=0)

    W_new = W - lr * grad_W
    b_new = b - lr * grad_b

    # finite-difference check for the *updated* parameters' tangent (derivative at t=0)
    h = 1e-4
    Wp, bp = step_linear_classifier(W0 + h * dW, b0 + h * db, X, y, lr, label_smoothing)
    Wm, bm = step_linear_classifier(W0 - h * dW, b0 - h * db, X, y, lr, label_smoothing)
    dW_fd = (Wp - Wm) / (2 * h)
    db_fd = (bp - bm) / (2 * h)

    # finite-difference check for loss directional derivative
    def loss_plain(W_plain, b_plain):
        logits_plain = F.linear(X, W_plain, b_plain)
        return F.cross_entropy(logits_plain, y, reduction="mean", label_smoothing=label_smoothing)

    lp = loss_plain(W0 + h * dW, b0 + h * db)
    lm = loss_plain(W0 - h * dW, b0 - h * db)
    dloss_fd = (lp - lm) / (2 * h)

    # print diagnostics
    print("device:", device)
    print("loss coeffs:", loss.coeffs.detach().cpu())
    print("loss directional-derivative abs err:", float((loss.coeff(1) - dloss_fd).abs().max().cpu()))

    print("W_new primal abs err:", float((W_new.coeff(0) - step_linear_classifier(W0, b0, X, y, lr, label_smoothing)[0]).abs().max().cpu()))
    print("b_new primal abs err:", float((b_new.coeff(0) - step_linear_classifier(W0, b0, X, y, lr, label_smoothing)[1]).abs().max().cpu()))

    print("W_new tangent abs err:", float((W_new.coeff(1) - dW_fd).abs().max().cpu()))
    print("b_new tangent abs err:", float((b_new.coeff(1) - db_fd).abs().max().cpu()))

    # show a couple coefficients
    print("W_new coeff[0] sample:", W_new.coeff(0).flatten()[:5].detach().cpu())
    print("W_new coeff[1] sample:", W_new.coeff(1).flatten()[:5].detach().cpu())
    print("W_new coeff[2] sample:", W_new.coeff(2).flatten()[:5].detach().cpu())


if __name__ == "__main__":
    main()
