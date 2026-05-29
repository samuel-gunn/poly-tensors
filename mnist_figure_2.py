import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from PolyTensor import PolyTensor


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = "data"
OUTPUT_PATH = "mnist_figure_2.png"
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 1e-3
DOWNWEIGHT_FRACTION = 0.05
DEGREE = 3
NUM_DOTS = 11
SEED = 0


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.GELU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def tensor_value(x):
    return x.value if isinstance(x, PolyTensor) else x


def make_poly_parameters(module, degree):
    for child in module.children():
        make_poly_parameters(child, degree)

    for name, param in list(module._parameters.items()):
        if param is None:
            continue
        coeffs = (param.detach(),) + tuple(torch.zeros_like(param) for _ in range(degree))
        module._parameters[name] = nn.Parameter(
            PolyTensor(coeffs, requires_grad=param.requires_grad),
            requires_grad=param.requires_grad,
        )


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_data(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_data = datasets.MNIST(
        args.data_dir,
        train=True,
        download=not args.no_download,
        transform=transform,
    )
    test_data = datasets.MNIST(
        args.data_dir,
        train=False,
        download=not args.no_download,
        transform=transform,
    )

    if args.max_train_examples is not None:
        subset_generator = make_generator(args.seed + 10)
        subset_indices = torch.randperm(len(train_data), generator=subset_generator)[
            : args.max_train_examples
        ]
        train_data = Subset(train_data, subset_indices.tolist())

    selection_generator = make_generator(args.seed + 20)
    num_deleted = int(round(args.downweight_fraction * len(train_data)))
    num_deleted = min(len(train_data), max(1, num_deleted))
    selected = torch.zeros(len(train_data), dtype=torch.bool)
    selected_indices = torch.randperm(len(train_data), generator=selection_generator)[:num_deleted]
    selected[selected_indices] = True

    if args.test_index is None:
        test_generator = make_generator(args.seed + 30)
        test_index = int(torch.randint(len(test_data), (), generator=test_generator).item())
    else:
        test_index = args.test_index

    test_image, test_label = test_data[test_index]
    return train_data, selected, test_image.unsqueeze(0), torch.tensor([test_label]), test_index


def make_epoch_indices(num_examples, epochs, seed):
    generator = make_generator(seed)
    return [
        torch.randperm(num_examples, generator=generator).tolist()
        for _ in range(epochs)
    ]


def make_model(device, seed, degree=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MNISTClassifier().to(device)
    if degree is not None:
        make_poly_parameters(model, degree)
    return model


def train_one_epoch(model, loader, selected, downweight, loss_fn, optimizer, device, quiet):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, indices in tqdm(loader, disable=quiet, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        selected_batch = selected[indices].to(device)

        logits = model(images)
        per_example_loss = loss_fn(logits, labels)
        weights = 1 - selected_batch.float() * downweight
        loss = (per_example_loss * weights).sum() * (1.0 / images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += tensor_value(loss).item() * batch_size
        correct += (tensor_value(logits).argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def train_model(model, train_data, selected, downweight, epoch_indices, args, device):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch, indices in enumerate(epoch_indices, start=1):
        loader = DataLoader(
            IndexedDataset(train_data),
            batch_size=args.batch_size,
            sampler=indices,
            num_workers=args.num_workers,
        )
        train_loss, train_acc = train_one_epoch(
            model,
            loader,
            selected,
            downweight,
            loss_fn,
            optimizer,
            device,
            args.quiet,
        )
        if not args.quiet:
            print(
                f"  epoch {epoch}: train loss {train_loss:.4f}, "
                f"train acc {train_acc:.2%}"
            )


def test_loss(model, image, label, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        image = image.to(device)
        label = label.to(device)
        loss = loss_fn(model(image), label)

    return loss


def polynomial_coefficients(train_data, selected, test_image, test_label, epoch_indices, args, device):
    z = PolyTensor(
        [torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)],
        degree=args.degree,
    )
    model = make_model(device, args.seed + 40, degree=args.degree)

    print("training PolyTensor model for Taylor coefficients")
    train_model(model, train_data, selected, z, epoch_indices, args, device)
    loss = test_loss(model, test_image, test_label, device)
    if not isinstance(loss, PolyTensor):
        raise TypeError("expected the test loss to be a PolyTensor")
    return [coeff.detach().cpu().item() for coeff in loss.coeffs]


def retrained_losses(train_data, selected, test_image, test_label, epoch_indices, args, device):
    zs = torch.linspace(0.0, 1.0, args.num_dots).tolist()
    losses = []

    for z in zs:
        print(f"retraining empirical model at z={z:.2f}")
        model = make_model(device, args.seed + 40)
        train_model(model, train_data, selected, z, epoch_indices, args, device)
        losses.append(test_loss(model, test_image, test_label, device).detach().cpu().item())

    return zs, losses


def evaluate_polynomial(coefficients, zs, degree):
    ys = torch.zeros_like(zs)
    for power in range(degree + 1):
        ys = ys + coefficients[power] * zs.pow(power)
    return ys


def plot_results(zs, empirical, coefficients, test_index, num_deleted, args):
    grid = torch.linspace(0.0, 1.0, 301, dtype=torch.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(zs, empirical, color="black", label="empirical", zorder=3)

    for degree in range(1, args.degree + 1):
        approx = evaluate_polynomial(coefficients, grid, degree)
        ax.plot(grid, approx, label=f"degree {degree}", linewidth=2)

    ax.set_xlabel("downweight z")
    ax.set_ylabel("test loss f(z 1_D)")
    ax.set_title("MNIST deletion Taylor approximations")
    ax.text(
        0.01,
        0.99,
        f"{num_deleted} random training examples, test index {test_index}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    print(f"saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce a Figure-2-style Taylor approximation plot on MNIST."
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--downweight-fraction", type=float, default=DOWNWEIGHT_FRACTION)
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--num-dots", type=int, default=NUM_DOTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-index", type=int)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.degree < 1:
        raise ValueError("--degree must be at least 1")
    if args.degree != DEGREE:
        print(f"using degree {args.degree}; pass no --degree flag for the requested degree 3")
    if not 0.0 <= args.downweight_fraction <= 1.0:
        raise ValueError("--downweight-fraction must be between 0 and 1")
    if args.num_dots < 2:
        raise ValueError("--num-dots must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, selected, test_image, test_label, test_index = make_data(args)
    epoch_indices = make_epoch_indices(len(train_data), args.epochs, args.seed + 50)

    print(
        f"device={device}, train examples={len(train_data)}, "
        f"deleted={selected.sum().item()} ({selected.float().mean().item():.1%}), "
        f"test index={test_index}"
    )

    coefficients = polynomial_coefficients(
        train_data,
        selected,
        test_image,
        test_label,
        epoch_indices,
        args,
        device,
    )
    print("Taylor coefficients:", ", ".join(f"{c:.6g}" for c in coefficients))

    zs, empirical = retrained_losses(
        train_data,
        selected,
        test_image,
        test_label,
        epoch_indices,
        args,
        device,
    )
    plot_results(zs, empirical, coefficients, test_index, selected.sum().item(), args)


if __name__ == "__main__":
    main()
