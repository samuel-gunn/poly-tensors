import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from PolyTensor import PolyTensor


DATA_DIR = "data"
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 1e-3
DOWNWEIGHT_FRACTION = 0.01
DEGREE = 2


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


def make_loaders():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    selected = torch.zeros(len(train_data), dtype=torch.bool)
    selected[torch.randperm(len(train_data))[: int(DOWNWEIGHT_FRACTION * len(train_data))]] = True

    train_loader = DataLoader(IndexedDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)
    return train_loader, test_loader, selected


def train_one_epoch(model, loader, selected, X, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, indices in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        selected_batch = selected[indices].to(device)

        logits = model(images)
        per_example_loss = loss_fn(logits, labels)
        weights = 1 - selected_batch.float() * X
        loss = (per_example_loss * weights).sum() * (1.0 / images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += tensor_value(loss).item() * batch_size
        correct += (tensor_value(logits).argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        batch_size = images.size(0)
        total_loss += tensor_value(loss).sum().item()
        correct += (tensor_value(logits).argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = PolyTensor(
        [torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)],
        degree=DEGREE,
    )

    train_loader, test_loader, selected = make_loaders()
    model = MNISTClassifier().to(device)
    make_poly_parameters(model, X.degree)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    
    print(
        f"initialization: "
        f"test loss {test_loss:.4f}, test acc {test_acc:.2%}"
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, selected, X, loss_fn, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        print(
            f"epoch {epoch}: "
            f"train loss {train_loss:.4f}, train acc {train_acc:.2%}, "
            f"test loss {test_loss:.4f}, test acc {test_acc:.2%}"
        )


if __name__ == "__main__":
    main()
