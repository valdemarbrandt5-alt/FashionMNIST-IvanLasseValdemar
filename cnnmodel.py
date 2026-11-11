import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2


#Training data download
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

#Test data download
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)


#Opdeler billeder den ser ad gang til størrelse af 64 ad gangen og sætter dem i greyscale billede i størrelse 28x28 pixels.
batch_size = 64

# Create data loaders.

batch_size = 64

train_data, val_data = random_split(
    training_data,
    [50_000, 10_000],
    generator=torch.Generator().manual_seed(42)
)

train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)

val_dataloader = DataLoader(
    val_data, batch_size=batch_size
)

test_dataloader = DataLoader(
    test_data, batch_size=batch_size
)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 10),
        )

        

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

model = NeuralNetwork().to(device)
print(model)



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.0001)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record results
        total_loss += loss.item() * len(y)
        correct += (pred.argmax(1) == y).sum().item()

    size = len(dataloader.dataset)
    return total_loss / size, correct / size


def evaluate(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item() * len(y)
            correct += (pred.argmax(1) == y).sum().item()

    size = len(dataloader.dataset)
    return total_loss / size, correct / size





epochs = 10

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
}

for epoch in range(epochs):
    train_loss, train_acc = train(
        train_dataloader, model, loss_fn, optimizer
    )

    val_loss, val_acc = evaluate(
        val_dataloader, model, loss_fn
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1}: "
        f"Train loss={train_loss:.4f}, "
        f"Val loss={val_loss:.4f}, "
        f"Train acc={train_acc:.2%}, "
        f"Val acc={val_acc:.2%}"
    )


x = range(1, epochs + 1)

# Loss curves
plt.figure()
plt.plot(x, history["train_loss"], label="Train")
plt.plot(x, history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

# Accuracy curves
plt.figure()
plt.plot(x, history["train_acc"], label="Train")
plt.plot(x, history["val_acc"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")
plt.show()



torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")