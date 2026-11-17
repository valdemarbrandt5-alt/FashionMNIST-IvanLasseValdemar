import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

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
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

model = NeuralNetwork().to(device)
print(model)



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    total_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (pred.argmax(1) == y).sum().item()

        if batch % 100 == 0:
            current = min((batch + 1) * len(X), size)
            print(f"loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")

    return total_loss / size, correct / size


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
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

    avg_loss = total_loss / size
    accuracy = correct / size

    print(f"Test Error: \n Accuracy: {accuracy:.1%}, Avg loss: {avg_loss:.4f}\n")

    return avg_loss, accuracy



epochs = 5

history = {
    "train_loss": [],
    "test_loss": [],
    "train_acc": [],
    "test_acc": [],
}

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    train_loss, train_acc = train(
        train_dataloader, model, loss_fn, optimizer
    )

    test_loss, test_acc = test(
        test_dataloader, model, loss_fn
    )

    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)

print("Done!")

x = range(1, epochs + 1)

# Loss graph
plt.figure()
plt.plot(x, history["train_loss"], label="Train")
plt.plot(x, history["test_loss"], label="Test")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Original Model - Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("original_loss_curve.png")
plt.show()

# Accuracy graph
plt.figure()
plt.plot(x, history["train_acc"], label="Train")
plt.plot(x, history["test_acc"], label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Original Model - Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("original_accuracy_curve.png")
plt.show()

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


print("\n=== RESULTS FOR OVERLEAF ===")

for i in range(len(history["train_acc"])):
    print(
        f"Epoch {i+1}: "
        f"Train accuracy={history['train_acc'][i]*100:.2f}, "
        f"Train loss={history['train_loss'][i]:.4f}"
    )

eval_acc = history.get("val_acc", history.get("test_acc"))
eval_loss = history.get("val_loss", history.get("test_loss"))

for i in range(len(eval_acc)):
    print(
        f"Epoch {i+1}: "
        f"Evaluation accuracy={eval_acc[i]*100:.2f}, "
        f"Evaluation loss={eval_loss[i]:.4f}"
    )