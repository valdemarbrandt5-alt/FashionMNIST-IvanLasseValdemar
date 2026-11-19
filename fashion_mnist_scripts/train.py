import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

from model import CNN_Model


def translate_training_images(images):
    batch_size = len(images)
    # I det oprindelige run blev der trukket et tal til rotation for hvert billede, selv om vinklen var 0 grader.
    # Vi kasserer tallene her, så de næste tilfældige tal til forskydning følger samme sekvens.
    torch.rand(batch_size, device=images.device)

    # torch.eye laver en 2 × 3 matrix med ettaller på diagonalen og nuller i resten.
    # Den matrix ændrer ikke billedet. Med repeat laver vi en kopi til hvert billede.
    image_transforms = torch.eye(2, 3, device=images.device).repeat(batch_size, 1, 1)
    # Vi trækker et tal mellem -1 og 1 for hver retning i hvert billede.
    random_shifts = torch.rand(batch_size, 2, device=images.device) * 2 - 1
    # En hel billedside svarer til 2 i grid-koordinaterne. Derfor sætter vi den
    # maksimale forskydning til 2 * 0.05 i matricernes sidste kolonne.
    image_transforms[:, :, 2] = random_shifts * (2 * 0.05)

    # affine_grid finder, hvor hver pixel i det nye billede skal hentes fra.
    # grid_sample interpolerer mellem de omkringliggende pixelværdier og bruger 0 uden for billedet.
    sampling_grid = F.affine_grid(image_transforms, images.shape, align_corners=False)
    return F.grid_sample(
        images,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def train_one_epoch(batches, model, loss_function, optimizer):
    model.train()
    loss_sum = 0.0
    correct_prediction_count = 0

    for images, true_labels in batches:
        images = images.to(training_device)
        true_labels = true_labels.to(training_device)
        shifted_images = translate_training_images(images)
        class_scores = model(shifted_images)
        loss = loss_function(class_scores, true_labels)

        # Vi nulstiller gradients fra forrige batch. backward beregner nye gradients,
        # og optimizer.step bruger dem til at opdatere modellens vægte.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss er et gennemsnit for hvert batch. Vi ganger med antallet af billeder,
        # så det sidste, mindre batch ikke tæller lige så meget som et helt batch.
        loss_sum += loss.item() * len(true_labels)
        predicted_labels = class_scores.argmax(dim=1)
        correct_prediction_count += (predicted_labels == true_labels).sum().item()

    image_count = len(batches.dataset)
    return loss_sum / image_count, correct_prediction_count / image_count


def evaluate_validation_set(batches, model, loss_function):
    # model.eval() slår dropout fra og får batch normalization til at bruge de
    # statistikker, som modellen har samlet under træning.
    model.eval()
    loss_sum = 0.0
    true_label_counts = torch.zeros(10, dtype=torch.int64, device=training_device)
    predicted_label_counts = torch.zeros_like(true_label_counts)
    correct_label_counts = torch.zeros_like(true_label_counts)

    # Med no_grad gemmer vi ikke information til backpropagation, så vi sparer hukommelse.
    with torch.no_grad():
        for images, true_labels in batches:
            images = images.to(training_device)
            true_labels = true_labels.to(training_device)
            class_scores = model(images)
            loss_sum += loss_function(class_scores, true_labels).item() * len(true_labels)

            # argmax(dim=1) finder nummeret på den class, der har højest score for hvert billede.
            predicted_labels = class_scores.argmax(dim=1)
            matches = true_labels == predicted_labels
            # bincount tæller labels for hver class. minlength=10 giver plads til alle classes.
            true_label_counts += torch.bincount(true_labels, minlength=10)
            predicted_label_counts += torch.bincount(predicted_labels, minlength=10)
            correct_label_counts += torch.bincount(true_labels[matches], minlength=10)

    # Macro F1 er gennemsnittet af F1-scoren for de ti classes. clamp undgår division
    # med nul, så en class uden rigtige eller forudsagte labels får en F1-score på 0.
    label_counts = true_label_counts + predicted_label_counts
    f1_per_class = 2 * correct_label_counts / label_counts.clamp(min=1)
    image_count = len(batches.dataset)
    accuracy = correct_label_counts.sum().item() / image_count
    return loss_sum / image_count, accuracy, f1_per_class.mean().item()


# Vi bruger ét seed til træningen og et andet til valget af valideringsbilleder.
torch.manual_seed(72)
training_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {training_device} device")

# ToImage laver billedet om til en tensor. Med scale=True skalerer vi pixels fra 0–255 til 0–1.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)
split_generator = torch.Generator().manual_seed(404)
training_set, validation_set = random_split(
    training_data, [50_000, 10_000], generator=split_generator
)
training_batches = DataLoader(training_set, batch_size=64, shuffle=True)
validation_batches = DataLoader(validation_set, batch_size=64)

model = CNN_Model().to(training_device)
# CrossEntropyLoss tager de rå scores for hver class og beregner selv log-softmax.
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
best_validation_f1 = -1.0

for epoch in range(1, 31):
    training_loss, training_accuracy = train_one_epoch(
        training_batches, model, loss_function, optimizer
    )
    validation_loss, validation_accuracy, validation_f1 = evaluate_validation_set(
        validation_batches, model, loss_function
    )

    # Med > beholder vi checkpointet fra første gang, vi opnåede den bedste F1-score, hvis den opnås igen.
    if validation_f1 > best_validation_f1:
        best_validation_f1 = validation_f1
        torch.save(model.state_dict(), "trained_model.pth")

    print(
        f"Epoch {epoch}: Train loss={training_loss:.4f}, Val loss={validation_loss:.4f}, "
        f"Train acc={training_accuracy:.2%}, Val acc={validation_accuracy:.2%}, "
        f"Val macro F1={validation_f1:.4f}",
        flush=True,
    )
