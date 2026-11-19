import csv

import torch
from PIL import Image
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2

from model import CNN_Model


model = CNN_Model()
# Med map_location kan vi også åbne et checkpoint fra GPU-træning på CPU.
saved_parameters = torch.load(
    "final_winner_translation/model.pth",
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(saved_parameters)
model.eval()
image_to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
class_names = FashionMNIST.classes

with open("samples/labels.csv", encoding="utf-8", newline="") as label_file, torch.no_grad():
    for sample in csv.DictReader(label_file):
        filename = sample["file"]
        with Image.open(f"samples/{filename}") as image:
            # Med unsqueeze(0) tilføjer vi en batch-dimension, så formen bliver (1, 1, 28, 28).
            image_batch = image_to_tensor(image).unsqueeze(0)
        class_scores = model(image_batch)
        predicted_class_index = class_scores.argmax(dim=1).item()
        true_class = class_names[int(sample["label"])]
        predicted_class = class_names[predicted_class_index]
        print(f"{filename}: true label = {true_class}, predicted label = {predicted_class}")
