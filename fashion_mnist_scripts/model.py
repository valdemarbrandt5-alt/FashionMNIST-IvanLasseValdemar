from torch import nn


class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Det første Conv2d-lag tager billeder med én gråtonekanal og bruger 16 filtre på 3 × 3.
        # Med padding=1 beholder vi samme højde og bredde gennem hvert convolution-lag.
        # MaxPool2d(2) tager den største værdi i hvert 2 × 2-felt, så højde og bredde går fra 28 til 14 og derefter 7.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Med Flatten samler vi de 32 feature maps på 7 × 7 til én vektor for hvert billede.
        # Dropout(p=0.2) nulstiller hver aktivering med 20 % sandsynlighed under træning.
        # Det sidste lag giver én rå score for hver class.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        image_features = self.features(images)
        return self.classifier(image_features)
