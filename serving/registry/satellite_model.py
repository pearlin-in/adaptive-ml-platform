# serving/registry/satellite_model.py
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from .base import ModelVersion

class SatelliteModel(ModelVersion):
    def __init__(self, weights_path, class_names):
        self.model = mobilenet_v3_small()
        self.model.classifier[-1] = torch.nn.Linear(self.model.classifier[-1].in_features, 10)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, raw_input):  # PIL Image
        x = self.transform(raw_input.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())
        return {"label": self.class_names[idx]}, float(probs[idx])

    def embed(self, raw_input):
        x = self.transform(raw_input.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = self.model.features(x).mean(dim=[2, 3])
        return feat.squeeze(0).numpy()