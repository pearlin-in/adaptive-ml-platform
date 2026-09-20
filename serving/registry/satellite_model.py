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
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, raw_input):
        return self.predict_batch([raw_input])[0]

    def predict_batch(self, raw_inputs: list) -> list[tuple[dict, float]]:
        tensors = [self.transform(img.convert("RGB")) for img in raw_inputs]
        batch_tensor = torch.stack(tensors)
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        
        results = []
        for prob in probs:
            idx = int(prob.argmax())
            results.append(({"label": self.class_names[idx]}, float(prob[idx])))
        return results

    def embed(self, raw_input):
        x = self.transform(raw_input.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = self.model.features(x).mean(dim=[2, 3])
        return feat.squeeze(0).numpy()