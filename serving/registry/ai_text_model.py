import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import ModelVersion

class AITextModel(ModelVersion):
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.label_map = {0: "human", 1: "ai"}

    def predict(self, raw_input: str):
        return self.predict_batch([raw_input])[0]

    def predict_batch(self, raw_inputs: list[str]) -> list[tuple[dict, float]]:
        inputs = self.tokenizer(
            raw_inputs,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        
        results = []
        for prob in probs:
            idx = int(prob.argmax())
            results.append(({"label": self.label_map[idx]}, float(prob[idx])))
        return results

    def embed(self, raw_input: str):
        inputs = self.tokenizer(raw_input, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = self.model.distilbert(**inputs)
        return out.last_hidden_state[0, 0, :].numpy()