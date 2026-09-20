import json
import os
import tempfile
from .fraud_model import FraudModel
from .satellite_model import SatelliteModel
from .ai_text_model import AITextModel

LOADERS = {"fraud": FraudModel, "satellite": SatelliteModel, "ai_text": AITextModel}

class ModelRegistry:
    def __init__(self, manifest_path="serving/registry/manifest.json"):
        self.manifest_path = manifest_path
        self._cache = {}
        self._load_manifest()

    def _load_manifest(self):
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)

    def _save_manifest(self):
        """Atomic write pattern: writes to temporary file then renames to avoid corruption."""
        dir_name = os.path.dirname(self.manifest_path) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as f:
            json.dump(self.manifest, f, indent=2)
            tmp_path = f.name
        os.replace(tmp_path, self.manifest_path)  # Atomic operation on POSIX & OS-level

    def register_version(self, model_id: str, version: str, config: dict):
        """Add a new version to the manifest WITHOUT restarting the server."""
        if model_id not in self.manifest:
            self.manifest[model_id] = {"active_version": version, "versions": {}}
        self.manifest[model_id]["versions"][version] = config
        self._save_manifest()

    def set_active(self, model_id: str, version: str):
        if model_id not in self.manifest or version not in self.manifest[model_id]["versions"]:
            raise ValueError(f"Version {version} not found for model_id '{model_id}'")
        self.manifest[model_id]["active_version"] = version
        self._save_manifest()

    def get(self, model_id: str, version: str = None):
        version = version or self.manifest[model_id]["active_version"]
        key = (model_id, version)
        if key not in self._cache:
            cfg = self.manifest[model_id]["versions"][version]
            self._cache[key] = self._build(model_id, cfg)
        return self._cache[key]

    def _build(self, model_id: str, cfg: dict):
        if model_id == "fraud":
            return FraudModel(
                model_path=f"{cfg['path']}/model_v1.joblib", 
                scaler_path=f"{cfg['path']}/scaler_v1.joblib"
            )
            
        if model_id == "satellite":
            metrics_path = cfg["metrics_file"]
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Metrics file missing for satellite v1 at: {metrics_path}")

            if "class_names" not in metrics:
                raise KeyError(
                    f"Configuration error in {metrics_path}: missing required key 'class_names'. "
                    f"Refusing to fall back to hardcoded defaults."
                )
            return SatelliteModel(f"{cfg['path']}/model_v1.pt", metrics["class_names"])
            
        if model_id == "ai_text":
            return AITextModel(cfg["path"])
            
        raise ValueError(f"Unknown model_id: {model_id}")