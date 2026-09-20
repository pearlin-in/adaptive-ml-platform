import hashlib, json, os, tempfile

class Router:
    def __init__(self, config_path="serving/routing_config.json"):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = json.load(f)

    def _save(self):
        dir_ = os.path.dirname(self.config_path) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self.config, f, indent=2)
        os.replace(tmp, self.config_path)

    def route(self, model_id: str, sticky_key: str) -> str:
        cfg = self.config[model_id]
        canary, pct, stable = cfg.get("canary"), cfg.get("canary_percent", 0), cfg["stable"]
        if not canary or pct <= 0:
            return stable
        h = int(hashlib.sha256(f"{model_id}:{sticky_key}".encode()).hexdigest(), 16)
        return canary if (h % 100) < pct else stable

    def set_canary(self, model_id: str, version: str, percent: int):
        self.config[model_id]["canary"] = version
        self.config[model_id]["canary_percent"] = percent
        self._save()

    def promote_canary(self, model_id: str):
        cfg = self.config[model_id]
        cfg["stable"], cfg["canary"], cfg["canary_percent"] = cfg["canary"], None, 0
        self._save()

    def set_active_stable(self, model_id: str, version: str):
        self.config[model_id]["stable"] = version
        self.config[model_id]["canary"] = None
        self.config[model_id]["canary_percent"] = 0
        self._save()