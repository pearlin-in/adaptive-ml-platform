import json

MANIFEST_PATH = "serving/registry/manifest.json"
ROUTING_CONFIG_PATH = "serving/routing_config.json"

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
with open(ROUTING_CONFIG_PATH) as f:
    routing_config = json.load(f)

for model_id in ("fraud", "satellite", "ai_text"):
    manifest[model_id]["active_version"] = "v2"
    routing_config[model_id]["stable"] = "v2"
    routing_config[model_id]["canary"] = None
    routing_config[model_id]["canary_percent"] = 0

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
with open(ROUTING_CONFIG_PATH, "w") as f:
    json.dump(routing_config, f, indent=2)
print("All 3 models reset to active/stable = v2.")