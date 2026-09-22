# scripts/seed_v2_versions.py — idempotent-safe version
import json, shutil, os

MANIFEST_PATH = "serving/registry/manifest.json"
ROUTING_CONFIG_PATH = "serving/routing_config.json"

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
with open(ROUTING_CONFIG_PATH) as f:
    routing_config = json.load(f)

for model_id in ("fraud", "satellite"):
    if "v2" in manifest[model_id]["versions"]:
        print(f"{model_id}: v2 already exists — leaving active_version/stable untouched "
              f"(currently {manifest[model_id]['active_version']!r}). Not re-seeding.")
        continue
    base_path = manifest[model_id]["versions"]["v1"]["path"]
    copy_map = {
        "fraud": [("model_v1.joblib", "model_v2.joblib"), ("scaler_v1.joblib", "scaler_v2.joblib"),
                  ("metrics_v1.json", "metrics_v2.json"), ("reference_sample_v1.parquet", "reference_sample_v2.parquet")],
        "satellite": [("model_v1.pt", "model_v2.pt"), ("metrics_v1.json", "metrics_v2.json"),
                      ("reference_embeddings_v1.pt", "reference_embeddings_v2.pt")],
    }[model_id]
    for src_name, dst_name in copy_map:
        src, dst = f"{base_path}/{src_name}", f"{base_path}/{dst_name}"
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    v2_cfg = dict(manifest[model_id]["versions"]["v1"])
    v2_cfg["metrics_file"] = f"{base_path}/metrics_v2.json"
    v2_cfg["reference_file"] = (
        f"{base_path}/reference_sample_v2.parquet" if model_id == "fraud"
        else f"{base_path}/reference_embeddings_v2.pt"
    )
    manifest[model_id]["versions"]["v2"] = v2_cfg
    manifest[model_id]["active_version"] = "v2"
    routing_config[model_id]["stable"] = "v2"
    routing_config[model_id]["canary"] = None
    routing_config[model_id]["canary_percent"] = 0

# ai_text: explicit path construction, not fragile substring replace.
if "v2" in manifest["ai_text"]["versions"]:
    print(f"ai_text: v2 already exists — leaving active_version untouched "
          f"(currently {manifest['ai_text']['active_version']!r}). Not re-seeding.")
else:
    v1_dir = manifest["ai_text"]["versions"]["v1"]["path"]          # models/ai_text/model_v1
    parent_dir = os.path.dirname(v1_dir)                            # models/ai_text
    v2_dir = f"{parent_dir}/model_v2"

    if not os.path.exists(v2_dir):
        shutil.copytree(v1_dir, v2_dir)

    v2_cfg = dict(manifest["ai_text"]["versions"]["v1"])
    v2_cfg["path"] = v2_dir
    v2_cfg["metrics_file"] = f"{parent_dir}/metrics_v2.json"
    v2_cfg["reference_file"] = f"{parent_dir}/reference_embeddings_v2.pt"
    manifest["ai_text"]["versions"]["v2"] = v2_cfg
    manifest["ai_text"]["active_version"] = "v2"
    routing_config["ai_text"]["stable"] = "v2"
    routing_config["ai_text"]["canary"] = None
    routing_config["ai_text"]["canary_percent"] = 0

REFERENCE_FILE_FIXUPS = [
    ("models/fraud/reference_sample_v1.parquet", "models/fraud/reference_sample_v2.parquet"),
    ("models/satellite/reference_embeddings_v1.pt", "models/satellite/reference_embeddings_v2.pt"),
    ("models/ai_text/reference_embeddings_v1.pt", "models/ai_text/reference_embeddings_v2.pt"),
]
for src, dst in REFERENCE_FILE_FIXUPS:
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"Copied missing reference file: {dst}")

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
with open(ROUTING_CONFIG_PATH, "w") as f:
    json.dump(routing_config, f, indent=2)
    
print("Done.")