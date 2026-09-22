import json

with open("serving/routing_config.json") as f:
    cfg = json.load(f)

cfg["satellite"]["stable"] = "v2"
cfg["satellite"]["canary"] = None
cfg["satellite"]["canary_percent"] = 0

with open("serving/routing_config.json", "w") as f:
    json.dump(cfg, f, indent=2)

print("satellite routing reset to stable=v2, no canary")