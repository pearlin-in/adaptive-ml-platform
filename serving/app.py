# serving/app.py
batchers: dict[tuple[str, str], MicroBatcher] = {}

def get_batcher(model_id: str, version: str) -> MicroBatcher:
    key = (model_id, version)
    if key not in batchers:
        model = registry.get(model_id, version)
        batchers[key] = MicroBatcher(model, max_latency_ms=30, max_batch_size=16)
    return batchers[key]

@app.post("/predict/fraud")
async def predict_fraud(payload: FraudInput, x_user_id: str | None = Header(default=None)):
    sticky_key = x_user_id or str(uuid.uuid4())  # no sticky id → effectively random per call, note this in the design doc
    version = router.route("fraud", sticky_key)
    result, confidence = await get_batcher("fraud", version).submit(payload.dict())
    return {"version_served": version, "result": result, "confidence": confidence}

@app.get("/compare/{model_id}")
async def compare_versions(model_id: str):
    # pull from your prediction log (SQLite/parquet, once Phase 5 exists) grouped by version_served
    # for now, a stub is fine — return avg confidence and count per version
    ...