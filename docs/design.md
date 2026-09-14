My bad on the double code block escaping! Here is the clean raw text in a single block so you can copy and paste it directly into your `docs/design.md` file:

```markdown
# System Design & Architecture Specification

## 1. Unified `ModelVersion` Interface
To serve heterogeneous payloads (tabular, image, text) through a single serving layer, all models must adhere to an interchangeable prediction contract. Raw payloads are normalized upstream by modal-specific preprocessors prior to executing the core prediction call.

### Signature Definition
```python
from abc import ABC, abstractmethod
from typing import Any, Tuple

class ModelVersion(ABC):
    @abstractmethod
    def predict(self, model_input: Any) -> Tuple[Any, float]:
        """
        Executes inference on pre-processed input tensor/array.
        
        Args:
            model_input: Standardized numerical representation (numpy array or torch Tensor).
            
        Returns:
            Tuple[output, confidence]: 
                - output: Predicted label, class integer, or target value.
                - confidence: Standardized confidence score in range [0.0, 1.0].
        """
        pass

```

### Design Rationale

Decoupling data preprocessing from the model prediction step guarantees that the core server loop (`asyncio.Queue` batching worker) operates on uniform array shapes without needing payload-specific branching logic.

---

## 2. Model Registry Manifest Schema

The Model Registry uses SQLite/JSON metadata manifests to track active artifacts, parameters, reference statistics, and version histories. This enables zero-downtime hot-swapping without requiring service restarts.

### Schema (JSON Format)

```json
{
  "model_id": "fraud-detection-lgb",
  "version": "v1.0.0",
  "artifact_path": "models/fraud/v1.joblib",
  "reference_stats_path": "models/fraud/v1_reference.json",
  "created_at": "2026-09-15T00:00:00Z",
  "metrics": {
    "accuracy": 0.984,
    "f1_score": 0.912,
    "auc_roc": 0.978
  }
}

```

---

## 3. Modality-Specific Drift Detection Metrics

Rather than relying on generic black-box drift libraries, statistical metrics are matched explicitly to each input modality's mathematical structure:

* **Tabular (Fraud)**: **Population Stability Index (PSI)** & **Kolmogorov-Smirnov (KS) Test**
* *Why:* PSI directly quantifies the degree of distribution shift across categorical and continuous feature bins. The non-parametric KS-test provides exact p-values for continuous variables (e.g., transaction amounts) without making underlying distribution assumptions.


* **Image (Satellite / EuroSAT)**: **Cosine Distance on Penultimate Layer Embeddings**
* *Why:* Pixel-level statistics fail to capture semantic shift (e.g., sensor calibration changes or resolution alterations). Comparing cosine distance between operational embedding centroids and baseline validation centroids isolates semantic domain shifts.


* **Text (AI-Text Detection / HC3)**: **Confidence Shift & Embedding Distance**
* *Why:* Text input features are discrete tokens. Measuring embedding distance alongside output confidence distribution shifts effectively detects out-of-distribution (OOD) generator outputs (such as modern LLM text unseen during training).



---

## 4. Automated Drift Mitigation & Rollback Policy

To ensure self-healing operations without human intervention, automated fallback actions trigger upon verified statistical drift.

### Trigger Conditions

* **Drift Metric Threshold ($\tau$)**:
* Tabular PSI $\ge 0.25$ (indicates significant distributional shift).
* Embedding Cosine Distance Shift $\ge 0.30$.


* **Consecutive Window Requirement ($M$)**: Drift score must exceed threshold $\tau$ for **$M = 3$ consecutive evaluation windows** (prevents false positives caused by transient noise spikes).

### Automated Response Execution

1. **Traffic Rollback**: Dynamic router immediately resets the traffic split config to route 100% of requests back to the previous stable champion version.
2. **Circuit Breaker**: If no stable fallback version exists, the system activates a fallback circuit breaker returning low-risk default predictions.
3. **Structured Incident Logging**: Emits an automated incident record (storing timestamp, affected model ID, failing metric score, and consecutive window count) to the SQLite log database.

```

```