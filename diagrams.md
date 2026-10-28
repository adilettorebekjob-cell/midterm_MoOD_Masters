# Project diagrams

These diagrams describe the implemented file-based pipeline and its conceptual data relationships. They do not represent a deployed SQL database or a production review service.

## Data lifecycle

```mermaid
flowchart TD
    A["Acquire public Java archive"] --> B["Record archive SHA256 and retain original"]
    B --> C["Validate code and labels"]
    C --> D["Quarantine conflicting labels and remove duplicates"]
    D --> E["Write clean train, validation and test JSONL"]
    E --> F["Train CodeT5 and evaluate on validation"]
    F --> G["Select checkpoint and threshold on validation"]
    G --> H["Evaluate selected model on test"]
    H --> I["Store configuration, predictions, metrics and figures"]
    I --> J["Retain grading evidence and delete disposable copies"]
```

The final retention/deletion step is a proposed policy, not an automated script. The original archive and derived hashes provide provenance; hashes do not themselves prove correct labels or authorized data use.

## ETL

| Stage | Implementation |
| --- | --- |
| Extract | `prepare.py` downloads the archive when absent and reads JSONL members |
| Transform | Check nonempty code and integer binary labels; compute normalized-code SHA256; remove conflicting labels and duplicate hashes |
| Load | Write clean JSONL partitions, consumed by the model's Dataset and DataLoader |

Global duplicate checking uses memory. Training uses minibatches but loads JSONL records into memory; this is not an end-to-end streaming system.

## Training and evaluation architecture

```mermaid
flowchart TD
    A["Public archive"] --> B["prepare.py and check.py"]
    B --> C["Clean training JSONL"]
    B --> D["Clean validation JSONL"]
    B --> E["Clean test JSONL"]
    C --> F["Tokenizer and CodeT5 encoder-decoder"]
    F --> G["EOS representation and classification head"]
    G --> H["Weighted training loss and optimizer"]
    H --> F
    D --> I["Validation scoring and threshold selection"]
    G --> I
    I --> J["Saved checkpoint and threshold"]
    J --> K["Test evaluation"]
    E --> K
    K --> L["Metrics, predictions, ROC and confusion matrix"]
```

V2 class weights are computed from training counts only. Checkpoint selection uses validation F1 with average precision as a tie-breaker. The original V2 threshold maximizes validation F1; the report's additional accuracy-constrained policy requires its separate postprocessing code and evidence. Test results do not feed the selection functions, but the overall development process reused test results between experiments, as disclosed in the README.

## Inference architecture

```mermaid
flowchart TD
    A["Java function text"] --> B["Saved tokenizer, maximum 256 tokens"]
    B --> C["CodeT5 encoder and decoder"]
    C --> D["Classification head and softmax score"]
    E["Saved run configuration and threshold"] --> F["Apply decision threshold"]
    D --> F
    F --> G["Predicted label and suggested manual review"]
```

Input features contain code only. Ground-truth labels and CWE metadata are not passed to the model as predictor inputs. Manual review is a recommended use, not an implemented review-queue application.

## Conceptual ER model

```mermaid
erDiagram
    DATASET ||--o{ FUNCTION : contains
    FUNCTION ||--|| LABEL : has
    DATASET ||--o{ RUN : supports
    RUN ||--o{ PREDICTION : produces
    FUNCTION ||--o{ PREDICTION : receives
    RUN ||--o{ METRIC : summarizes
    DATASET {
        string dataset_id PK
        string archive_sha256
        string source_url
    }
    FUNCTION {
        string code_sha256 PK
        string dataset_id FK
        string split
        string code
    }
    LABEL {
        string code_sha256 FK
        int target
        string cwe
    }
    RUN {
        string run_id PK
        string dataset_id FK
        string model_revision
        string configuration
        int best_epoch
        float threshold
    }
    PREDICTION {
        string run_id FK
        string code_sha256 FK
        string partition
        float probability
    }
    METRIC {
        string run_id FK
        string partition
        float threshold
        string name
        float value
    }
```

Prediction identity is the combination of run and function. Metric identity additionally includes partition, decision threshold and metric name. Conceptual entities are represented across JSONL and JSON files; there are no SQL foreign-key constraints in the implementation.
