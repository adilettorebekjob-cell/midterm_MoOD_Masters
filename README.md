# Java Vulnerability Detection with CodeT5

Management of Organizational Data — Midterm Group Project 2026, Group 3.

**Authors:** Adilet Torebek (25MD0297) and Galymzhan Beketay (25MD0314).

This project classifies Java functions as safe (`0`) or vulnerable (`1`) using **CodeT5-base with a classification head**. It covers dataset acquisition, validation, duplicate removal, data modeling, training, evaluation, and recommendations for organizational use.

The assigned base model was successfully trained. CodeT5-small, the permitted fallback, was not used. The model is a research prototype for assisting manual code review; its output does not certify code as safe.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `prepare.py` | Download the source archive, validate records, remove duplicates and contradictory labels, write clean JSONL partitions and a quality audit |
| `check.py` | Check hashes, class presence, uniqueness, and exact hash overlap across partitions |
| `train.py`, `predict.py` | Original V1 training and inference, retained for experimental reproducibility |
| `train_v2.py`, `predict_v2.py` | V2 training with class weights, validation-based selection, and inference using the saved threshold |
| `run_v2.py` | Start V2 using the active Python interpreter, check CUDA, create a separate run directory, and save console output |
| `baseline.py` | Diagnostic character TF-IDF and logistic-regression baseline; not a replacement for the assigned CodeT5 method |
| `test_model.py`, `test_v2.py` | Architecture and pipeline checks using a small random T5 backbone; not scientific performance experiments |
| `Group3_Colab.ipynb` | Original V1 notebook; see the notebook note below |
| `requirements.txt` | Pinned project dependencies; select the appropriate PyTorch CUDA build separately |
| `diagrams.md` | Data lifecycle, system architecture, and conceptual ER diagrams |
| `data/` | Source archive and prepared training, validation, and test JSONL files |
| `results/` | Run configurations, epoch histories, predictions, evaluation metrics, and figures |

The final report and presentation accompany the submission. Earlier generated drafts with pending results should not be used as final deliverables.

## Dataset and data quality

Source: [rmusab/vul-llm-finetune](https://github.com/rmusab/vul-llm-finetune/tree/main/Datasets).

Archive: `Datasets/with_p3/java_k_1_strict_2023_06_30.tar.gz`.

Source archive SHA256:

```text
76e1bc8dd360d122b8939569bf57919df4f388d37d3f1762b0088732ba919e58
```

| Partition | Raw rows | Clean rows | Safe | Vulnerable |
| --- | ---: | ---: | ---: | ---: |
| Train | 13,247 | 13,199 | 12,803 | 396 |
| Validation | 5,131 | 5,125 | 4,996 | 129 |
| Test | 4,576 | 4,572 | 4,453 | 119 |
| Total | 22,954 | 22,896 | 22,252 | 644 |

The pipeline removed 55 duplicate records and three records with contradictory labels. No records failed the implemented nonempty-code and binary-integer-target checks.

Duplicate identification uses SHA256 of whitespace-normalized code. The original code is retained as the model input. The existing partitions are preserved; duplicate retention prioritizes test, then validation, then training. There are no matching normalized-code hashes across the resulting partitions.

This is a heuristic duplicate check, not a parser or near-clone detector. Whitespace normalization can also merge strings with different whitespace inside literals. Project-level or temporal independence has not been established.

Files are used for storage; no SQL database, production warehouse, or streaming platform was implemented. The conceptual data model is documented in [diagrams.md](diagrams.md).

## Model and training

Only the `code` field is used as input. `target` supplies the training label; status, CWE, and other metadata are not input features.

- Backbone: `Salesforce/codet5-base`, revision `02cd2d31bb7c6d0e4d91156167b2de044989c733` in the recorded runs.
- Input limit: 256 tokens, with truncation and padding.
- Encoder-decoder output: representation at the final EOS position.
- Head: dropout, dense layer, tanh, dropout, and a two-class output layer.
- Full backbone and head fine-tuning; no LoRA or quantization.
- Three epochs; batch size 4; four accumulation steps; effective batch size 16 for complete groups.
- AdamW, learning rate `2e-5`, weight decay `0.01`, gradient clipping `1.0`, seed `42`.

V1 uses ordinary cross-entropy, threshold `0.5`, and checkpoint selection by validation F1. F1 was zero in all epochs, so its first checkpoint was retained.

V2 uses training-only inverse-frequency weights `N/(2*n_class)`: approximately `0.5155` for safe and `16.6654` for vulnerable examples. Per-example weighted loss is averaged over each batch. Validation F1 selects the threshold, with the higher threshold preferred on ties. Checkpoints are selected by this validation F1, then average precision on ties. The selected V2 checkpoint is epoch 3.

## Recorded home GPU experiments

Both full experiments used an NVIDIA RTX 5080, Python 3.13.7 and PyTorch `2.8.0+cu128`.

| Experiment / operating point | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1, threshold 0.5 | 97.40% | 0.00% | 0.00% | 0.0000 | 0.8354 |
| V2, validation-selected threshold 0.10551365 | 91.84% | 14.72% | 44.54% | 0.2213 | 0.8636 |
| V2, standard threshold 0.5 | 96.89% | 23.26% | 8.40% | 0.1235 | 0.8636 |

V1 predicted every test function as safe. V2 at the validation-selected threshold detected **53 of 119** vulnerable functions, missed 66, and produced 307 false alarms. At threshold 0.5, V2 detected 10 vulnerabilities and produced 33 false alarms.

The assignment requires accuracy of at least 93%. V2 at its maximum-validation-F1 threshold does **not** satisfy that condition on test. V2 at threshold 0.5 does, but has low recall. Always predicting safe already gives 97.40% accuracy on this test set, so accuracy alone is insufficient evidence of useful detection.

Measured training-and-evaluation time was approximately 20 min 34 s for V1 and 15 min 40 s for V2. Initial model download/loading is outside the training timer.

### Additional operating point in the report

The report/presentation also describes a postprocessing policy: maximize validation F1 subject to validation accuracy of at least 94%, giving an approximately `0.185` threshold and reported test accuracy of 94.1% (36 detected vulnerabilities, 83 missed, and 185 false alarms).

This policy is **separate from the original `train_v2.py`**, which selected approximately `0.1055`. Its result must be accompanied by the actual threshold-selection script (for example, `choose_threshold.py`) and saved outputs. The original `metrics.json` must remain unchanged. The inference threshold must match the operating point being demonstrated. Do not assume that rerunning `run_v2.py` produces the report's `0.185` operating point.

## Environment and installation

Use a dedicated virtual environment. Python 3.13.7 was used in the recorded home runs; Python 3.11 or 3.12 can also be used with compatible dependencies, but their results have not been measured here.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --force-reinstall torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Install only `torch` from the CUDA index. CodeT5 does not require torchvision or torchaudio. If incompatible optional image/audio packages were previously installed in a shared environment, use a fresh project environment or remove those packages.

Check the interpreter and GPU:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

Select the same interpreter in PyCharm and restart an existing notebook kernel after changing packages. A compatible NVIDIA driver is required. CUDA-enabled PyTorch wheels provide their runtime dependencies; a separate local CUDA Toolkit is not required for this project.

## Reproduce the experiments

Run commands from the repository root:

```powershell
python prepare.py
python check.py
python test_v2.py
python run_v2.py
```

`run_v2.py` checks CUDA and writes a separate timestamped run under `results/codet5_v2_...`. It prints progress every 100 batches and saves console output alongside the run directory.

To reproduce V1 with a new output directory:

```powershell
python -u train.py --output codet5_v1_reproduction
```

For the optional diagnostic baseline:

```powershell
python baseline.py
```

Use a new run name for each experiment; the training scripts reject an existing output directory. Record the actual environment:

```powershell
python -m pip freeze > results/environment_actual.txt
```

### Inspect a V2 run

- `config.json`: environment, settings, selected epoch and threshold.
- `history.json`: epoch metrics and cumulative elapsed time.
- `selection.json`: selected checkpoint and validation threshold.
- `validation_predictions.jsonl`: evidence for validation threshold selection.
- `metrics.json`: test results at the saved validation threshold.
- `metrics_at_0_5.json`: test results at threshold 0.5.
- `predictions.jsonl`: test labels, hashes and scores.
- `confusion_matrix.png`, `roc.png`: evaluation figures.

### Demonstration

```powershell
python predict_v2.py example.java --run results/ACTUAL_RUN_DIRECTORY
```

Replace `ACTUAL_RUN_DIRECTORY` with the real run directory and supply a Java text file. This command requires its `best.pt`, `tokenizer/` and `config.json` locally. Large trained checkpoints are excluded from regular Git tracking. A fresh clone can reproduce them by training; a saved checkpoint may also be distributed separately if needed.

The inference script uses the threshold in the run's configuration. Confirm it before claiming that a demo represents the report's separate operating point.

## Notebook

`Group3_Colab.ipynb` is the original V1 notebook. Renaming it does not update its logic. Before final submission it should be adapted to V2 and the intended environment. Windows/PyCharm should not use Colab upload cells or a Linux-only `tee` command.

For a local notebook beside the scripts, V2 can be launched with:

```python
%run run_v2.py
```

The notebook should also explain the pipeline and display the saved metrics and plots. The Python scripts remain the executable reference implementation.

## Data governance, security and limitations

Only public research data is used. Java source is read as text, never executed. No private bank code is part of the project. Data-owner, steward, developer and reviewer responsibilities are conceptual project roles. Production access control, encryption and retention enforcement have not been implemented as infrastructure; they are recommendations.

V2 was developed after inspecting V1 results, including test performance. The test set was reused, so it is not a fully independent final assessment of the development process. Loss weighting, threshold selection and checkpoint selection changed together; their individual effects were not isolated. There is one seed per version, no comprehensive near-clone/project-level audit, and inputs longer than 256 tokens are truncated.

## References

- Course assignment: *MoOD 10 Groups Midterm 2026*.
- [Dataset repository](https://github.com/rmusab/vul-llm-finetune).
- Shestov et al., *Finetuning Large Language Models for Vulnerability Detection*, IEEE Access, 2025. [DOI: 10.1109/ACCESS.2025.3546700](https://doi.org/10.1109/ACCESS.2025.3546700).
- Wang et al., *CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation*, 2021. [arXiv:2109.00859](https://arxiv.org/abs/2109.00859).
- [CodeT5-base model repository](https://huggingface.co/Salesforce/codet5-base).

Dataset and pretrained-model reuse remain subject to their respective terms. This repository does not assign a new license to third-party data or weights.
