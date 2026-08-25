# Hypoglycemia-DL

Code for training and evaluating a multi-input bidirectional long short-term memory (BiLSTM) model for binary hypoglycemia prediction.

## Code

- `tuner_CW_bootstrap.py`: uses binary cross-entropy with class-based sample weights.
- `tuner_wbce_bootstrap.py`: uses a custom weighted binary cross-entropy loss.

Both scripts perform Bayesian hyperparameter tuning, retrain the best model, generate test-set probabilities, calculate bootstrap performance estimates, and plot training and validation loss.

## Required data

Each train, validation, and test split requires medication, laboratory, diet, meal, static, and outcome files:

```text
X_{split}_meds.csv.gz
X_{split}_labs.csv.gz
X_{split}_diet.csv.gz
X_{split}_meal.csv.gz
X_{split}_static.csv.gz
y_{split}.csv.gz
```

Replace `{split}` with `train`, `val`, or `test`. Files must be headerless, numeric, row-aligned, and stored in the script's working directory.

| Input | Shape per observation | CSV columns |
| --- | --- | ---: |
| Medications | `(30, 33)` | 990 |
| Laboratory values | `(30, 13)` | 390 |
| Diet | `(30, 4)` | 120 |
| Meal | `(30, 1)` | 30 |
| Static variables | `(65,)` | 65 |
| Outcome | `(1,)` | 1 |

## Model

Each time-series input is processed by a separate BiLSTM branch. Static variables pass through a dense layer. The five representations are concatenated and passed through a dense layer and sigmoid output.

The tuner searches over:

- One or two BiLSTM layers
- 16–256 units per layer
- Dropout from 0.5–0.9
- 64–96 dense units
- Learning rates from `1e-5` to `5e-4`

The tuning objective is validation F1 at a probability cutoff of 0.5.

## Requirements

```text
tensorflow
keras-tuner
numpy
pandas
matplotlib
scikit-learn
```

## Run

Run one class-imbalance strategy:

```bash
python tuner_CW_bootstrap.py
```

or:

```bash
python tuner_wbce_bootstrap.py
```

## Outputs

| Strategy | Probabilities | Bootstrap metrics | Loss plot |
| --- | --- | --- | --- |
| Class weights | `CW_y_pred_probs.csv` | `CW_bootstrap_metrics.csv` | `CW_train_val_loss.svg` |
| Weighted loss | `wbce_y_pred_probs.csv` | `wbce_bootstrap_metrics.csv` | `wbce_train_val_loss.svg` |

The scripts evaluate thresholds of 0.5–0.9 using 100 bootstrap samples.
