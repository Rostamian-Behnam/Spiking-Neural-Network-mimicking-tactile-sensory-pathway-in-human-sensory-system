# Multi-Task Classifier Pipeline

A clean, general-purpose machine-learning pipeline for running **multiple classification tasks** with **multiple scikit-learn classifiers** in one shot. Designed for neuro-sensory data (passive-touch texture / speed / direction decoding) but completely data-agnostic — swap in any feature matrix and label vector.

---

## Features

- **Any number of tasks** — texture, speed, direction, or your own labels
- **Any scikit-learn classifier** — KNN, SVM, Random Forest, Logistic Regression (and anything else you add to the catalogue)
- **Auto confusion-matrix heat-maps** — one per classifier × task combination, saved as PNG or shown interactively
- **Grouped accuracy bar-chart** — all tasks and classifiers side-by-side
- **Formatted console report** — classification report + summary accuracy table
- **Self-test mode** — works out-of-the-box with built-in sklearn datasets; no files needed to try it

---

## Table of Contents

- [Background](#background)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Quick Examples](#quick-examples)
- [Adding Your Own Classifiers](#adding-your-own-classifiers)
- [Data Loader (Neuron CSVs)](#data-loader-neuron-csvs)
- [Output Files](#output-files)
- [Code Structure](#code-structure)
- [Troubleshooting](#troubleshooting)

---

## Background

This pipeline was built to decode passive-touch sensory information from neuron spike-count recordings.  Three parallel classification problems are solved simultaneously:

| Task | What is classified | Labels (example) |
|------|--------------------|------------------|
| **Texture** | Which surface is being touched | `circ_0`, `rect_2`, `wave_4`, … |
| **Speed** | How fast the stimulus moves | `40`, `60`, `80` (mm/s) |
| **Direction** | Scan direction of the stimulus | `X`, `Y` |

All three share the same feature vectors (X- and Y-axis neuron spike counts, concatenated), so loading happens once and the features are simply relabelled per task.

---

## Pipeline Architecture

```
 ┌─────────────────────────────────────────────────┐
 │               run_pipeline(tasks, classifiers)   │
 │                                                   │
 │  for each ClassificationTask:                     │
 │   └─ run_task()                                   │
 │       for each classifier:                        │
 │        └─ evaluate_classifier()                   │
 │            ├─ clf.fit(X_train, y_train)           │
 │            ├─ clf.predict(X_test)                 │
 │            ├─ print classification_report         │
 │            └─ _plot_confusion_matrix()  ──► PNG   │
 │                                                   │
 │  _print_summary()          ──► console table      │
 │  _plot_accuracy_comparison() ──► bar chart PNG    │
 └─────────────────────────────────────────────────┘
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/classifier-pipeline.git
cd classifier-pipeline

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install numpy scikit-learn matplotlib seaborn
```

> Python 3.8 or later required.

---

## Usage

### Self-test (no data files needed)

```bash
python classifier_pipeline.py
```

This uses the built-in Iris / Wine / Cancer datasets as stand-ins for your real tasks. Confusion matrices and an accuracy bar-chart will appear, and PNGs are saved to `outputs/`.

### With your own data

1. Open `classifier_pipeline.py` and go to **OPTION A** in the `__main__` block.
2. Set `DATA_DIR` to your data folder.
3. Uncomment the OPTION A block and comment out OPTION B.
4. Run `python classifier_pipeline.py`.

---

## Quick Examples

### Example 1 — Two sklearn datasets as tasks

```python
from sklearn.datasets import load_iris, load_wine
from classifier_pipeline import ClassificationTask, run_pipeline

iris = load_iris()
wine = load_wine()

tasks = [
    ClassificationTask("Iris", iris.data, iris.target),
    ClassificationTask("Wine", wine.data, wine.target),
]

results = run_pipeline(tasks, show_plot=False)
# results == {"Iris": {"KNN (k=5)": 0.97, ...}, "Wine": {...}}
```

---

### Example 2 — Single task, custom classifiers

```python
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from classifier_pipeline import ClassificationTask, run_pipeline

cancer = load_breast_cancer()
task   = ClassificationTask("Cancer", cancer.data, cancer.target)

my_classifiers = {
    "SVM (RBF)":     SVC(kernel="rbf", C=10),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=0),
}

results = run_pipeline([task], classifiers=my_classifiers, show_plot=True)
```

---

### Example 3 — Sweep a hyperparameter and compare

```python
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from classifier_pipeline import ClassificationTask, run_pipeline

wine  = load_wine()
task  = ClassificationTask("Wine", wine.data, wine.target)

clfs  = {f"KNN k={k}": KNeighborsClassifier(n_neighbors=k) for k in [1, 3, 5, 7, 9]}
results = run_pipeline([task], classifiers=clfs, show_plot=True)
```

Produces a bar chart comparing all five k values at once.

---

### Example 4 — Save results to disk (no pop-up windows)

```python
results = run_pipeline(
    tasks,
    save_dir  = "my_results",   # folder is created automatically
    show_plot = False,           # suppress interactive windows (CI-friendly)
)
```

---

## Adding Your Own Classifiers

Edit the `DEFAULT_CLASSIFIERS` dictionary near the top of `classifier_pipeline.py`:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

DEFAULT_CLASSIFIERS = {
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
    "SVM (linear)":        SVC(kernel="linear", C=1),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    # ── add below ──────────────────────────────────────────────────────────
    "MLP":                 MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500),
    "Naive Bayes":         GaussianNB(),
}
```

Any scikit-learn compatible estimator (implements `.fit()` and `.predict()`) works.

---

## Data Loader (Neuron CSVs)

`load_neuron_data()` expects files named:

```
count_neuronX_Texture{idx}_trial{t}_speed{s}_{type}_dirc{dir}.csv
count_neuronY_Texture{idx}_trial{t}_speed{s}_{type}_dirc{dir}.csv
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | (required) | Root folder containing CSV files |
| `texture_types` | `["circ","rect","wave"]` | Texture category strings |
| `texture_indices` | `[0,1,2,3,4]` | Integer texture IDs within each category |
| `speeds` | `[40, 60, 80]` | Stimulus speeds in mm/s |
| `directions` | `["X","Y"]` | Scan direction codes |
| `n_trials` | `10` | Repeated trials per condition |

The X- and Y-neuron count arrays are horizontally stacked (`np.hstack`) into a single feature vector per trial.

---

## Output Files

When `save_dir` is set, the following files are written:

```
outputs/
├── cm_KNN_(k=5)_–_Iris.png
├── cm_SVM_(linear)_–_Iris.png
├── cm_Random_Forest_–_Iris.png
│   …  (one PNG per classifier × task)
└── accuracy_comparison.png     ← grouped bar chart
```

---

## Code Structure

```
classifier_pipeline.py
│
├── ClassificationTask              dataclass – wraps X, y, split settings
│
├── DEFAULT_CLASSIFIERS             dict – the classifier catalogue
│
├── evaluate_classifier()           fit → predict → report → confusion matrix
├── run_task()                      loop evaluate_classifier over all classifiers
├── run_pipeline()                  loop run_task over all tasks + final plots
│
├── _plot_confusion_matrix()        seaborn heat-map helper
├── _plot_accuracy_comparison()     grouped bar chart helper
├── _print_summary()                formatted accuracy table to stdout
│
├── load_neuron_data()              project-specific CSV loader
│
└── __main__                        OPTION A (real data) / OPTION B (self-test)
```

### Key design choices

| Choice | Reason |
|--------|--------|
| `ClassificationTask` dataclass | Keeps features, labels, and split settings together — easy to pass around |
| `DEFAULT_CLASSIFIERS` dict at module level | One place to add/remove classifiers without touching evaluation logic |
| `save_dir` + `show_plot` flags | Same code works interactively and in headless CI/batch runs |
| `plt.close(fig)` after each plot | Prevents memory leaks when generating many plots in a loop |
| Sorted class labels in confusion matrix | Reproducible axis order regardless of label appearance order in data |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | `pip install numpy scikit-learn matplotlib seaborn` |
| `FileNotFoundError` on CSV | Check `data_dir` path and filename convention |
| `ConvergenceWarning` (Logistic Regression) | Increase `max_iter` in `DEFAULT_CLASSIFIERS` |
| Blank plot windows | Run `pip install pyqt5` for the Qt backend |
| Want to suppress all windows | Pass `show_plot=False` to `run_pipeline()` |
| Accuracy suspiciously high | Check for data leakage — features and labels should be independent |

---

## References

- scikit-learn user guide: <https://scikit-learn.org/stable/user_guide.html>
- Confusion matrix: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>
- seaborn heatmap: <https://seaborn.pydata.org/generated/seaborn.heatmap.html>

---

## License

MIT License — see [LICENSE](LICENSE) for details.
