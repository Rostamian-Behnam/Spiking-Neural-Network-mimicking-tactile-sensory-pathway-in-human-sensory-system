"""
classifier_pipeline.py
=======================
A general-purpose, reusable machine-learning classification pipeline.

Supports any number of classification *tasks* (e.g. texture / speed /
direction), any scikit-learn compatible classifier, and produces:
    • Per-classifier console reports (accuracy, precision, recall, F1)
    • Confusion-matrix heat-maps (saved as PNG or shown interactively)
    • A side-by-side accuracy bar-chart across all tasks and classifiers

The pipeline is completely data-agnostic: swap in your own feature matrix
and label vector via the ``ClassificationTask`` dataclass.

Dependencies
------------
    pip install numpy scikit-learn matplotlib seaborn

Typical usage
-------------
    # 1. Drop-in your own X and y, then run:
    python classifier_pipeline.py

    # 2. Import the helpers in a notebook:
    from classifier_pipeline import run_pipeline, ClassificationTask

Author  : <your name>
License : MIT
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ===========================================================================
# 1.  DATA STRUCTURES
# ===========================================================================

@dataclass
class ClassificationTask:
    """
    Container that pairs a feature matrix with a label vector for ONE task.

    Parameters
    ----------
    name : str
        Human-readable task name used in plot titles and console output.
        Example: ``"Texture"``, ``"Speed"``, ``"Direction"``
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Class labels (integers or strings).
    test_size : float, optional
        Fraction of data held out for testing.  Default 0.3.
    random_state : int, optional
        Seed for reproducible train/test splits.  Default 42.

    Example
    -------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> task = ClassificationTask(name="Iris", X=iris.data, y=iris.target)
    >>> task.name
    'Iris'
    """
    name:         str
    X:            np.ndarray
    y:            np.ndarray
    test_size:    float = 0.3
    random_state: int   = 42

    def split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_test, y_train, y_test)."""
        return train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )


# ---------------------------------------------------------------------------
# Default classifier catalogue
# ---------------------------------------------------------------------------
# Each value is a freshly-constructed, unfitted scikit-learn estimator.
# Add, remove, or swap classifiers here without touching any other code.
# ---------------------------------------------------------------------------
DEFAULT_CLASSIFIERS: Dict = {
    "KNN (k=5)":          KNeighborsClassifier(n_neighbors=5),
    "SVM (linear)":       SVC(kernel="linear", C=1),
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}


# ===========================================================================
# 2.  CORE PIPELINE FUNCTIONS
# ===========================================================================

def evaluate_classifier(
    clf,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    clf_name:  str,
    task_name: str,
    save_dir:  Optional[str] = None,
    show_plot: bool = True,
) -> float:
    """
    Fit *clf* on the training split, predict on the test split, and report.

    Side-effects
    ------------
    • Prints a scikit-learn classification report to stdout.
    • Draws (and optionally saves) a confusion-matrix heat-map.

    Parameters
    ----------
    clf        : sklearn estimator  – unfitted classifier instance.
    X_train, X_test, y_train, y_test : np.ndarray
        Pre-split feature/label arrays.
    clf_name   : str  – label for the classifier (e.g. ``"SVM"``).
    task_name  : str  – label for the task (e.g. ``"Texture"``).
    save_dir   : str or None
        If given, confusion-matrix PNGs are saved here instead of shown.
    show_plot  : bool
        Set to ``False`` to suppress interactive plot windows (useful in CI).

    Returns
    -------
    float
        Test accuracy in [0, 1].

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> iris = load_iris()
    >>> Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target)
    >>> acc = evaluate_classifier(
    ...     KNeighborsClassifier(), Xtr, Xte, ytr, yte,
    ...     "KNN", "Iris", show_plot=False
    ... )
    >>> 0 <= acc <= 1
    True
    """
    # -- Train ---------------------------------------------------------------
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # -- Metrics -------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {clf_name}  |  Task: {task_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    # -- Confusion matrix ----------------------------------------------------
    _plot_confusion_matrix(
        y_test, y_pred,
        title=f"{clf_name}  –  {task_name}",
        save_dir=save_dir,
        show_plot=show_plot,
    )

    return accuracy


def run_task(
    task:        ClassificationTask,
    classifiers: Dict,
    save_dir:    Optional[str] = None,
    show_plot:   bool = True,
) -> Dict[str, float]:
    """
    Run every classifier in *classifiers* against a single ``ClassificationTask``.

    Parameters
    ----------
    task        : ClassificationTask  – holds X, y and split settings.
    classifiers : dict                – ``{name: unfitted_estimator}``.
    save_dir    : str or None         – folder for PNG output (see ``evaluate_classifier``).
    show_plot   : bool                – passed through to ``evaluate_classifier``.

    Returns
    -------
    dict
        ``{classifier_name: accuracy}`` for this task.

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> task = ClassificationTask("Iris", iris.data, iris.target)
    >>> accs = run_task(task, DEFAULT_CLASSIFIERS, show_plot=False)
    >>> list(accs.keys())
    ['KNN (k=5)', 'SVM (linear)', 'Random Forest', 'Logistic Regression']
    """
    print(f"\n{'#'*60}")
    print(f"  TASK : {task.name.upper()}")
    print(f"{'#'*60}")

    X_train, X_test, y_train, y_test = task.split()
    accuracies: Dict[str, float] = {}

    for clf_name, clf in classifiers.items():
        accuracies[clf_name] = evaluate_classifier(
            clf, X_train, X_test, y_train, y_test,
            clf_name=clf_name,
            task_name=task.name,
            save_dir=save_dir,
            show_plot=show_plot,
        )

    return accuracies


def run_pipeline(
    tasks:       Sequence[ClassificationTask],
    classifiers: Optional[Dict] = None,
    save_dir:    Optional[str]  = None,
    show_plot:   bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run the full pipeline: all classifiers × all tasks.

    Parameters
    ----------
    tasks       : list of ClassificationTask
        One entry per classification problem (texture, speed, direction, …).
    classifiers : dict or None
        ``{name: estimator}`` mapping.  Defaults to ``DEFAULT_CLASSIFIERS``.
    save_dir    : str or None
        Directory for saving confusion-matrix PNGs.  Created if missing.
    show_plot   : bool
        ``False`` suppresses all ``plt.show()`` calls (useful for scripting).

    Returns
    -------
    dict
        ``{task_name: {clf_name: accuracy}}`` – full result matrix.

    Example
    -------
    >>> from sklearn.datasets import load_iris, load_wine
    >>> iris = load_iris();  wine = load_wine()
    >>> tasks = [
    ...     ClassificationTask("Iris", iris.data, iris.target),
    ...     ClassificationTask("Wine", wine.data, wine.target),
    ... ]
    >>> results = run_pipeline(tasks, show_plot=False)
    >>> "Iris" in results and "Wine" in results
    True
    """
    if classifiers is None:
        classifiers = DEFAULT_CLASSIFIERS

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    all_results: Dict[str, Dict[str, float]] = {}

    for task in tasks:
        all_results[task.name] = run_task(
            task, classifiers, save_dir=save_dir, show_plot=show_plot,
        )

    # -- Summary table -------------------------------------------------------
    _print_summary(all_results)
    _plot_accuracy_comparison(all_results, save_dir=save_dir, show_plot=show_plot)

    return all_results


# ===========================================================================
# 3.  PLOTTING HELPERS  (private – prefix _ means "internal use")
# ===========================================================================

def _plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    title:     str,
    save_dir:  Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Draw a labelled confusion-matrix heat-map.

    Parameters
    ----------
    y_true, y_pred : array-like  – ground-truth and predicted labels.
    title          : str         – figure title and (optionally) filename stem.
    save_dir       : str or None – if given, save PNG here; else show interactively.
    show_plot      : bool        – set False to suppress the window.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true))           # class labels in alphabetical order

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    plt.tight_layout()

    if save_dir:
        # Replace characters that are unsafe in filenames
        safe_title = title.replace(" ", "_").replace("|", "").replace("/", "-")
        path = os.path.join(save_dir, f"cm_{safe_title}.png")
        fig.savefig(path, dpi=150)
        print(f"  [saved] {path}")

    if show_plot:
        plt.show()

    plt.close(fig)


def _plot_accuracy_comparison(
    results:   Dict[str, Dict[str, float]],
    save_dir:  Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Grouped bar chart: classifiers (x-axis) × tasks (colour groups).

    Parameters
    ----------
    results   : nested dict  ``{task_name: {clf_name: accuracy}}``.
    save_dir  : str or None  – PNG output folder.
    show_plot : bool         – suppress interactive window when False.
    """
    task_names = list(results.keys())
    clf_names  = list(next(iter(results.values())).keys())

    n_tasks = len(task_names)
    n_clfs  = len(clf_names)
    x       = np.arange(n_clfs)
    width   = 0.8 / n_tasks          # bar width shrinks as tasks increase
    colours = plt.cm.tab10.colors    # up to 10 distinct colours

    fig, ax = plt.subplots(figsize=(max(8, n_clfs * 2), 5))

    for i, task_name in enumerate(task_names):
        accs   = [results[task_name][clf] for clf in clf_names]
        offset = (i - n_tasks / 2 + 0.5) * width
        bars   = ax.bar(x + offset, accs, width, label=task_name, color=colours[i % 10])

        # Annotate each bar with its percentage
        for bar, acc in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc*100:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classifier Accuracy Comparison Across Tasks", fontsize=13, fontweight="bold")
    ax.legend(title="Task", loc="upper right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "accuracy_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"\n  [saved] {path}")

    if show_plot:
        plt.show()

    plt.close(fig)


def _print_summary(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted accuracy table to stdout.

    Parameters
    ----------
    results : nested dict  ``{task_name: {clf_name: accuracy}}``.
    """
    print(f"\n{'='*60}")
    print("  SUMMARY – Test Accuracy")
    print(f"{'='*60}")
    clf_names = list(next(iter(results.values())).keys())

    # Header row
    col_w = 22
    header = f"{'Classifier':<{col_w}}" + "".join(f"{t:>12}" for t in results)
    print(header)
    print("-" * len(header))

    for clf in clf_names:
        row = f"{clf:<{col_w}}"
        for task_name in results:
            acc = results[task_name].get(clf, float("nan"))
            row += f"{acc*100:>11.2f}%"
        print(row)
    print()


# ===========================================================================
# 4.  DATA LOADER  (project-specific – replace with your own loader)
# ===========================================================================

def load_neuron_data(
    data_dir:       str,
    texture_types:  List[str]  = ("circ", "rect", "wave"),
    texture_indices: List[int] = (0, 1, 2, 3, 4),
    speeds:         List[int]  = (40, 60, 80),
    directions:     List[str]  = ("X", "Y"),
    n_trials:       int        = 10,
) -> Tuple[
    List[np.ndarray], List[str],   # texture
    List[np.ndarray], List[int],   # speed
    List[np.ndarray], List[str],   # direction
]:
    """
    Load passive-touch neuron spike-count CSVs and build three feature/label sets.

    File naming convention expected on disk::

        count_neuronX_Texture{idx}_trial{t}_speed{s}_{type}_dirc{dir}.csv
        count_neuronY_Texture{idx}_trial{t}_speed{s}_{type}_dirc{dir}.csv

    The two CSV files (X-axis and Y-axis neurons) are horizontally stacked
    into a single feature vector per sample.

    Parameters
    ----------
    data_dir        : root folder containing all CSV files.
    texture_types   : list of texture category strings.
    texture_indices : integer indices within each category.
    speeds          : list of speed values (mm/s).
    directions      : list of scan direction codes.
    n_trials        : number of repeated trials per condition.

    Returns
    -------
    Six lists (features, labels) for the texture, speed, and direction tasks.

    Raises
    ------
    FileNotFoundError
        If any expected CSV file is missing.
    """

    def _load_one(tex_type, tex_idx, trial, speed, direct):
        """Load and concatenate the X- and Y-neuron spike counts for one trial."""
        def _path(axis):
            return os.path.join(
                data_dir,
                f"count_neuron{axis}_Texture{tex_idx}_trial{trial}"
                f"_speed{speed}_{tex_type}_dirc{direct}.csv",
            )
        counts_x = np.loadtxt(_path("X"), delimiter=",")
        counts_y = np.loadtxt(_path("Y"), delimiter=",")
        return np.hstack([counts_x, counts_y])

    tex_feat, tex_lbl = [], []
    spd_feat, spd_lbl = [], []
    dir_feat, dir_lbl = [], []

    for tex_type in texture_types:
        for tex_idx in texture_indices:
            for speed in speeds:
                for direct in directions:
                    for trial in range(n_trials):
                        data = _load_one(tex_type, tex_idx, trial, speed, direct)

                        tex_feat.append(data);  tex_lbl.append(f"{tex_type}_{tex_idx}")
                        spd_feat.append(data);  spd_lbl.append(speed)
                        dir_feat.append(data);  dir_lbl.append(direct)

    return tex_feat, tex_lbl, spd_feat, spd_lbl, dir_feat, dir_lbl


# ===========================================================================
# 5.  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # OPTION A – Use your real data
    # -----------------------------------------------------------------------
    # Uncomment and set DATA_DIR to your actual folder, then run the script.
    #
    # DATA_DIR = r"H:\Test Passive Touch"
    #
    # (tex_X, tex_y,
    #  spd_X, spd_y,
    #  dir_X, dir_y) = load_neuron_data(DATA_DIR)
    #
    # tasks = [
    #     ClassificationTask("Texture",   np.array(tex_X), np.array(tex_y)),
    #     ClassificationTask("Speed",     np.array(spd_X), np.array(spd_y)),
    #     ClassificationTask("Direction", np.array(dir_X), np.array(dir_y)),
    # ]

    # -----------------------------------------------------------------------
    # OPTION B – Quick self-test with built-in sklearn datasets (no files needed)
    # -----------------------------------------------------------------------
    # Remove this block when using real data.
    # -----------------------------------------------------------------------
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer

    iris   = load_iris()
    wine   = load_wine()
    cancer = load_breast_cancer()

    tasks = [
        ClassificationTask("Iris (texture proxy)",   iris.data,   iris.target),
        ClassificationTask("Wine (speed proxy)",     wine.data,   wine.target),
        ClassificationTask("Cancer (direction proxy)", cancer.data, cancer.target),
    ]

    # -----------------------------------------------------------------------
    # Run the full pipeline
    # -----------------------------------------------------------------------
    results = run_pipeline(
        tasks       = tasks,
        classifiers = DEFAULT_CLASSIFIERS,
        save_dir    = "outputs",   # set to None to skip saving PNGs
        show_plot   = True,        # set to False for headless / CI runs
    )
