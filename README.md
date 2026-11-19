# Structure-Preserving Multi-Task Label Distribution Learning (MTLDL)

This repository contains a Python implementation of **Structure-Preserving Multi-Task Label Distribution Learning via Shared–Specific Subspace Learning**.

## 1. Overview

MTLDL learns label distributions for multiple related tasks that share the same feature and label spaces. A shared mapping captures information common to all tasks, while task-specific mappings capture each task's individual characteristics. The resulting representations are combined for label distribution prediction.

The model jointly learns the shared and task-specific mappings, prediction weights, and task-wise similarity graphs through alternating optimization. Graph regularization preserves sample relationships in the original feature space, the learned representation space, and the linear prediction space. Predicted label distributions are obtained by projecting the linear outputs onto the probability simplex.

This implementation uses blockwise Duchi simplex projection for similarity updates and reuses the shared intermediate matrices in the mapping gradients.

FBP5500 is included as the demo dataset, with four tasks corresponding to the `AF`, `AM`, `CF`, and `CM` sample blocks.

## 2. Usage

### Prerequisites

- Python 3.10 or later.
- NumPy `>=1.26,<3` and SciPy `>=1.11,<2`.

Run the commands below from the directory containing `train.py` and `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

The implementation uses NumPy and SciPy for CPU computation.

### Prepare the data

The bundled demo file `dataset/FBP5500.mat` contains the following two matrices.

| Variable | Shape | Description |
| --- | --- | --- |
| `features` | `n_samples × n_features` | Precomputed feature vectors, with one sample per row. |
| `labels` | `n_samples × n_labels` | Nonnegative label distributions, with each row summing to 1. |

The file contains 5,500 samples with 256 features and five label-description columns per sample. Samples from each task occupy one contiguous row block, in the following order.

| Task | Number of samples | MAT row range (1-based) |
| --- | --- | --- |
| `AF` | 2,000 | 1–2,000 |
| `AM` | 2,000 | 2,001–4,000 |
| `CF` | 750 | 4,001–4,750 |
| `CM` | 750 | 4,751–5,500 |

The release has the following main files.

```text
MTLDL_FBP5500_demo/
├── README.md
├── requirements.txt
├── train.py
├── experiment.py
├── mtldl.py
├── LoadDataSet.py
├── InitSandUpdateS.py
├── ldl_metrics.py
└── dataset/
    └── FBP5500.mat
```

The default path is resolved relative to `train.py`. Use `--data` to specify another copy of the FBP5500 MAT file. If the task blocks follow a different order, specify their actual order with `--task-names`.

### Run ten-fold cross-validation

Run the FBP5500 demo with fixed parameters.

```bash
python train.py --output results/fbp5500.json
```

The command uses the bundled data with `C=1`, `beta1=0.1`, `beta3=0.1`, shared and task-specific dimensions of 50, and seed 0. Set these values explicitly as follows.

```bash
python train.py --dataset fbp5500 --dimension 50 --c 1 --beta1 0.1 --beta3 0.1 --seed 0 --output results/fbp5500.json
```

For example, an explicit data path can be provided as follows.

```bash
python train.py --data /path/to/FBP5500.mat --output results/fbp5500.json
```

Each task is shuffled and divided into ten folds using the specified seed. Each outer fold trains on nine parts and evaluates the remaining part. `--dimension` sets both the shared and task-specific subspace dimensions. Its default is 50, and it must not exceed the input feature dimension. The default maximum is ten alternating iterations, with early stopping controlled by `--tolerance`.

 
### Evaluation results

The command prints the mean and sample standard deviation across the ten outer folds for each task and for `ALL`. Within each fold, `ALL` weights the task scores by their numbers of test samples.

| Metric | Preferred direction |
| --- | --- |
| Chebyshev distance | Lower |
| Clark distance | Lower |
| Canberra distance | Lower |
| Kullback–Leibler divergence | Lower |
| Cosine similarity | Higher |
| Intersection similarity | Higher |

The JSON output records the fold-specific parameters, training losses, test indices, test scores, and summary statistics. With `--tune`, it also records the inner validation KL for every candidate.

 

## 3. Citation

If you find this work useful, please cite the accompanying manuscript.

```bibtex
@unpublished{xiao_mtldl,
  author = {Yanshan Xiao and Rui Dong and Bo Liu},
  title  = {Structure-Preserving Multi-Task Label Distribution Learning via Shared--Specific Subspace Learning},
  note   = {International Journal of Machine Learning and Cybernetics}
}
```
