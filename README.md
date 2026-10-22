# Abalone ring prediction

HEC MLOps project, completed individually with the instructor's agreement.

Author: Rémi Barbier, GitHub `RemiBp`.

## Environment

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
make setup
make check
```

The lockfile fixes dependency versions. Prefect 2.13.7 and MLflow 2.7.1 follow the course environment. Runtime data, trained objects and experiment files stay outside version control.

## Project steps

The five numbered branches cover environment setup, notebooks, training modules, Prefect orchestration and the Docker API. Each is integrated into `main` through a separate pull request, preserving commits.

Dataset: [Abalone on Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset). The target is the number of shell rings. Approximate age in years is rings plus 1.5.
