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

## Notebooks

```bash
make notebooks
uv run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5050
```

`notebooks/eda.ipynb` checks data quality and plots the training partition. `notebooks/modelling.ipynb` compares a mean baseline, linear regression and a random forest, recording the runs in MLflow. The download checks the dataset's SHA-256 before use.

Two zero-height rows are excluded, leaving 4,175 records. With seed 42, training, validation and test contain 2,505, 835 and 835 rows. The random forest has the lowest validation RMSE (2.154 rings, compared with 2.249 for linear regression and 3.198 for the baseline). After refitting on development data, its test RMSE is 2.222 rings and MAE is 1.542 rings. These scores describe a random holdout of this dataset, not performance on a new population.
