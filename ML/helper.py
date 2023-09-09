import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error

def read_data(parent_file, file_list):
    df = pd.DataFrame()
    with open(file_list, "r") as data_file:
        for line in data_file:
            print(f"Reading {parent_file}/{line.strip()}...")
            df_temp = pd.read_csv(f"{parent_file}/{line.strip()}")
            print(f"Shape of {line.strip()} is {df_temp.shape}")
            df = pd.concat([df, df_temp], axis=0)
    return df

def feature_importance(base_model, x, y, dataset_partition_name):
    importances = permutation_importance(base_model,
                                         x,
                                         y,
                                         scoring="neg_root_mean_squared_error",
                                         n_jobs=-1,
                                         random_state=43)
    importances_df = pd.DataFrame(importances.importances.T, columns=x.columns)

    ax = sns.boxplot(data=importances_df.melt(),
                     y="variable",
                     x="value",
                     order=x.columns[np.flip(np.argsort(importances.importances_mean))])
    ax.set_xlabel("Average increase in RMSE score (ppm)")
    ax.set_ylabel("Feature Name")
    ax.set_title(f"Permutation Importances on {dataset_partition_name}")
    plt.show()

def validation_curve_constructor(base_model, x_train, y_train, param, search_range, default):
    train_scores, dev_scores = validation_curve(base_model,
                                                x_train,
                                                y_train,
                                                param_name=param,
                                                param_range=search_range,
                                                scoring="neg_root_mean_squared_error",
                                                n_jobs=-1,
                                                verbose=1)

    validation_score_df = pd.concat(
        [pd.DataFrame(
            train_scores.T,
            columns=search_range,
        ).assign(dataset="train"),
         pd.DataFrame(
             dev_scores.T,
             columns=search_range,
         ).assign(dataset="dev")
         ]
    )

    ax = sns.lineplot(data=validation_score_df.melt(id_vars="dataset"),
                      x="variable",
                      y="value",
                      hue="dataset")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Negative RMSE")
    ax.set_title(f"Validation curve for {param} (Red Line shows default value)")
    plt.axvline(default, color="red", label="Default Value")
    plt.show()

    return validation_score_df

def evaluator(model, x_dev, y_dev):
    y_pred = model.predict(x_dev)
    score = np.sqrt(mean_squared_error(y_dev, y_pred))

    print(f"RMSE for SearchCV Model is {score: .2f} ppm.")

    resids = y_pred - y_dev

    sns.residplot(x=np.linspace(0, 1, resids.size),
                  y=resids,
                  lowess=True,
                  line_kws={"color": "red", "linestyle": "-."},
                  scatter_kws={"alpha": 0.25, "color": "gray"})
    plt.show()

    print(resids.describe())


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, by=1, columns=None):
        self.by = by
        self.columns = columns

    def fit(self, X, y=None):
        print("fit")
        print(X)
        print(y)
        return self

    def transform(self, X, y=None):
        print("trans")
        print(X)
        print(y)
        return self