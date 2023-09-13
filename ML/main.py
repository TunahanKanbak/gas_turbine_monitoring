import pandas as pd
import numpy as np
import seaborn as sns
import helper as hp
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
import joblib

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)

data_file = "data_file_list.txt"

df = hp.read_data("datasets", data_file)
df.drop(columns=["CO"], inplace=True)  # Only NOx predictions will be carried out.

print(df.head())
print(df.shape)

target = "NOX"
indep_vars = df.columns.values.tolist()
indep_vars.remove(target)

########## Explanatory Data Analysis & Simple Preprocessing ############
print(df.info())  # No NaN values, every variable is float64 type, 8 independent variable
print(df.isnull().sum(axis=0))
print(df.describe())
# Ambient temperature (AT) C ! Has reasonable data range
# Ambient pressure (AP) mbar ! Has reasonable data range
# Ambient humidity (AH) (%) ! Has reasonable data range (supersaturated air may lead to AH higher than 100%)
# Air filter difference pressure (AFDP) mbar ! Has reasonable data range
# Gas turbine exhaust pressure (GTEP) mbar ! Has reasonable data range
# Turbine inlet temperature (TIT) C ! Has reasonable data range
# Turbine after temperature (TAT) C ! Has reasonable data range
# Turbine energy yield (TEY) MWH ! Has reasonable data range
# Compressor discharge pressure (CDP) bar ! Has reasonable data range
# Nitrogen oxides (NOx) mg/m3 ! Has reasonable data range

df["CDP"] = df["CDP"] * 1000  # Conversion to mbar
df["AT"] = df["AT"] + 273  # Conversion to Kelvin
df["TIT"] = df["TIT"] + 273  # Conversion to Kelvin
df["TAT"] = df["TAT"] + 273  # Conversion to Kelvin

sns.clustermap(df.corr(), annot=True)  # Linear correlations!
# Strong correlations exist btw CDP, GTEP, TEY and TIT which are all main operation parameters.
# Target param (NOx) has highest correlation with AT.
plt.show()

mutual_info = mutual_info_regression(df[indep_vars], df[target], random_state=2023)
ax=sns.barplot(y=indep_vars,
            x=mutual_info,
            order=np.array(indep_vars)[np.flip(np.argsort(mutual_info))])
ax.set_xlabel("Mutual Information")
ax.set_ylabel("Features")
ax.set_title("NOx-Feature Relation")
plt.show()
# If we consider non-linear correlations, main operation parameters are also have high correlation with target param
# as high as AT.

# Independent Variable distribution - Violinplot
fig, ax = plt.subplots(3, 3)
for var, ax in zip(indep_vars, ax.flatten()):
    sns.violinplot(data=df, y=var, ax=ax)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis="both", which="major", labelsize=15)
fig.set_size_inches(15, 15)
plt.tight_layout()
plt.show()
# Independent Variable distribution - Boxplot
fig, ax = plt.subplots(3, 3)
for var, ax in zip(indep_vars, ax.flatten()):
    sns.boxplot(data=df, y=var, ax=ax)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis="both", which="major", labelsize=15)
fig.set_size_inches(15, 15)
plt.tight_layout()
plt.show()
# Target Variable distribution
fig, ax = plt.subplots(1, 2)
sns.violinplot(data=df, y="NOX", ax=ax[0])
sns.boxplot(data=df, y="NOX", ax=ax[1])
ax[0].yaxis.label.set_fontsize(15)
ax[0].tick_params(axis="y", which="major", labelsize=10)
plt.show()

# When indep_vars examined with individual boxplot, each variable contains several outlier values. However, this
# outliers may be reasonable according to other operation conditions. To examine this, we may use LOF.
""""
outlier_classifier = LocalOutlierFactor(n_neighbors=20)
lof_labels = outlier_classifier.fit_predict(df)
lof_scores = outlier_classifier.negative_outlier_factor_

df = df.assign(lof_labels=lof_labels, lof_scores=-1 * lof_scores)

print(df.lof_labels.value_counts())  # -1 means outlier, 1 means inlier
#### DR with t-SNE
scaler_model = StandardScaler()
transformed_space = scaler_model.fit_transform(df[indep_vars + [target]])

tsne_model = TSNE(perplexity=150, early_exaggeration=50, n_jobs=-1)
reduced_space = tsne_model.fit_transform(transformed_space)

reduced_space_df = pd.DataFrame().assign(tSNE1=reduced_space[:, 0],
                                         tSNE2=reduced_space[:, 1],
                                         lof_labels=lof_labels,
                                         lof_scores=-1 * lof_scores)

ax = sns.scatterplot(data=reduced_space_df[reduced_space_df.lof_labels == 1],
                x="tSNE1",
                y="tSNE2",
                alpha=0.2,
                color="grey")

sns.scatterplot(data=reduced_space_df[reduced_space_df.lof_labels == -1],
                x="tSNE1",
                y="tSNE2",
                color="red",
                ax=ax)
plt.legend(title='LOF Analysis', loc='upper left', labels=['Inliers', 'Outliers'])
plt.title("tSNE Dimension Reduction")
plt.show()

df = df[df.lof_labels == 1]  # Remove outliers according to LOF
df.drop(columns=["lof_scores", "lof_labels"], inplace=True)
"""
## Relation with Target ##
fig, ax = plt.subplots(3, 3, sharey=True)
for var, ax in zip(indep_vars, ax.flatten()):
    sns.scatterplot(data=df, x=var, y="NOX", ax=ax, alpha=0.2)
    ax.yaxis.label.set_fontsize(25)
    ax.xaxis.label.set_fontsize(25)
    ax.tick_params(axis="both", which="major", labelsize=15)
fig.set_size_inches(15, 15)
plt.tight_layout()
plt.show()

NOX_levels = pd.qcut(df.NOX, 3)
sns.jointplot(data=df, x="CDP", y="TIT", hue=NOX_levels, hue_order=NOX_levels.unique(), kind="scatter")
plt.show()

sns.jointplot(data=df, x="CDP", y="AT", hue=NOX_levels, hue_order=NOX_levels.unique(), kind="scatter")
plt.show()

sns.jointplot(data=df, x="TIT", y="TAT", hue=NOX_levels, hue_order=NOX_levels.unique(), kind="scatter")
plt.show()

########## Feature Engineering ############

## Compressor Exit Temperature via Isentropic Compression
# Ref: https://www.grc.nasa.gov/www/k-12/airplane/compth.html

gamma_air = 1.4  # Usual value to assume for specific heats ratio of air
df["TAC"] = np.power(df["CDP"] / df["AP"], (gamma_air - 1) / gamma_air) * df["AT"]

## Brayton Cycle Efficeincy via Temperature Changes
# Ref: https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node27.html

df["IBCE"] = 1 - df["AT"] / df["TAC"]  # Ideal Brayton Cycle Efficiency
df["RBCE"] = 1 - (df["TAT"] - 1 / df["AT"]) / (df["TIT"] - 1 / df["TAC"])  # Real Brayton Cycle Efficiency
df["ER"] = df["RBCE"] / df["IBCE"]  # Real operation efficiency ratio

## Square root of CDP
# Ref: https://www.ge.com/content/dam/gepower-new/global/en_US/downloads/gas-new-site/resources/reference/ger-4211-gas-turbine-emissions-and-control.pdf

df["SRCDP"] = np.sqrt(df["CDP"])  # Square root of CDP

## Mutual Informations with new features
indep_vars = df.columns.values.tolist()
indep_vars.remove(target)

mutual_info = mutual_info_regression(df[indep_vars], df[target], random_state=2023)
sns.barplot(y=indep_vars, x=mutual_info, order=np.array(indep_vars)[np.flip(np.argsort(mutual_info))])
plt.show()

####### Model Development ########
#### Base Model Selection & Inspection ####
X = df[indep_vars]
y = df[target]

scaler = StandardScaler()

model_list = [LinearRegression(),
              Ridge(random_state=43),
              LinearSVR(random_state=43),
              RandomForestRegressor(random_state=43),
              HistGradientBoostingRegressor(random_state=43),
              GradientBoostingRegressor(random_state=43)]

x_train, x_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=43)
x_dev, x_test, y_dev, y_test = train_test_split(x_dev, y_dev, test_size=0.5, random_state=43)

#Outlier Detection with LOF
outlier_classifier = LocalOutlierFactor(n_neighbors=20)
lof_labels = outlier_classifier.fit_predict(StandardScaler().fit_transform(x_train))
lof_scores = outlier_classifier.negative_outlier_factor_

#Visualization of data in reduced space, so we can see how outliers are arranged
reduced_space = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(x_train))

reduced_space_df = pd.DataFrame().assign(PCA1=reduced_space[:, 0],
                                         PCA2=reduced_space[:, 1],
                                         lof_labels=lof_labels,
                                         lof_scores=-1 * lof_scores)

ax = sns.scatterplot(data=reduced_space_df[reduced_space_df.lof_labels == 1],
                x="PCA1",
                y="PCA2",
                alpha=0.2,
                color="grey")

sns.scatterplot(data=reduced_space_df[reduced_space_df.lof_labels == -1],
                x="PCA1",
                y="PCA2",
                color="red",
                ax=ax)
plt.legend(title='LOF Analysis', loc='upper left', labels=['Inliers', 'Outliers'])
plt.title("PCA Dimension Reduction")
plt.show()

#Compare distribution of outliers to compare it with original distribution
fig, ax = plt.subplots(3, 3)
for var, ax in zip(indep_vars, ax.flatten()):
    sns.boxplot(data=x_train[lof_labels == -1], y=var, ax=ax)
    ax.yaxis.label.set_fontsize(25)
    ax.tick_params(axis="both", which="major", labelsize=15)
fig.set_size_inches(15, 15)
plt.tight_layout()
plt.show()
#Decide not to remove outliers
#x_train, y_train = x_train[lof_labels == 1], y_train[lof_labels == 1]

base_model_rmse = {}
for model in model_list:
    pipe = Pipeline([
        ("scaler", scaler),
        ("model", model)
    ])

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_dev)

    score = np.sqrt(mean_squared_error(y_dev, y_pred))
    base_model_rmse[model.__class__.__name__] = score

    print(f"RMSE for {model.__class__.__name__} Model is {score: .2f} ppm.")

# Refit Best Base Model
best_model = HistGradientBoostingRegressor(random_state=43)
pipe = Pipeline([
    ("scaler", scaler),
    ("model", best_model)
])
pipe.fit(x_train, y_train)

# Inspect feature importances according to the best base model
hp.feature_importance(pipe, x_train, y_train, "Train Set")
hp.feature_importance(pipe, x_dev, y_dev, "Dev Set")

#### Hyperparameter Optimization of Chosen Model ####
# Grid Construction
param_grid = {
    "model__max_iter": np.logspace(1,3.5,20).astype(int),
    "model__learning_rate": np.logspace(-3, 0, 20),
    "model__max_bins": np.logspace(2, 8, 15, base=2).astype(int) - 1,
    "model__max_depth": np.linspace(1, 15, 15).astype(int),
    "model__min_samples_leaf": np.linspace(5, 100, 20).astype(int),
    "model__l2_regularization": np.logspace(-1, 1.5, 10)
}

defaults_params = {
    "model__max_iter": 100,
    "model__learning_rate": 0.1,
    "model__max_bins": 255,
    "model__max_depth": 1,
    "model__min_samples_leaf": 20,
    "model__l2_regularization": 0
}

#for param, param_space in param_grid.items():
#    hp.validation_curve_constructor(pipe,
#                                    x_train,
#                                    y_train,
#                                    param,
#                                    param_space,
#                                    defaults_params[param])

# Randomized Search
param_grid = {
    "model__max_iter": np.logspace(1, 3, 20).astype(int),
    "model__learning_rate": np.logspace(-1.5, 0, 20),
    "model__max_depth": np.linspace(4, 10, 7).astype(int),
    "model__min_samples_leaf": np.linspace(5, 100, 10).astype(int)
}

random_search = RandomizedSearchCV(pipe,
                                   param_distributions=param_grid,
                                   n_iter=2500,
                                   scoring="neg_root_mean_squared_error",
                                   n_jobs=-1,
                                   verbose=1,
                                   random_state=43)

#random_search.fit(x_train, y_train)

#joblib.dump(random_search, "RandomizedSearchCV.pkl")
random_search = joblib.load("RandomizedSearchCV.pkl")

cv_results_random = pd.DataFrame(random_search.cv_results_)
[print(f"(Random Search) Best parameter for {key} is {value}") for key, value in random_search.best_params_.items()]

print("################ DEV SET RESULTS ################")
hp.evaluator(random_search, x_dev, y_dev)

for param in param_grid.keys():
    sns.lineplot(data=cv_results_random, x=f"param_{param}", y="mean_test_score")
    plt.show()

# Grid Search
param_grid = {
    "model__max_iter": [1500],
    "model__learning_rate": np.logspace(-2, -0.3, 10),
    "model__max_depth": np.linspace(8, 15, 8).astype(int),
    "model__min_samples_leaf": np.linspace(15, 32, 3).astype(int)
}

grid_search = GridSearchCV(pipe,
                           param_grid=param_grid,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=-1,
                           verbose=1)

#grid_search.fit(x_train, y_train)

#joblib.dump(grid_search, "GridSearchCV.pkl")
grid_search = joblib.load("GridSearchCV.pkl")

cv_results_grid = pd.DataFrame(grid_search.cv_results_)
[print(f"(Grid Search) Best parameter for {key} is {value}") for key, value in grid_search.best_params_.items()]

print("################ DEV SET RESULTS ################")
hp.evaluator(grid_search, x_dev, y_dev)

for param in param_grid.keys():
    sns.lineplot(data=cv_results_grid, x=f"param_{param}", y="mean_test_score")
    plt.show()

print("################ TRAIN SET RESULTS ################")
hp.evaluator(random_search, x_train, y_train)
hp.evaluator(grid_search, x_train, y_train)

print("################ DEV SET RESULTS ################")
hp.evaluator(random_search, x_dev, y_dev)
hp.evaluator(grid_search, x_dev, y_dev)

print("################ TEST SET RESULTS ################")
hp.evaluator(random_search, x_test, y_test)
hp.evaluator(grid_search, x_test, y_test)

hp.feature_importance(grid_search, x_train, y_train, "Train Set")
hp.feature_importance(grid_search, x_dev, y_dev, "Dev Set")

#### Partial Dependence ####
fig, ax = plt.subplots()
ax = PartialDependenceDisplay.from_estimator(estimator=grid_search,
                                             X=x_dev,
                                             features=["AT", "TAC", "TAT", "TEY", "ER", "IBCE"],
                                             n_jobs=-1,
                                             kind="both",
                                             ax=ax,
                                             grid_resolution=250,
                                             random_state=43)
ax.axes_[0][0].set_ylabel("NOx (ppm)")
ax.axes_[1][0].set_ylabel("NOx (ppm)")
fig.set_size_inches(15, 10)
plt.show()