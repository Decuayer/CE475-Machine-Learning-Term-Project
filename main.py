import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import RobustScaler
import sklearn.base

# Models 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import HuberRegressor

# CONFIGURATION
DATA_FILE = "data.xlsx"
SHEET_NAME = "Data"
OUTPUT_CSV = "predictions.csv"
PLOT_DIR = "plots"
RANDOM_STATE = 42
CV_FOLDS = 5

np.random.seed(RANDOM_STATE)


# DATA LOADING
def load_data_from_excel(file_path: str):
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet = workbook[SHEET_NAME]

    X_train = np.array(
        [[cell.value for cell in row] for row in sheet.iter_rows(
            min_row=2, max_row=101, min_col=2, max_col=7
        )],
        dtype=np.float64,
    )
    y_train = np.array(
        [sheet.cell(row=r, column=8).value for r in range(2, 102)],
        dtype=np.float64,
    )

    X_test = np.array(
        [[cell.value for cell in row] for row in sheet.iter_rows(
            min_row=102, max_row=121, min_col=2, max_col=7
        )],
        dtype=np.float64,
    )

    workbook.close()
    return X_train, y_train, X_test


# PREPROCESSING
def preprocess(X_train_raw, X_test_raw):

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_test_sc = scaler.transform(X_test_raw)

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train_sc)
    X_test_poly = poly.transform(X_test_sc)

    return X_train_sc, X_test_sc, X_train_poly, X_test_poly, scaler, poly


#  MODEL DEFINITIONS
def get_models():
    base_models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        "Huber Regression": HuberRegressor(max_iter=2000),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "SVR (RBF Kernel)": SVR(kernel="rbf", C=100, epsilon=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0),
    }
    
    def symlog(x):
        return np.sign(x) * np.log1p(np.abs(x))

    def symexp(x):
        return np.sign(x) * np.expm1(np.abs(x))

    return {
        name: TransformedTargetRegressor(
            regressor=m, 
            func=symlog, 
            inverse_func=symexp,
            check_inverse=True  # scikit-learn'ün doğrulamadan geçmesini sağlar
        ) 
        for name, m in base_models.items()
    }


# EVALUATION (5-Fold Cross-Validation)
def evaluate_models_cv(models: dict, X, y, is_linear_dict, X_poly, cv_folds=CV_FOLDS):

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    records = []
    for name, model in models.items():
        X_data = X_poly if is_linear_dict[name] else X

        # MAE
        mae_scores = -cross_val_score(model, X_data, y, cv=kf, scoring="neg_mean_absolute_error")
        mse_scores = -cross_val_score(model, X_data, y, cv=kf, scoring="neg_mean_squared_error")
        rmse_scores = np.sqrt(mse_scores)
        r2_scores = cross_val_score(model, X_data, y, cv=kf, scoring="r2")

        records.append({
            "Model": name,
            "MAE (mean)": mae_scores.mean(),
            "MAE (std)": mae_scores.std(),
            "RMSE (mean)": rmse_scores.mean(),
            "RMSE (std)": rmse_scores.std(),
            "R² (mean)": r2_scores.mean(),
            "R² (std)": r2_scores.std(),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("MAE (mean)").reset_index(drop=True)
    return df


# HOLDOUT EVALUATION (for diagnostics / visualisation)
def holdout_evaluate(models: dict, X, y, test_fraction=0.2):
    split = int(len(X) * (1 - test_fraction))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        results[name] = (model, y_te, y_pred)
    return results


# PLOTTING UTILITIES
def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_model_comparison(cv_results: pd.DataFrame):
    ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("viridis", len(cv_results))
    bars = ax.barh(
        cv_results["Model"], cv_results["MAE (mean)"],
        xerr=cv_results["MAE (std)"], color=colors, edgecolor="black",
        capsize=4,
    )
    ax.set_xlabel("Mean Absolute Error (5-Fold CV)", fontsize=12)
    ax.set_title("Model Comparison — Cross-Validated MAE", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, cv_results["MAE (mean)"]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"  ✓ Saved {PLOT_DIR}/model_comparison.png")


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="black", s=60, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="red", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Y", fontsize=12)
    ax.set_ylabel("Predicted Y", fontsize=12)
    ax.set_title(f"{model_name}: Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(PLOT_DIR, f"actual_vs_pred_{fname}.png"), dpi=150)
    plt.close()
    print(f"  ✓ Saved {PLOT_DIR}/actual_vs_pred_{fname}.png")


def plot_residuals(y_true, y_pred, model_name):
    ensure_plot_dir()
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=15, color="#55A868", edgecolor="black", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Predicted − Actual)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"{model_name}: Residual Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(PLOT_DIR, f"residuals_{fname}.png"), dpi=150)
    plt.close()
    print(f"  ✓ Saved {PLOT_DIR}/residuals_{fname}.png")


def plot_feature_importance(model, feature_names, model_name):
    ensure_plot_dir()
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        range(len(importances)), importances[indices],
        color=sns.color_palette("mako", len(importances)), edgecolor="black",
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_ylabel("Importance", fontsize=12)
    ax.set_title(f"{model_name}: Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(PLOT_DIR, f"feature_importance_{fname}.png"), dpi=150)
    plt.close()
    print(f"  ✓ Saved {PLOT_DIR}/feature_importance_{fname}.png")


def plot_correlation_heatmap(X, y, feature_names):
    """Correlation heatmap of features + Y."""
    ensure_plot_dir()
    df = pd.DataFrame(X, columns=feature_names)
    df["Y"] = y
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, ax=ax, linewidths=0.5,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"  ✓ Saved {PLOT_DIR}/correlation_heatmap.png")


def plot_y_distribution(y):
    """Histogram of target variable Y."""
    ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=20, color="#C44E52", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Y Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Target Variable (Y)", fontsize=14, fontweight="bold")
    ax.axvline(y.mean(), color="navy", linestyle="--", linewidth=1.5, label=f"Mean = {y.mean():.1f}")
    ax.axvline(np.median(y), color="green", linestyle="--", linewidth=1.5, label=f"Median = {np.median(y):.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "y_distribution.png"), dpi=150)
    plt.close()
    print(f"  ✓ Saved {PLOT_DIR}/y_distribution.png")


# MAIN PIPELINE
def main():
    print("=" * 65)
    print("  CE 475 — Machine Learning Regression Project")
    print("=" * 65)

    # 1. Load data 
    print("\n[1/7] Loading data from Excel …")
    X_train_raw, y_train, X_test_raw = load_data_from_excel(DATA_FILE)
    print(f"  Training samples : {X_train_raw.shape[0]}")
    print(f"  Test samples     : {X_test_raw.shape[0]}")
    print(f"  Features         : {X_train_raw.shape[1]} (X1–X6)")
    print(f"  Y range          : [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"  Y mean / std     : {y_train.mean():.2f} / {y_train.std():.2f}")

    feature_names = [f"X{i}" for i in range(1, 7)]

    # 2. Preprocess
    print("\n[2/7] Preprocessing (StandardScaler + Polynomial Features) …")
    X_tr_sc, X_te_sc, X_tr_poly, X_te_poly, scaler, poly = preprocess(
        X_train_raw, X_test_raw
    )
    poly_feature_names = poly.get_feature_names_out(feature_names)
    print(f"  Scaled features  : {X_tr_sc.shape[1]}")
    print(f"  Poly features    : {X_tr_poly.shape[1]} (degree-2)")

    # 3. Exploratory plots
    print("\n[3/7] Generating exploratory plots …")
    plot_correlation_heatmap(X_train_raw, y_train, feature_names)
    plot_y_distribution(y_train)

    # 4. Cross-validation
    print("\n[4/7] Running 5-fold cross-validation …")

    models = get_models()
    
    models_linear = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
    }
    models_nonlinear = {
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "SVR (RBF Kernel)": SVR(kernel="rbf", C=100, epsilon=0.1),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, verbosity=0,
        ),
    }
    is_linear_dict = {
        "Ridge Regression": True, "Lasso Regression": True, "ElasticNet": True, "Huber Regression": True,
        "K-Nearest Neighbors": False, "SVR (RBF Kernel)": False, "Random Forest": False, "Gradient Boosting": False, "XGBoost": False
    }

    cv_linear = evaluate_models_cv({k: v for k, v in models.items() if is_linear_dict[k]}, X_tr_sc, y_train, is_linear_dict, X_tr_poly)
    cv_nonlinear = evaluate_models_cv({k: v for k, v in models.items() if not is_linear_dict[k]}, X_tr_sc, y_train, is_linear_dict, X_tr_poly)
    cv_results = pd.concat([cv_linear, cv_nonlinear], ignore_index=True)
    cv_results = cv_results.sort_values("MAE (mean)").reset_index(drop=True)

    print("\n  ┌──────────────────────────────────────────────────────────────────────────┐")
    print("  │                  5-Fold Cross-Validation Results                         │")
    print("  └──────────────────────────────────────────────────────────────────────────┘")
    print(cv_results.to_string(index=False))

    # 5. Visualise comparison
    print("\n[5/7] Generating model comparison plot …")
    plot_model_comparison(cv_results)

    # 6. Select best model, retrain on ALL data, predict test
    print("\n[6/7] Selecting best model & predicting test set …")
    best_name = cv_results.iloc[0]["Model"]
    best_mae = cv_results.iloc[0]["MAE (mean)"]
    best_r2 = cv_results.iloc[0]["R² (mean)"]

    print(f"\n  ★  Best Model: {best_name}")
    print(f"     CV MAE  = {best_mae:.2f}")
    print(f"     CV R²   = {best_r2:.4f}")

    is_linear = best_name in models_linear
    all_models = models

    best_model = sklearn.base.clone(all_models[best_name])

    X_train_final = X_tr_poly if is_linear_dict[best_name] else X_tr_sc
    X_test_final = X_te_poly if is_linear_dict[best_name] else X_te_sc


    best_model.fit(X_train_final, y_train)
    y_test_pred = best_model.predict(X_test_final)

    print(f"\n  Predicted Y values for test IDs 101–120:")
    for i, val in enumerate(y_test_pred):
        print(f"    ID {101 + i:3d} → {val:10.2f}")

    # Holdout diagnostics for the best model
    holdout_models = {best_name: sklearn.base.clone(all_models[best_name])}

    if is_linear:
        holdout_res = holdout_evaluate(holdout_models, X_tr_poly, y_train)
    else:
        holdout_res = holdout_evaluate(holdout_models, X_tr_sc, y_train)

    for name, (fitted_model, y_true, y_pred) in holdout_res.items():
        plot_actual_vs_predicted(y_true, y_pred, name)
        plot_residuals(y_true, y_pred, name)
        if not is_linear:
            plot_feature_importance(fitted_model, feature_names, name)
        else:
            if hasattr(fitted_model, "feature_importances_"):
                plot_feature_importance(
                    fitted_model,
                    list(poly_feature_names),
                    name,
                )

    # Also generate feature importance for top tree based models
    for name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        if name != best_name:
            m = sklearn.base.clone(all_models[name])
            m.fit(X_tr_sc, y_train)
            plot_feature_importance(m, feature_names, name)

    # 7. Export predictions
    print(f"\n[7/7] Exporting predictions to {OUTPUT_CSV} …")
    with open(OUTPUT_CSV, 'w') as f:
        for val in y_test_pred:
            f.write(f"{val:.6f}".replace('.', ',') + '\n')

    # Verify the CSV
    check = pd.read_csv(OUTPUT_CSV, header=None, decimal=',', sep='|').values.flatten()
    assert check.shape == (20,), f"CSV shape mismatch: {check.shape}"
    print(f"  ✓ {OUTPUT_CSV} written — {len(check)} rows, no header, no index, decimal=','")

    # Summary
    print("\n" + "=" * 65)
    print("  DONE — All outputs generated successfully")
    print("=" * 65)
    print(f"  • predictions.csv  : 20 predicted Y values")
    print(f"  • plots/           : {len(os.listdir(PLOT_DIR))} visualisation PNGs")
    print(f"  • Best model       : {best_name} (CV MAE = {best_mae:.2f})")
    print("=" * 65)

    return cv_results, best_name, best_model


if __name__ == "__main__":
    main()
