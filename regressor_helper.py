
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.base import RegressorMixin


# Helper function, RMSLE Scorer
def rmsle_scorer():
    def rmsle(y_actual, y_predicted):
        y_predicted = np.maximum(0, y_predicted)
        return np.sqrt(mean_squared_log_error(y_actual, y_predicted))
    return make_scorer(rmsle, greater_is_better=False)


# Helper function to apply multiple functions to a DataFrame
def apply_all(df, funcs):
    for func in funcs:
        df = func(df)
    return df

# Encode categorical features, after all NaNs handled
def encode_df(df : pd.DataFrame):
    df = pd.get_dummies(df, drop_first=True)
    return df


# >>> CROSS-VALIDATION <<<
def cross_val_info(model : RegressorMixin, df_train : pd.DataFrame, target_col : str, splits : int = 5, use_scaler=True):

    # Separate features and target
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]
    # y = np.log1p(y)


    # Scaler AFTER splitting, to avoid data leakage
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # KFold
    kf = KFold(n_splits=splits, shuffle=True, random_state=42)

    # Scoring
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
        "rmsle": rmsle_scorer()
    }

    results = cross_validate(model, X_scaled, y, cv=kf, scoring=scoring, return_train_score=False)

    # Convert to positive for RMSE/RMSLE
    rmse_scores = -results["test_rmse"]
    rmsle_scores = -results["test_rmsle"]
    r2_scores = results["test_r2"]


    print("KFold cross validation:")

    # Print individual RMSE folds
    for i in range(splits):
        print(f"Fold {i+1}: RMSE = {rmse_scores[i]:.4f}")

    # Print average RMSE
    print(f"\nAverage RMSE: {rmse_scores.mean():.4f}")
    print("---" * 10)

    # Print individual R² folds
    for i in range(splits):
        print(f"Fold {i+1}: R² = {r2_scores[i]:.4f}")

    # Print average R²
    print(f"\nAverage R²: {r2_scores.mean():.4f}")
    print("---" * 10)

    # Print individual RMSLE folds
    for i in range(splits):
        print(f"Fold {i+1}: RMSLE = {rmsle_scores[i]:.4f}")

    # Print average RMSLE
    print(f"\nAverage RMSLE: {rmsle_scores.mean():.4f}")
    print("---" * 10)



    # avg_rmse_log = rmse_scores.mean()
    # avg_rmse_original = np.expm1(avg_rmse_log)

    # print(f"\nAverage RMSE (log scale): {avg_rmse_log:.4f}")
    # print(f"Average RMSE (original scale): {avg_rmse_original:.2f}")
    # print("---" * 10)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
import numpy as np

def cross_val_info_real_scale(model: RegressorMixin, df_train: pd.DataFrame, target_col: str, splits: int = 5, use_scaler=True):
    X = df_train.drop(columns=[target_col])
    y = np.log1p(df_train[target_col])  # log1p here

    kf = KFold(n_splits=splits, shuffle=True, random_state=42)

    rmse_list = []
    r2_list = []
    rmsle_list = []

    for i, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Convert back to real values
        y_val_real = np.expm1(y_val)
        y_pred_real = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
        r2 = r2_score(y_val_real, y_pred_real)
        rmsle = np.sqrt(mean_squared_log_error(y_val_real, y_pred_real))

        rmse_list.append(rmse)
        r2_list.append(r2)
        rmsle_list.append(rmsle)

        # print(f"Fold {i}: RMSE = {rmse:.2f}, R² = {r2:.4f}, RMSLE = {rmsle:.4f}")

    print("\nAverage RMSE:", round(np.mean(rmse_list), 2))
    print("Average R²:", round(np.mean(r2_list), 4))
    print("Average RMSLE:", round(np.mean(rmsle_list), 4))



def kaggle_eval(model: RegressorMixin, df_train: pd.DataFrame, df_test: pd.DataFrame, use_scaler=True, test_ids: pd.DataFrame = None):

    # Separate features and target
    X = df_train.drop(columns=["SalePrice"])
    y = df_train["SalePrice"]
    # y = np.log1p(y)


    # Scale full training data
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Align test to train features?
    df_test = df_test.reindex(columns=X.columns, fill_value=0)

    # Scale test set
    df_test_scaled = scaler.transform(df_test)

    # Train and predict
    model.fit(X_scaled, y)
    prediction = model.predict(df_test_scaled)
    # prediction = np.expm1(prediction)

    # Create submission
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": prediction
    })
    submission.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' created successfully.")


def kaggle_eval_real_scale(model: RegressorMixin, df_train: pd.DataFrame, df_test: pd.DataFrame, use_scaler=True, test_ids: pd.DataFrame = None):

    # Separate features and target
    X = df_train.drop(columns=["SalePrice"])
    y = df_train["SalePrice"]
    y_log = np.log1p(y)  # Log-transform the target

    # Scale full training data
    if use_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X

    # Align test to train features
    df_test_aligned = df_test.reindex(columns=X.columns, fill_value=0)

    # Scale test set
    if use_scaler:
        df_test_scaled = scaler.transform(df_test_aligned)
    else:
        df_test_scaled = df_test_aligned

    # Train the model and predict
    model.fit(X_scaled, y_log)
    prediction_log = model.predict(df_test_scaled)
    prediction = np.expm1(prediction_log)  # Revert log1p

    # Create submission
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": prediction
    })
    submission.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' created successfully.")
