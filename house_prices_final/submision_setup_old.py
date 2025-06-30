import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_squared_log_error
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin


# Add parent folder to sys.path so we can import regressor_helper
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regressor_helper import apply_all, encode_df, cross_val_info

# Helper function to print info about specific null columns
def info_null_cols(df : pd.DataFrame, dtype : list[str]):
    null_cols = df.columns[df.isnull().any()]
    df[null_cols].select_dtypes(include=dtype).info()

# Helper function to print info about all null columns
def all_info_null_cols(df : pd.DataFrame):
    null_cols = df.columns[df.isnull().any()]
    df[null_cols].info()


# Helper function to apply multiple functions to a DataFrame
def apply_all(df, funcs):
    for func in funcs:
        df = func(df)
    return df

# Helper function, RMSLE Scorer
def rmsle_scorer():
    def rmsle(y_actual, y_predicted):
        y_predicted = np.maximum(0, y_predicted)
        return np.sqrt(mean_squared_log_error(y_actual, y_predicted))
    return make_scorer(rmsle, greater_is_better=False)


# >>> DATA INFO <<<
# full_data.select_dtypes(include='int64').info()
# info_null_cols(full_data, 'int64')
# all_info_null_cols(full_data)
# print(full_data.info())


# >>> PREPROCESSING <<<

def fill_object_cols(df : pd.DataFrame):
    # Fill "None" for some categorical columns
    cols_str_fill_none = [
        "Alley", "MasVnrType", "BsmtQual", "BsmtCond",
        "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
        "GarageCond", "PoolQC", "Fence", "MiscFeature"
    ]
    df[cols_str_fill_none] = df[cols_str_fill_none].fillna("None")


    # Fill mode for some categorical columns
    cols_fill_mode = [
        "MSZoning", "Electrical", 
        "Exterior1st", "Exterior2nd",
        "KitchenQual", "Functional", 
        "SaleType"
    ]
    for col in cols_fill_mode:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop Utilities column (most(?) values are "AllPub"), and drop PoolQC column (99% values are "None")
    df = df.drop(columns=["Utilities", "PoolQC", "PoolArea", "MiscFeature", "Alley", "SaleType"])

    return df



def fill_float64_cols(df : pd.DataFrame):

    # Fill 0 for some float columns
    cols_fill_0 = [
        "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
        "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", 
        "GarageCars", "GarageArea", "GarageYrBlt"
    ]
    df[cols_fill_0] = df[cols_fill_0].fillna(0)


    # Fill median for LotFrontage, grouped by Neighborhood
    df["LotFrontage"] = df["LotFrontage"].fillna(
        df.groupby("Neighborhood")["LotFrontage"].transform("median")
    )

    return df

def additional_cols(df : pd.DataFrame):

    # extra value columns, that MIGHT be usefull:
    # Total Square Feet. Bigger homes = higher value. 
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # More bathrooms usually higher value.
    df["TotalBath"] = (
        df["FullBath"] + (0.5 * df["HalfBath"]) +
        df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"])
    )

    # older houses usually lower value.
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]


    # How long ago it was remodelled.
    df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]

    # Total porch square feet
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] +
        df["3SsnPorch"] + df["ScreenPorch"]
    )

    # # Is it a new house?
    # df["IsNew"] = (df["YearBuilt"] == df["YrSold"]).astype(int)


    return df


# Encode categorical features, after all NaNs handled
def encode_df(df : pd.DataFrame):
    df = pd.get_dummies(df, drop_first=True)
    return df


# >>> CROSS-VALIDATION <<<
def cross_val_info_old(model : RegressorMixin, df_train : pd.DataFrame):

    # Separate features and target
    X = df_train.drop(columns=["SalePrice"])
    y = df_train["SalePrice"]
    # y = np.log1p(y)

    # Scaler AFTER splitting, to avoid data leakage
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rmse_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="neg_root_mean_squared_error")
    cv_r2_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
    cv_rmsle_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring=rmsle_scorer())

    rmse_scores = -cv_rmse_scores  # or np.abs(cv_rmse_scores)
    rmsle_scores = -cv_rmsle_scores


    print("KFold cross validation:")
    # Print individual RMSE folds
    for i, score in enumerate(rmse_scores, 1):
        print(f"Fold {i}: RMSE = {score:.4f}")

    # Print average RMSE
    print(f"\nAverage RMSE: {rmse_scores.mean():.4f}")
    print("---" * 10)


    # Print individual R² folds
    for i, score in enumerate(cv_r2_scores, 1):
        print(f"Fold {i}: R² = {score:.4f}")
    
    # Print average R²
    print(f"\nAverage R²: {cv_r2_scores.mean():.4f}")
    print("---" * 10)


    # Print individual RMSLE folds
    for i, score in enumerate(rmsle_scores, 1):
        print(f"Fold {i}: RMSLE = {score:.4f}")

    # # Print average RMSLE
    print(f"\nAverage RMSLE: {rmsle_scores.mean():.4f}")
    print("---" * 10)


    # avg_rmse_log = rmse_scores.mean()
    # avg_rmse_original = np.expm1(avg_rmse_log)

    # print(f"\nAverage RMSE (log scale): {avg_rmse_log:.4f}")
    # print(f"Average RMSE (original scale): {avg_rmse_original:.2f}")
    # print("---" * 10)


# # >>> FINAL-EVAL <<<
# def final_eval(model : RegressorMixin):
#     Train-test split
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
    
#     model.fit(X_train_scaled, y_train)
#     prediction = model.predict(X_val_scaled)

#     mse = mean_squared_error(y_val, prediction)
#     final_rmse = np.sqrt(mse)
    
#     final_r2 = r2_score(y_val, prediction)

#     print(f"Final Validation RMSE: {final_rmse:.4f}")
#     print(f"Final Validation R²: {final_r2:.4f}")


def kaggle_eval(model: RegressorMixin, df_train: pd.DataFrame, df_test: pd.DataFrame):

    # Separate features and target
    X = df_train.drop(columns=["SalePrice"])
    y = df_train["SalePrice"]
    # y = np.log1p(y)


    # Scale full training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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




#  >>> Model init <<<
# TODO: GridSearchCV

# model = SVR(kernel="linear", C=100)

# from xgboost import XGBRegressor
# model = XGBRegressor(
#     n_estimators=400,
#     learning_rate=0.04,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )



from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# >>> Execution <<<
# Load data
df_train = pd.read_csv("house_prices_final/train.csv")
df_test = pd.read_csv("house_prices_final/test.csv")

# Store and drop IDs
test_ids = df_test["Id"]
df_train = df_train.drop(columns=["Id"])
df_test = df_test.drop(columns=["Id"])

df_train = apply_all(df_train, [fill_object_cols, fill_float64_cols, additional_cols, encode_df])
df_test = apply_all(df_test, [fill_object_cols, fill_float64_cols, additional_cols, encode_df])


cross_val_info(model, df_train)
# kaggle_eval(model, df_train, df_test)

# cross_val_info(svr, df_train)
# kaggle_eval(svr, df_train, df_test)