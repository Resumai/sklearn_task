import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_squared_log_error
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# Add parent folder to sys.path so we can import regressor_helper
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regressor_helper import apply_all, encode_df, cross_val_info, kaggle_eval, cross_val_info_real_scale, kaggle_eval_real_scale

# Helper function to print info about specific null columns
def info_null_cols(df : pd.DataFrame, dtype : list[str]):
    null_cols = df.columns[df.isnull().any()]
    df[null_cols].select_dtypes(include=dtype).info()

# Helper function to print info about all null columns
def all_info_null_cols(df : pd.DataFrame):
    null_cols = df.columns[df.isnull().any()]
    df[null_cols].info()


def check_importance(model, df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model.fit(X, y)
    importances = model.feature_importances_
    for name, score in zip(X.columns, importances):
        print(f"{name}: {score:.4f}")


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

    # df["Spaciousness"] = df["TotalSF"] * df["TotalBath"] * df["GarageCars"] * df["TotalPorchSF"] * df["BsmtExposure"]

    df["Qual_SF"] = df["OverallQual"] * df["TotalSF"]
    df["Bath_Garage"] = df["TotalBath"] * df["GarageCars"]
    df["Age_Qual"] = df["HouseAge"] * df["OverallQual"]


    # Is it a new house?
    # df["IsNew"] = (df["YearBuilt"] == df["YrSold"]).astype(int)


    return df



def drop_low_importance(df: pd.DataFrame, model, target_col: str, num_to_drop: int = 100) -> pd.DataFrame:
    """
    Trains the model on given DataFrame and drops `num_to_drop` least important features.
    Designed to be compatible with apply_all() pipeline.
    """
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col] if target_col in df.columns else None

    # Fit model only if target_col is present (skip for test set)
    if y is not None:
        model.fit(X, y)
        importances = model.feature_importances_
        importance_dict = dict(zip(X.columns, importances))
        lowest = sorted(importance_dict.items(), key=lambda x: x[1])[:num_to_drop]
        to_drop = [name for name, _ in lowest if name in df.columns]
        df = df.drop(columns=to_drop)
    return df




#  >>> Model init <<<
# TODO: GridSearchCV

# model = SVR(kernel="linear", C=10)

# from xgboost import XGBRegressor
# model = XGBRegressor(
#     n_estimators=400,
#     learning_rate=0.04,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )


# from sklearn.ensemble import HistGradientBoostingRegressor
# model = HistGradientBoostingRegressor(
#     learning_rate=0.11,
#     max_depth=4,
#     random_state=42,
#     max_iter=350,
# )

# from lightgbm import LGBMRegressor
# model = LGBMRegressor(
#     n_estimators=800,
#     learning_rate=0.04,
#     max_depth=6,               # try 6–10
#     min_child_samples=10,      # lower → allows more splits
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )


from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.04,
    depth=5,
    loss_function='RMSE',
    verbose=100,
    random_seed=42
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
df_train = drop_low_importance(df_train, model, target_col="SalePrice", num_to_drop=175)
df_test = apply_all(df_test, [fill_object_cols, fill_float64_cols, additional_cols, encode_df])
# df_test = drop_low_importance(df_test, model, target_col="SalePrice", num_to_drop=175)


# df_train.info()
# check_importance(model, df_train, target_col="SalePrice")

# cross_val_info(model, df_train, target_col="SalePrice")
# for k in range(160, 181, 1):
#     df_tmp = drop_low_importance(df_train.copy(), model, target_col="SalePrice", num_to_drop=k)
#     print(f"\n--- num_to_drop = {k} ---")
#     cross_val_info_real_scale(model, df_tmp, target_col="SalePrice", splits=4)


cross_val_info_real_scale(model, df_train, target_col="SalePrice", splits=4, use_scaler=True)

# kaggle_eval(model, df_train, df_test, test_ids=test_ids)
# kaggle_eval_real_scale(model, df_train, df_test, test_ids=test_ids)
