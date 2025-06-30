import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin

# Load data
df_train = pd.read_csv("house_prices_final/train.csv")
df_test = pd.read_csv("house_prices_final/test.csv")

# Store and drop IDs
test_ids = df_test["Id"]
df_train = df_train.drop(columns=["Id"])
df_test = df_test.drop(columns=["Id"])

# Combine train and test data for preprocessing
full_data = pd.concat([df_train.drop(columns=["SalePrice"]), df_test])


# >>> PREPROCESSING <<<

# Helper function to print info about specific null columns
def info_null_cols(df : pd.DataFrame, dtype : list[str]):
    null_cols = df.columns[df.isnull().any()]
    df[null_cols].select_dtypes(include=dtype).info()

# Helper function to print info about all null columns
def all_info_null_cols(df : pd.DataFrame):
    null_cols = df.columns[df.isnull().any()]
    df[null_cols].info()


# full_data.select_dtypes(include='int64').info()
# info_null_cols(full_data, 'int64')
# all_info_null_cols(full_data)
# print(full_data.info())


# >> object <<

# Fill "None" for some categorical columns
cols_str_fill_none = [
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature"
]
full_data[cols_str_fill_none] = full_data[cols_str_fill_none].fillna("None")


# Fill mode for some categorical columns
cols_fill_mode = [
    "MSZoning", "Electrical", 
    "Exterior1st", "Exterior2nd",
    "KitchenQual", "Functional", 
    "SaleType"
]
for col in cols_fill_mode:
    full_data[col] = full_data[col].fillna(full_data[col].mode()[0])

# Drop Utilities column (most(?) values are "AllPub"), and drop PoolQC column (99% values are "None")
full_data = full_data.drop(columns=["Utilities", "PoolQC"])


# >> float64 <<

# Fill 0 for some float columns
cols_fill_0 = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", 
    "GarageCars", "GarageArea", "GarageYrBlt"
]
full_data[cols_fill_0] = full_data[cols_fill_0].fillna(0)


# Fill median for LotFrontage, grouped by Neighborhood
full_data["LotFrontage"] = full_data["LotFrontage"].fillna(
    full_data.groupby("Neighborhood")["LotFrontage"].transform("median")
)

# >>> ADDITIONAL <<<

# extra value columns, that MIGHT be usefull:
# Total Square Feet. Bigger homes = higher value. 
full_data["TotalSF"] = full_data["TotalBsmtSF"] + full_data["1stFlrSF"] + full_data["2ndFlrSF"]

# More bathrooms usually higher value.
full_data["TotalBath"] = (
    full_data["FullBath"] + (0.5 * full_data["HalfBath"]) +
    full_data["BsmtFullBath"] + (0.5 * full_data["BsmtHalfBath"])
)
# older houses usually lower value.
full_data["HouseAge"] = full_data["YrSold"] - full_data["YearBuilt"]

# How long ago it was remodelled.
full_data["RemodelAge"] = full_data["YrSold"] - full_data["YearRemodAdd"]


# full_data.select_dtypes(include='float64').info()
# info_null_cols(full_data, 'object')
# all_info_null_cols(full_data)
# print(full_data.info())


# >> encoding <<

# After all NaNs handled, encode categorical features
full_data_encoded = pd.get_dummies(full_data, drop_first=True)

# Split back to train/test sets
X = full_data_encoded.iloc[:len(df_train)]
X_pred = full_data_encoded.iloc[len(df_train):]
y = df_train["SalePrice"]
# y = np.log1p(y)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scaler AFTER splitting, to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# >>> PLOTTING <<<
def plot_tree(dtr : DecisionTreeRegressor):
    dtr.fit(X_train, y_train)
    plt.figure(figsize=(20, 10))
    tree.plot_tree(dtr, filled=True, feature_names=X_train.columns, fontsize=10)
    plt.title("Decision Tree for Student Performance Prediction")
    plt.show()


# >>> CROSS-VALIDATION <<<
def cross_val_info(model : RegressorMixin):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rmse_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    cv_r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="r2")
    
    rmse_scores = -cv_rmse_scores  # or np.abs(cv_rmse_scores)

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


# >>> FINAL-EVAL <<<
def final_eval(model : RegressorMixin):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    final_rmse = np.sqrt(mse)
    
    final_r2 = r2_score(y_test, y_pred)

    print(f"Final Test RMSE: {final_rmse:.4f}")
    print(f"Final Test R²: {final_r2:.4f}")


#  >>> KNR Model init <<<
# knr = KNeighborsRegressor(n_neighbors=5)

#  >>> SVR Model init <<<
svr = SVR(kernel="linear", C=140)

# >>> DT Model init <<<
# dtr = DecisionTreeRegressor(random_state=42, max_depth=10)
# plot_tree(dtr)


# >>> Execution <<<
cross_val_info(svr)
# final_eval(svr)