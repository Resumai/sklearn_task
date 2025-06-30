# Add parent folder to sys.path so we can import regressor_helper
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regressor_helper import apply_all, encode_df, cross_val_info

import pandas as pd
import numpy as np

def check_importance(model, df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model.fit(X, y)
    importances = model.feature_importances_
    for name, score in zip(X.columns, importances):
        print(f"{name}: {score:.4f}")

# df["StudentID"] = np.random.permutation(df["StudentID"])

# >>> Model init <<<

# TODO: GridSearchCV

# ** RandomForestRegressor **
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=None,
#     min_samples_leaf=3,
#     random_state=42,
#     n_jobs=-1  # Uses all cores
# )


# from xgboost import XGBRegressor
# model = XGBRegressor(
#     n_estimators=350,
#     learning_rate=0.05,
#     max_depth=3,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )


# ** XGBoostRegressor **
from xgboost import XGBRegressor
base_model = XGBRegressor(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(
    estimator=base_model,
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)



# >>> Execution <<<
df = pd.read_csv("student_performance/Student_performance_data.csv")
print(df.info())
df = df.drop(columns=["StudentID"])
df = encode_df(df)


# import matplotlib.pyplot as plt
# plt.scatter(df["StudentID"], df["GPA"])
# plt.show()

# check_importance(model, df, target_col="GPA")
cross_val_info(model, df, target_col="GPA", splits=5, use_scaler=True)

