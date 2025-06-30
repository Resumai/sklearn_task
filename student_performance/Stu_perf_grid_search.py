import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load and encode dataset
df = pd.read_csv("student_performance/Student_performance_data.csv")

df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop("GPA", axis=1)
y = df_encoded["GPA"]

# Define RMSE scorer (version-independent)
def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

# Define model and grid
# model = RandomForestRegressor(random_state=42)
# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_leaf': [1, 3, 5]
# }

from xgboost import XGBRegressor
model = XGBRegressor(random_state=42)
param_grid = {
    'n_estimators': [250, 300, 350, 400, 450],
    'learning_rate': [0.03, 0.05, 0.06, 0.07, 0.08],
    'max_depth': [2, 3, 4],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# model = XGBRegressor(
#     n_estimators=250,
#     learning_rate=0.05,
#     max_depth=3,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )



# Setup GridSearch with KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold,
    scoring=rmse_scorer,
    n_jobs=-1,
    verbose=2
)

# Run search
grid_search.fit(X, y)

# Output results
print("Best parameters:", grid_search.best_params_)
print(f"Best RMSE score (CV): {-grid_search.best_score_:.4f}")
