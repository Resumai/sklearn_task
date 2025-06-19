import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data and check it
df = pd.read_csv("vg_sales/vgsales.csv")
# print(df.head(20))
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# 2. Drop missing values (Year and Publisher have NaN)
df.dropna(inplace=True)

# 3. Encode categorical columns
categorical_cols = ['Platform', 'Genre', 'Publisher']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 4. Define target and features
y = df['Global_Sales']
X = df.drop(columns=['Global_Sales', 'Name'])  # 'Name' likely not useful

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")
