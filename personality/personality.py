import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load data
df = pd.read_csv("personality/personality_datasert.csv")
# print(df.head(20))
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# 2. Preprocessing
# Encode specific columns only
le_stage = LabelEncoder()
df["Stage_fear"] = le_stage.fit_transform(df["Stage_fear"])

le_drained = LabelEncoder()
df["Drained_after_socializing"] = le_drained.fit_transform(df["Drained_after_socializing"])

# OR 
# label_encoders = {}

# for col in df.columns:
#     if df[col].dtype == 'object' and col != 'Personality':
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["Personality"])
X = df.drop(columns="Personality")

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Model training
model = RandomForestClassifier(min_samples_leaf=20, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("CV Accuracy Scores:", scores)
print("Average Accuracy:", scores.mean())
