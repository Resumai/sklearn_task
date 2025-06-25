import pandas as pd
import numpy as np

df = pd.read_csv("house_prices/house_prices_regression.csv")


df = df.drop(columns="Id")



null_cols = df.columns[df.isnull().any()]
df[null_cols].info()
# print(df.info())



df["Alley"] = df["Alley"].fillna("None")
# Masonry veneer type
df["MasVnrType"] = df["MasVnrType"].fillna("Unknown")
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
basement_cols = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
df[basement_cols] = df[basement_cols].fillna("None")

# print(df[df["Fireplaces"] == 0]["FireplaceQu"].isna().sum())
# print(df["FireplaceQu"].isna().sum())
df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

# print(df[df["PoolArea"] == 0]["PoolQC"].isna().sum())
# print(df["PoolQC"].isna().sum())
df["PoolQC"] = df["PoolQC"].fillna("None")

# print(df[df["GarageArea"] == 0]["GarageQual"].isna().sum())
# print(df["GarageQual"].isna().sum())
# garage_str_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
# garage_num_cols = ["GarageYrBlt", "GarageArea"]
# print(df.info())

# df[garage_str_cols] = df[garage_str_cols].fillna("None")
# df[garage_num_cols] = df[garage_num_cols].fillna(0)
# df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

# print(df["Electrical"].unique())
# print(df["Electrical"].value_counts(dropna=False))
df["Electrical"] = df["Electrical"].fillna("SBrkr")

df["Fence"] = df["Fence"].fillna("None")
df["MiscFeature"] = df["MiscFeature"].fillna("None")


# pd.set_option('display.max_columns', None)
# print(df.info())
# print(df.describe())


# 1.88 result. If > 1 then strongly skewed. Between 0.5-1 then moderately skewed.
# print(df["SalePrice"].skew()) 

# Log transformation - supposedly 
df["SalePrice"] = np.log1p(df["SalePrice"])



# import matplotlib.pyplot as plt
# df["SalePrice"].hist(bins=50)
# plt.title("SalePrice Distribution")
# plt.show()


# For checking sum of duplicates
# print(df.duplicated().sum())

# Encoding
# categorical_cols = df.select_dtypes(include=["object"]).columns
# df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# encoded_feature_names = df_encoded.columns.tolist()

# print(encoded_feature_names)
# print(df_encoded.shape)
# print(df_encoded.head())


import seaborn as sns
import matplotlib.pyplot as plt

# sns.boxplot(x=df["LotArea"])
# plt.title("LotArea Outliers (Boxplot)")
# plt.show()
# print("Skew after modification:", df["LotArea"].skew())

df["LotArea"] = np.log1p(df["LotArea"])
# q_high = df["LotArea"].quantile(0.99)
# df["LotArea"] = np.where(df["LotArea"] > q_high, q_high, df["LotArea"])


# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# df["LotArea"].hist(bins=50)
# plt.title("Distribution of LotArea")
# plt.xlabel("LotArea")
# plt.ylabel("Number of Houses")
# plt.grid(True)
# plt.show()

# leaving for example, not usefull tho
# plt.figure(figsize=(10, 5))
# df["LotArea"].hist(bins=50)
# plt.title("Zoomed-In LotArea Distribution (up to 50,000)")
# plt.xlabel("LotArea")
# plt.ylabel("Count")
# plt.xlim(0, 50000)  # Clip long tail to focus on bulk data
# plt.grid(True)
# plt.show()


# For checking outliers
# q99 = df["LotArea"].quantile(0.99)
# outliers = df[df["LotArea"] > q99]
# print(f"Outliers: {len(outliers)} rows ({len(outliers) / len(df) * 100:.2f}%)")


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

print(df.info())

# print(df["TotalSF"])