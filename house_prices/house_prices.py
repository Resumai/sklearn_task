# Re-import required libraries and reload the dataset after code execution state reset
import pandas as pd
import numpy as np

# Reload the dataset
file_path = "house_prices/house_prices_regression.csv"
df = pd.read_csv(file_path)
print(df.info())


# Separate numerical and categorical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features = [col for col in numerical_features if col not in ["Id", "SalePrice"]]
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

# Regenerate the summary for continuous features with updated structure
continuous_summary = []
for col in numerical_features:
    series = df[col]
    summary = {
        "Feature": col,
        "Count": series.count(),
        "% Miss": series.isna().mean() * 100,
        "Cardinality": series.nunique(),
        "Min": series.min(),
        "Q1": series.quantile(0.25),
        "Mean": series.mean(),
        "Median": series.median(),
        "Q3": series.quantile(0.75),
        "Max": series.max(),
        "Std.Dev.": series.std(),
    }
    continuous_summary.append(summary)

continuous_df = pd.DataFrame(continuous_summary)

# Regenerate the summary for categorical features with updated structure
categorical_summary = []
for col in categorical_features:
    series = df[col]
    value_counts = series.value_counts(dropna=True)
    total_count = series.count()
    summary = {
        "Feature": col,
        "Count": total_count,
        "% Miss": series.isna().mean() * 100,
        "Cardinality": series.nunique(),
        "Mode": value_counts.index[0] if len(value_counts) > 0 else None,
        "Mode Freq": value_counts.iloc[0] if len(value_counts) > 0 else None,
        "Mode %": (value_counts.iloc[0] / total_count * 100) if total_count > 0 else None,
        "2nd Mode": value_counts.index[1] if len(value_counts) > 1 else None,
        "2nd Mode Freq": value_counts.iloc[1] if len(value_counts) > 1 else None,
        "2nd Mode %": (value_counts.iloc[1] / total_count * 100) if len(value_counts) > 1 and total_count > 0 else None,
    }
    categorical_summary.append(summary)

categorical_df = pd.DataFrame(categorical_summary)

print("Continuous Feature Stats:")
print(continuous_df.to_string(index=False))

print("\nCategorical Feature Stats:")
print(categorical_df.to_string(index=False))
