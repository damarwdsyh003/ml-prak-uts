import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the prostate cancer dataset
file_path = 'prostate_cancerone.csv'  # Ganti dengan path dataset Anda
df = pd.read_csv(file_path)

# Checking for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Select only numeric columns for detecting outliers, excluding 'id'
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = numeric_columns.drop('id')  # Exclude 'id' column

# Detecting outliers using IQR method only on numeric columns
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Detecting outliers
outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()
print("\nOutliers detected in each numeric column:\n", outliers)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMax scaling only to numeric columns, excluding 'id'
df_scaled = df.copy()
df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Show the transformed data
print("\nTransformed Data (Min-Max Scaled):\n", df_scaled.head())

# Save the transformed data to a new CSV file (optional)
df_scaled.to_csv('Prostate_Cancer_Scaled.csv', index=False)
