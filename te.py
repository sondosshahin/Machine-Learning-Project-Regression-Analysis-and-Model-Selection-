import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Define the missing values
missing_values = [" ", "NA", "N/A", "N A", "NaN"]

# Read the data and replace the missing values with NaN
data = pd.read_csv("cars.csv", na_values=missing_values)
df_upd = data.copy()

# Print the dataframe before any changes
print("Before filling missing values:")
print(df_upd.head())


categorical_features = ['car name', 'brand', 'country']
for column in categorical_features:
    # Calculate frequency of each category
    freq_encoding = df_upd[column].value_counts() / len(df_upd)
    # Replace each value in the column with its frequency
    df_upd[column] = df_upd[column].map(freq_encoding)
    print(f"Frequency Encoding for '{column}':\n{freq_encoding}\n")  # Display frequency map


# Apply price adjustment function (ensure 'apply_price_adj' is defined)
df_upd['price'] = df_upd['price'].apply(apply_price_adj)

# Function to convert non-numeric values to NaN
def to_numeric(value):
    try:
        return pd.to_numeric(value, errors='coerce')  # Coerce non-numeric values to NaN
    except Exception as e:
        return np.nan

# Apply the conversion to numeric for the relevant columns
df_upd['cylinder'] = df_upd['cylinder'].apply(to_numeric)
df_upd['horse_power'] = df_upd['horse_power'].apply(to_numeric)
df_upd['top_speed'] = df_upd['top_speed'].apply(to_numeric)
df_upd['seats'] = df_upd['seats'].astype(str).str.extract(r'(\d+)')[0].apply(pd.to_numeric, errors='coerce')


print("\n\n")

columns_to_fill = ['car name', 'brand', 'country','price', 'cylinder', 'horse_power', 'top_speed', 'seats']  

for column in columns_to_fill:
    # Calculate the median, ignoring NaN values
    median_value = df_upd[column].median()
    print(f"Median value for {column}: {median_value}")
    df_upd[column] = df_upd[column].fillna(median_value)

# Print the dataframe after filling missing values
print("\nAfter filling missing values:")
print(df_upd.head())



################################
print("\n\n");

columns_to_standardize = ['price', 'horse_power','engine_capacity']
columns_to_normalize = ['top_speed']

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

data_standardized = df_upd.copy()
data_standardized[columns_to_standardize] = scaler_standard.fit_transform(df_upd[columns_to_standardize])
df_upd[columns_to_standardize] = scaler_standard.fit_transform(df_upd[columns_to_standardize])
data_normalized = df_upd.copy()
data_normalized[columns_to_normalize] = scaler_minmax.fit_transform(df_upd[columns_to_normalize])
df_upd[columns_to_normalize] = scaler_minmax.fit_transform(df_upd[columns_to_normalize])
for column in columns_to_standardize + columns_to_normalize:
    if column in columns_to_standardize:
        print(f'{column} range before standardizing: {df_upd[column].min()} - {df_upd[column].max()}')
        print(f'{column} range after standardizing: {data_standardized[column].min()} - {data_standardized[column].max()}')
    elif column in columns_to_normalize:
        print(f'{column} range before normalizing: {df_upd[column].min()} - {df_upd[column].max()}')
        print(f'{column} range after normalizing: {data_normalized[column].min()} - {data_normalized[column].max()}')
    print("\n\n")