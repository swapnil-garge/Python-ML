import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import warnings
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Data Simulation & Preparation ---
# In a real scenario, you would load your data like this:
# df = pd.read_csv('your_plant_data.csv')

# For demonstration, let's create a more comprehensive and realistic sample dataset.
np.random.seed(42)
dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=365, freq='D'))
num_records = len(dates)

# Target Variables
data = {
    'GMT Prod Yield Date': dates,
    'Merge': np.random.choice(['ProductA', 'ProductB', 'ProductC', 'Shutdown-Planned'], size=num_records),
    'Downtime (Hrs)': np.random.rand(num_records) * 8,
    'Uptime': np.random.rand(num_records) * 16,
    'FPY %': 95 - np.random.rand(num_records) * 10,
    'Wound %': 98 - np.random.rand(num_records) * 5,
    'OEE Availability': 90 - np.random.rand(num_records) * 15
}

# Process, Raw Material, and Pre-Step Variables
# Create 45 dummy variables (X1 to X44, plus one extra)
for i in range(1, 46):
    data[f'X{i}'] = np.random.rand(num_records) * 100

df = pd.DataFrame(data)

# Make specific variables categorical as described
df['X14'] = np.random.choice(['Type1', 'Type2', 'Type3', np.nan], size=num_records, p=[0.4, 0.4, 0.15, 0.05])
df['X17'] = np.random.choice(['SupplierA', 'SupplierB', 'SupplierC', np.nan], size=num_records, p=[0.5, 0.3, 0.15, 0.05])

# Introduce some artificial shifts for demonstration
df.loc[(df['Merge'] == 'ProductA') & (df['GMT Prod Yield Date'] > '2023-06-01'), 'FPY %'] -= 10
df.loc[(df['Merge'] == 'ProductA') & (df['GMT Prod Yield Date'] > '2023-06-01'), 'X11'] += 20 # Correlated change
df.loc[(df['Merge'] == 'ProductB') & (df['GMT Prod Yield Date'] > '2023-09-01'), 'Downtime (Hrs)'] += 4
df.loc[(df['Merge'] == 'ProductB') & (df['GMT Prod Yield Date'] > '2023-09-01'), 'X5'] -= 15 # Correlated change

# Introduce some missing values
for col in df.columns:
    if df[col].dtype != 'datetime64[ns]' and col not in ['Merge', 'X14', 'X17']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

# Convert date column to datetime objects and set as index
if not pd.api.types.is_datetime64_any_dtype(df['GMT Prod Yield Date']):
    df['GMT Prod Yield Date'] = pd.to_datetime(df['GMT Prod Yield Date'])
df = df.sort_values('GMT Prod Yield Date').set_index('GMT Prod Yield Date')

print("--- Initial Data Head ---")
print(df.head())

# --- 2. Data Preprocessing (Imputation & Encoding) ---
print("\n--- Starting Data Preprocessing ---")
# Separate numerical and categorical columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = ['X14', 'X17']

# Impute categorical variables with the mode
for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)
    # Use Label Encoding for simplicity in this example
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
print("Categorical variables imputed and encoded.")

# Impute numerical variables using k-NN
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print("Numerical variables imputed using k-NN.")
print("\n--- Preprocessed Data Head ---")
print(df.head())


# --- 3. Feature Engineering (Time Lags) ---
print("\n--- Creating Lagged Features ---")
# Define variable groups
raw_material_vars = [f'X{i}' for i in [11, 12, 13, 14, 15, 17, 24] + list(range(38, 45))]
pre_step_vars = [f'X{i}' for i in [5, 6, 7, 8, 18, 19, 25, 26, 28, 29, 34, 35, 36]]

# Create a copy for lagged features to avoid issues with original df
df_lagged = df.copy()

# Raw material lags (2 to 7 days)
for col in raw_material_vars:
    for i in range(2, 8):
        df_lagged[f'{col}_lag{i}'] = df_lagged[col].shift(i)

# Pre-step lag (1 day)
for col in pre_step_vars:
    df_lagged[f'{col}_lag1'] = df_lagged[col].shift(1)

# Drop rows with NaNs created by the shifting process
df_lagged.dropna(inplace=True)
print(f"Lagged features created. New shape of dataframe: {df_lagged.shape}")


# --- 4. Change Point Detection Function (Modified) ---
def find_change_points(df, product_type, target_variable):
    """
    Finds change points using PELT and returns the indices.
    """
    product_df = df[df['Merge'] == product_type][[target_variable]].dropna()
    if len(product_df) < 20:
        return [], None
    points = product_df[target_variable].values
    n = len(points)
    sigma = np.std(points)
    penalty = 2 * np.log(n) * (sigma**2)
    algo = rpt.Pelt(model="l2").fit(points)
    result = algo.predict(pen=penalty)
    if len(result) > 0 and result[-1] == len(points):
        result = result[:-1]
    return result, product_df


# --- 5. Root Cause Analysis Function ---
def analyze_change_points(change_points, product_df, full_df, product_type, target_variable):
    """
    Analyzes the data before and after change points and runs a feature importance model.
    """
    if not change_points:
        print(f"\nNo change points to analyze for {product_type} - {target_variable}.")
        return

    print(f"\n--- Root Cause Analysis for [Product: {product_type}] | [Target: {target_variable}] ---")

    # Define feature columns (all except targets and merge col)
    feature_cols = [col for col in full_df.columns if col not in ['Downtime (Hrs)', 'Uptime', 'FPY %', 'Wound %', 'OEE Availability', 'Merge']]

    # 1. Statistical Comparison
    print("\n1. Mean Comparison Before vs. After Change Point:")
    for i, cp_index in enumerate(change_points):
        change_date = product_df.index[cp_index-1]
        print(f"\nAnalyzing shift on {change_date.date()}:")

        # Define 'before' and 'after' periods (e.g., 30 days window)
        before_df = full_df.loc[change_date - pd.Timedelta(days=30):change_date - pd.Timedelta(days=1)]
        after_df = full_df.loc[change_date:change_date + pd.Timedelta(days=29)]

        if before_df.empty or after_df.empty:
            print("  Skipping analysis due to insufficient data around change point.")
            continue

        # Compare means of top 5 changing variables
        mean_diff = (after_df[feature_cols].mean() - before_df[feature_cols].mean()).abs().sort_values(ascending=False)
        print("  Top 5 variables with largest mean change:")
        print(mean_diff.head(5))

    # 2. Machine Learning Feature Importance
    print("\n2. Feature Importance from Random Forest Model:")
    product_full_df = full_df[full_df['Merge'] == product_type]
    X = product_full_df[feature_cols]
    y = product_full_df[target_variable]

    if len(X) < 20:
        print("  Skipping model training due to insufficient data.")
        return

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_10_features = importances.sort_values(ascending=False).head(10)

    print("  Top 10 most influential variables on the target:")
    print(top_10_features)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    top_10_features.sort_values().plot(kind='barh', color='teal')
    plt.title(f'Top 10 Feature Importances for {target_variable} on {product_type}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


# --- 6. Main Execution Loop ---
target_variables = ['Downtime (Hrs)', 'Uptime', 'FPY %', 'Wound %', 'OEE Availability']
product_types = [p for p in df['Merge'].unique() if 'shutdown' not in p.lower()]

for product in product_types:
    for target in target_variables:
        # Find change points on the original (non-lagged) data for accuracy
        change_points, product_df = find_change_points(df.copy(), product, target)

        # Run analysis using the full lagged dataframe
        if product_df is not None:
             analyze_change_points(change_points, product_df, df_lagged.copy(), product, target)

