import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Data Simulation & Preparation ---
# In a real scenario, you would load your data like this:
# df = pd.read_csv('your_plant_data.csv')

# For demonstration, let's create a realistic sample dataset.
np.random.seed(42)
dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=365, freq='D'))
data = {
    'GMT Prod Yield Date': dates,
    'Merge': np.random.choice(['ProductA', 'ProductB', 'ProductC', 'Shutdown-Planned'], size=365),
    'Downtime (Hrs)': np.random.rand(365) * 8,
    'Uptime': np.random.rand(365) * 16,
    'FPY %': 95 - np.random.rand(365) * 10,
    'Wound %': 98 - np.random.rand(365) * 5,
    'OEE Availability': 90 - np.random.rand(365) * 15
}
df = pd.DataFrame(data)

# Introduce some artificial shifts for demonstration
# ProductA FPY% drops after June 1st
df.loc[(df['Merge'] == 'ProductA') & (df['GMT Prod Yield Date'] > '2023-06-01'), 'FPY %'] -= 10
# ProductB Downtime increases after September 1st
df.loc[(df['Merge'] == 'ProductB') & (df['GMT Prod Yield Date'] > '2023-09-01'), 'Downtime (Hrs)'] += 4


# Convert date column to datetime objects and set as index
if not pd.api.types.is_datetime64_any_dtype(df['GMT Prod Yield Date']):
    df['GMT Prod Yield Date'] = pd.to_datetime(df['GMT Prod Yield Date'])
df = df.sort_values('GMT Prod Yield Date').set_index('GMT Prod Yield Date')

print("Data prepared. Head of the dataframe:")
print(df.head())
print("\nUnique product types in 'Merge' column:", df['Merge'].unique())


# --- 2. Change Point Detection Function ---
def find_and_plot_change_points(df, product_type, target_variable):
    """
    This function filters data for a specific product, applies the PELT algorithm
    to find change points in a target variable, and plots the results.

    Args:
        df (pd.DataFrame): The input dataframe.
        product_type (str): The product type from the 'Merge' column to analyze.
        target_variable (str): The name of the target variable column.
    """
    print(f"\n--- Analyzing: [Product: {product_type}] | [Target: {target_variable}] ---")

    # Filter data for the specific product and target, dropping any missing values
    product_df = df[df['Merge'] == product_type][[target_variable]].dropna()

    if len(product_df) < 20: # Need sufficient data to detect changes
        print(f"Skipping {product_type} for {target_variable} due to insufficient data points ({len(product_df)}).")
        return

    # Convert the series to a numpy array for the algorithm
    points = product_df[target_variable].values

    # --- Apply the PELT Algorithm ---
    # We use the 'l2' model which detects shifts in the mean.
    # The penalty value (pen) is crucial. A higher penalty results in fewer change points.
    # We can estimate a penalty based on the data's noise and size.
    # A common rule of thumb is pen = log(n) * sigma^2, where n is number of samples
    # and sigma is the residual standard deviation.
    n = len(points)
    sigma = np.std(points)
    penalty = 2 * np.log(n) * (sigma**2)

    algo = rpt.Pelt(model="l2").fit(points)
    result = algo.predict(pen=penalty)

    # The result includes the last index, so we remove it for plotting
    if len(result) > 0 and result[-1] == len(points):
        result = result[:-1]

    if not result:
        print(f"No significant change points detected for {product_type} - {target_variable}.")
    else:
        print(f"Detected {len(result)} change point(s) at indices: {result}")


    # --- 3. Visualization ---
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(product_df.index, points, label=f'Daily {target_variable}', color='skyblue', linewidth=2)

    # Draw vertical lines for each detected change point
    for i, cp_index in enumerate(result):
        # We need to get the actual date from the dataframe index
        change_date = product_df.index[cp_index-1]
        ax.axvline(x=change_date, color='crimson', linestyle='--', lw=2, label=f'Change Point {i+1} ({change_date.date()})')

    # Formatting the plot
    plt.title(f'Shift Detection for Product: {product_type}\nTarget Variable: {target_variable}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(target_variable, fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()


# --- 4. Main Execution Loop ---
# We will iterate through each product type (excluding shutdowns for this analysis)
# and each target variable.
target_variables = ['Downtime (Hrs)', 'Uptime', 'FPY %', 'Wound %', 'OEE Availability']
product_types = [p for p in df['Merge'].unique() if 'shutdown' not in p.lower()]

for product in product_types:
    for target in target_variables:
        find_and_plot_change_points(df.copy(), product, target)

