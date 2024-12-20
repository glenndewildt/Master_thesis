import pandas as pd
from scipy.stats import pearsonr
import hvplot.pandas  # HoloViews-based plotting extension for Pandas
from sklearn.preprocessing import MinMaxScaler


# Load the CSV files
file1 = 'data_extraction/labels.csv'
file2 = 'data_extraction/labels_og.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
print(len(df1['upper_belt']))
print(len(df2['upper_belt']))
# Convert non-numeric 'upper_belt' values to NaN
df1['upper_belt'] = pd.to_numeric(df1['upper_belt'], errors='coerce')
df2['upper_belt'] = pd.to_numeric(df2['upper_belt'], errors='coerce')
print(len(df1['upper_belt']))
print(len(df2['upper_belt']))
# Ensure that NaN positions are the same in both arrays
nan_positions = df1['upper_belt'].isna() | df2['upper_belt'].isna()
df1.loc[nan_positions, 'upper_belt'] = float('nan')
df2.loc[nan_positions, 'upper_belt'] = float('nan')

# Extract the 'upper_belt' values and fill NaN with 0
upper_belt1 = df1['upper_belt'].fillna(0).values.reshape(-1, 1)
upper_belt2 = df2['upper_belt'].fillna(0).values.reshape(-1, 1)

# Normalize the values between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
print(len(upper_belt1))
print(len(upper_belt2))

upper_belt1_normalized = scaler.fit_transform(upper_belt1).flatten()
upper_belt2_normalized = scaler.fit_transform(upper_belt2).flatten()

# Calculate the Pearson correlation coefficient
correlation, _ = pearsonr(upper_belt1_normalized, upper_belt2_normalized)

# Create a DataFrame with normalized values for plotting
df_plot = pd.DataFrame({
    'Index': range(len(upper_belt1_normalized)),
    'File 1 Upper Belt': upper_belt1_normalized,
    'File 2 Upper Belt': upper_belt2_normalized
})

# Plot the normalized values using HoloViews (hvPlot)
plot = df_plot.hvplot.line(x='Index', y=['File 1 Upper Belt', 'File 2 Upper Belt'], title=f'Normalized Upper Belt Values\nPearson Correlation: {correlation:.2f}', ylabel='Normalized Upper Belt Value')

# Show the plot in a GUI window
hvplot.show(plot)