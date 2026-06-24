from dataretrieval import waterdata
import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Fetch the data
# ============================================================
print("Fetching Delaware River temperature data...")

df, metadata = waterdata.get_daily(
    monitoring_location_id="USGS-01463500",
    parameter_code="00010",
    statistic_id="00003",
    time="2015-01-01/2024-12-31"
)

# Keep only the columns we need
df = df[['time', 'value']].copy()
df.columns = ['date', 'temp_C']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"\nRaw fetch: {len(df)} rows")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Missing temp values: {df['temp_C'].isnull().sum()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================
# STEP 2: Build a complete daily date range and fill gaps
# ============================================================
print("\n--- STEP 2: Fill missing days ---")

full_dates = pd.DataFrame({
    'date': pd.date_range('2015-01-01', '2024-12-31', freq='D')
})

df = full_dates.merge(df, on='date', how='left')

missing_before = df['temp_C'].isnull().sum()
print(f"Missing days after merge: {missing_before}")

# Forward fill gaps (carries last known temperature forward)
df['temp_C'] = df['temp_C'].ffill()

missing_after = df['temp_C'].isnull().sum()
print(f"Missing days after forward fill: {missing_after}")
print(f"Total rows: {len(df)}")

# ============================================================
# STEP 3: Resample to weekly means
# ============================================================
print("\n--- STEP 3: Resample to weekly means ---")

df = df.set_index('date')
weekly = df['temp_C'].resample('W-MON').mean().reset_index()
weekly.columns = ['week_start', 'temp_C_mean']

# Drop any weeks with NaN (shouldn't be any after ffill)
weekly = weekly.dropna().reset_index(drop=True)

print(f"Weekly rows: {len(weekly)}")
print(f"Date range: {weekly['week_start'].min()} to {weekly['week_start'].max()}")
print(f"\nFirst few rows:")
print(weekly.head(10))

# ============================================================
# STEP 4: Encode as +1 / 0 / -1 trends
# ============================================================
print("\n--- STEP 4: Encode temperature trends ---")

# Calculate week-to-week change
weekly['temp_change'] = weekly['temp_C_mean'].diff()

# Try several thresholds and check distribution
for threshold in [0.5, 1.0, 1.5, 2.0]:
    weekly['trend'] = 0
    weekly.loc[weekly['temp_change'] > threshold, 'trend'] = 1
    weekly.loc[weekly['temp_change'] < -threshold, 'trend'] = -1

    counts = weekly['trend'].value_counts().sort_index()
    total = len(weekly.dropna())
    print(f"\nThreshold = {threshold} degC:")
    print(f"  -1 (decreasing): {counts.get(-1, 0)} ({counts.get(-1, 0) / total * 100:.1f}%)")
    print(f"   0 (stable):     {counts.get(0, 0)} ({counts.get(0, 0) / total * 100:.1f}%)")
    print(f"  +1 (increasing): {counts.get(1, 0)} ({counts.get(1, 0) / total * 100:.1f}%)")

# ============================================================
# STEP 5: Check sliding windows would work
# ============================================================
print("\n--- STEP 5: Check sliding window construction ---")

# Use threshold 1.0 for this check
weekly['trend'] = 0
weekly.loc[weekly['temp_change'] > 1.0, 'trend'] = 1
weekly.loc[weekly['temp_change'] < -1.0, 'trend'] = -1

# Drop the first row (NaN change) and reset
sequence = weekly.dropna(subset=['temp_change'])['trend'].values

window_length = 4
windows = []
targets = []
for i in range(len(sequence) - window_length):
    windows.append(sequence[i:i + window_length])
    targets.append(sequence[i + window_length])

print(f"Sequence length: {len(sequence)}")
print(f"Windows created with length {window_length}: {len(windows)}")
print(f"\nFirst 5 windows and targets:")
for i in range(5):
    print(f"  Input: {windows[i]}  ->  Target: {targets[i]}")

print("\nDone. Review threshold distribution above and choose the best one.")