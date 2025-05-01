import pandas as pd

# Load CSV
df = pd.read_csv("auv/data/raw/stasmc_fault_T3_curve_spiral_adaptive_data_6.csv")

# Extract time as a DataFrame
time_df = df[["__time"]].rename(columns={"__time": "time"})
time_df["time"] = time_df["time"] - time_df["time"].iloc[0]

# Create column groups and rename them
eta_cols = [f"/xplorer_mini/gnc/auv_status/eta[{i}]" for i in range(6)]
eta_df = df[eta_cols].rename(columns={col: f"eta_{i}" for i, col in enumerate(eta_cols)})

nu_cols = [f"/xplorer_mini/gnc/auv_status/nu[{i}]" for i in range(6)]
nu_df = df[nu_cols].rename(columns={col: f"nu_{i}" for i, col in enumerate(nu_cols)})

nu_diff_cols = [f"/xplorer_mini/gnc/auv_status/nu_diff[{i}]" for i in range(6)]
nu_diff_df = df[nu_diff_cols].rename(columns={col: f"nu_diff_{i}" for i, col in enumerate(nu_diff_cols)})

tau_cols = [f"/xplorer_mini/gnc/auv_status/tau[{i}]" for i in range(6)]
tau_df = df[tau_cols].rename(columns={col: f"tau_{i}" for i, col in enumerate(tau_cols)})

# Combine all DataFrames
combined_df = pd.concat([time_df, eta_df, nu_df, nu_diff_df, tau_df], axis=1)

# Display or export
combined_df.to_csv("auv/data/processed/stasmc_fault_T3_curve_spiral_adaptive_data_6.csv", index=False)
