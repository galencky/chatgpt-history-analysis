import pandas as pd
import matplotlib.pyplot as plt

def analyze_model_usage(merged_csv_path="data/merged_conversations.csv", show_table=True):
    """
    Analyzes model usage frequency and first use from the merged conversations CSV.
    Returns:
        stats: DataFrame with columns [model, frequency, first_use]
        total_frequency: int, total number of messages
    """
    # --- 1) Load merged CSV ---
    merged = pd.read_csv(merged_csv_path, dtype=str)

    # --- 2) Fill missing 'model' entries ---
    merged['model'] = merged['model'].fillna('unknown')

    # --- 3) Parse 'create_time' as datetime ---
    merged['create_time_parsed'] = pd.to_datetime(
        merged['create_time'],
        format='%Y%m%d_%H%M%S.%f',
        errors='coerce'
    )

    # --- 4) Group by model: frequency & first use ---
    stats = (
        merged
        .groupby('model', as_index=False)
        .agg(
            frequency=('model', 'size'),
            first_use=('create_time_parsed', 'min')
        )
    )

    # --- 5) Sort chronologically by first use ---
    stats = stats.sort_values('first_use').reset_index(drop=True)

    # --- 6) Total frequency ---
    total_frequency = stats['frequency'].sum()

    # --- 7) Optionally display ---
    if show_table:
        print("Model usage frequency (chronologically ordered):")
        try:
            from IPython.display import display
            display(stats)
        except ImportError:
            print(stats)
        print(f"Total messages across all models: {total_frequency}")

    return stats, total_frequency

# === CLI/Script usage ===
if __name__ == "__main__":
    stats, total = analyze_model_usage()
