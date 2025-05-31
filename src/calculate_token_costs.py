import pandas as pd
import numpy as np
import os

# --- Pricing schedule (add/adjust as needed) ---
PRICE_SCHEDULE = {
    'text-davinci-002-render-sha': {'input': 0.50, 'output': 1.50},
    'gpt-4':                       {'input': 30.00, 'output': 60.00},
    'gpt-4-mobile':                {'input': 30.00, 'output': 60.00},
    'gpt-4-browsing':              {'input': 30.00, 'output': 60.00},
    'gpt-4-plugins':               {'input': 30.00, 'output': 60.00},
    'text-davinci-002-render-sha-mobile': {'input': 0.50, 'output': 1.50},
    'gpt-4-gizmo':                 {'input': 30.00, 'output': 60.00},
    'gpt-3.5-turbo':               {'input': 0.50, 'output': 1.50},
    'gpt-4o':                      {'input': 2.50,  'output': 10.00},
    'o1-preview':                  {'input': 15.00, 'output': 60.00},
    'gpt-4o-canmore':              {'input': 2.50,  'output': 10.00},
    'o1':                          {'input': 15.00, 'output': 60.00},
    'o1-mini':                     {'input': 1.10,  'output': 4.40},
    'o3-mini':                     {'input': 1.10,  'output': 4.40},
    'o3-mini-high':                {'input': 1.10,  'output': 4.40},
    'gpt-4-5':                     {'input': 75.00, 'output': 150.00},
    'gpt-4o-mini':                 {'input': 0.15,  'output': 0.60},
    'o3':                          {'input': 10.00, 'output': 40.00},
    'o4-mini-high':                {'input': 1.10,  'output': 4.40},
    'gpt-4-1':                     {'input': 2.00,  'output': 8.00},
    'o4-mini':                     {'input': 1.10,  'output': 4.40},
}
DEFAULT_PRICES = {'input': np.nan, 'output': np.nan}

def calculate_token_costs(
    input_csv,
    output_csv,
    price_schedule=PRICE_SCHEDULE,
    debug=False
):
    """
    Calculate costs for input and output tokens per row in a token counts CSV.
    Saves the result as a new CSV with cost columns appended.
    """
    # --- Ensure output directory exists ---
    out_dir = os.path.dirname(output_csv)
    if not os.path.isdir(out_dir):
        if debug:
            print(f"[Debug] Creating output directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

    # --- Load token counts ---
    df = pd.read_csv(input_csv, dtype={'input_tokens': int, 'output_tokens': int, 'model': str})

    # --- Compute cost per row ---
    def compute_cost(row):
        model = row['model']
        prices = price_schedule.get(model, DEFAULT_PRICES)
        input_cost = (row['input_tokens'] / 1e6) * prices['input']
        output_cost = (row['output_tokens'] / 1e6) * prices['output']
        total_cost = input_cost + output_cost
        return pd.Series({
            'input_token_cost': input_cost,
            'output_token_cost': output_cost,
            'total_cost': total_cost
        })

    costs_df = df.apply(compute_cost, axis=1)

    # --- Merge costs back ---
    df = pd.concat([df, costs_df], axis=1)

    # --- Debug: report any models missing price entries ---
    if debug:
        missing = sorted(set(df['model']) - set(price_schedule.keys()))
        print(f"[Debug] Models with default (nan) prices: {missing}")

    # --- Display sample if debug ---
    if debug:
        try:
            from IPython.display import display
            display(df.head())
        except ImportError:
            print(df.head())

    # --- Save to CSV ---
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved token costs to {output_csv}, total records: {len(df)}")

    return df

# === CLI usage ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate token costs per message and save to CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input token_counts.csv")
    parser.add_argument("--output_csv", type=str, required=True, help="Path for output token_costs.csv")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    calculate_token_costs(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        debug=args.debug
    )
