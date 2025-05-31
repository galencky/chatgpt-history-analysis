import pandas as pd
import numpy as np
import os

# --- Default context windows ---
MODEL_CONTEXT_WINDOW = {
    'text-davinci-002-render-sha': 4096,
    'gpt-3.5-turbo': 16385,
    'gpt-4': 8192,
    'gpt-4-mobile': 8192, 'gpt-4-browsing': 8192, 'gpt-4-plugins': 8192, 'gpt-4-gizmo': 8192,
    'gpt-4-5': 128000,
    'gpt-4o': 128000, 'gpt-4o-canmore': 128000, 'gpt-4o-mini': 128000, 'gpt-4-1': 1000000,
    'o3': 200000, 'o3-mini': 200000, 'o3-mini-high': 200000, 'o4-mini': 200000, 'o4-mini-high': 200000,
    'o1': 200000, 'o1-preview': 200000, 'o1-mini': 128000,
    'text-davinci-002-render-sha-mobile': 4096,
}
DEFAULT_CONTEXT_WINDOW = 8192

# --- Default prices ---
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

BUFFER_TOKENS = 300

def emulate_true_api_chat_cost(
    df,
    model_context_window=MODEL_CONTEXT_WINDOW,
    price_schedule=PRICE_SCHEDULE,
    default_context_window=DEFAULT_CONTEXT_WINDOW,
    default_prices=DEFAULT_PRICES,
    buffer_tokens=BUFFER_TOKENS,
    debug=False
):
    """
    Emulates true OpenAI API chat cost for each message, considering context window, buffer, and pruning.
    Returns a new DataFrame with API emulated cost columns.
    """
    df = df.copy()
    df['api_input_tokens'] = 0
    df['api_input_token_cost'] = 0.0
    df['api_output_token_cost'] = 0.0
    df['api_total_cost'] = 0.0

    for conv_id, group in df.groupby('conversation_id'):
        context = []
        for idx, row in group.iterrows():
            model = row['model']
            context_window = model_context_window.get(model, default_context_window)
            price = price_schedule.get(model, default_prices)

            context.append({
                'input_tokens': row['input_tokens'],
                'output_tokens': row['output_tokens'],
                'model': model,
                'row_idx': idx
            })

            prompt_tokens = []
            for msg in context:
                if msg['input_tokens'] > 0:
                    prompt_tokens.append(msg['input_tokens'])
                if msg['output_tokens'] > 0:
                    prompt_tokens.append(msg['output_tokens'])
            total_prompt = sum(prompt_tokens)

            while total_prompt + buffer_tokens > context_window and len(context) > 1:
                removed = context.pop(0)
                if removed['input_tokens'] > 0:
                    prompt_tokens.pop(0)
                if removed['output_tokens'] > 0:
                    prompt_tokens.pop(0)
                total_prompt = sum(prompt_tokens)

            api_input_tokens = total_prompt
            input_cost = (api_input_tokens / 1e6) * price['input']
            output_cost = (row['output_tokens'] / 1e6) * price['output']
            total_cost = input_cost + output_cost

            df.at[idx, 'api_input_tokens'] = api_input_tokens
            df.at[idx, 'api_input_token_cost'] = input_cost
            df.at[idx, 'api_output_token_cost'] = output_cost
            df.at[idx, 'api_total_cost'] = total_cost

    return df

def main(
    input_csv,
    output_csv,
    debug=False
):
    """
    Full workflow: load CSV, emulate API cost, save result.
    """
    # Load token counts
    df = pd.read_csv(input_csv, dtype={'input_tokens': int, 'output_tokens': int, 'model': str})

    # Sort DataFrame for correct API emulation order
    if 'conversation_id' in df.columns and 'message_id' in df.columns:
        df = df.sort_values(['conversation_id', 'message_id'])
    elif 'conversation_id' in df.columns and 'create_time' in df.columns:
        df = df.sort_values(['conversation_id', 'create_time'])
    elif 'create_time' in df.columns:
        df = df.sort_values(['create_time'])

    # Emulate API chat cost
    df = emulate_true_api_chat_cost(df, debug=debug)

    # Debug: models not in price_schedule
    if debug:
        missing = sorted(set(df['model']) - set(PRICE_SCHEDULE.keys()))
        print(f"[Debug] Models with default (nan) prices: {missing}")

    # Show DataFrame head
    if debug:
        try:
            from IPython.display import display
            display(df.head())
        except ImportError:
            print(df.head())

    # Save to CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved true API emulation costs to {output_csv}, total records: {len(df)}")

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Emulate OpenAI API chat costs with context window logic.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to token_counts.csv")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV for true API emulation costs")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        debug=args.debug
    )
