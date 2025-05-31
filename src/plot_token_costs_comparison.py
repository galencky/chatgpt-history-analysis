import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

DEBUG = True

DATA_DIR = "data"
TOKEN_COUNTS_CSV = os.path.join(DATA_DIR, "token_counts.csv")
COSTS_COMBINED_CSV = os.path.join(DATA_DIR, "monthly_token_cost_comparison.csv")
PLOT_NAIVE = os.path.join(DATA_DIR, "plot_naive_costs.png")
PLOT_EMU   = os.path.join(DATA_DIR, "plot_api_emulation_costs.png")
MONTHLY_PLUS = 21.0

# --- Model info ---
model_context_window = {
    'text-davinci-002-render-sha': 4096,
    'gpt-3.5-turbo': 16385,
    'gpt-4': 8192,
    'gpt-4-mobile': 8192, 'gpt-4-browsing': 8192,
    'gpt-4-plugins': 8192, 'gpt-4-gizmo': 8192,
    'gpt-4-5': 128000,
    'gpt-4o': 128000, 'gpt-4o-canmore': 128000,
    'gpt-4o-mini': 128000, 'gpt-4-1': 1000000,
    'o3': 200000, 'o3-mini': 200000, 'o3-mini-high': 200000,
    'o4-mini': 200000, 'o4-mini-high': 200000,
    'o1': 200000, 'o1-preview': 200000, 'o1-mini': 128000,
    'text-davinci-002-render-sha-mobile': 4096,
}
DEFAULT_CONTEXT_WINDOW = 8192

price_schedule = {
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
default_prices = {'input': np.nan, 'output': np.nan}
BUFFER_TOKENS = 300

def emulate_true_api_chat_cost(df):
    """Emulate OpenAI API billing: context window, accumulation, truncation."""
    df = df.copy()
    df['api_input_tokens'] = 0
    df['api_input_token_cost'] = 0.0
    df['api_output_token_cost'] = 0.0
    df['api_total_cost'] = 0.0

    for conv_id, group in df.groupby('conversation_id'):
        context = []
        for idx, row in group.iterrows():
            model = row['model']
            context_window = model_context_window.get(model, DEFAULT_CONTEXT_WINDOW)
            price = price_schedule.get(model, default_prices)
            # Add this message to context
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
            while total_prompt + BUFFER_TOKENS > context_window and len(context) > 1:
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

def main():
    # --- Load data ---
    df = pd.read_csv(TOKEN_COUNTS_CSV, dtype={'input_tokens': int, 'output_tokens': int, 'model': str})

    # --- Parse conversation_create_time for grouping ---
    if 'conversation_create_time' in df.columns:
        df['parsed_date'] = pd.to_datetime(df['conversation_create_time'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        df['month'] = df['parsed_date'].dt.strftime('%Y-%m')
    else:
        raise ValueError("conversation_create_time not found in dataframe!")

    # --- NAIVE monthly costs (no API emulation) ---
    def get_prices(row):
        prices = price_schedule.get(row['model'], default_prices)
        return pd.Series({'input_cost': (row['input_tokens'] / 1e6) * prices['input'],
                          'output_cost': (row['output_tokens'] / 1e6) * prices['output']})
    naive_costs = df.join(df.apply(get_prices, axis=1))
    naive_monthly = naive_costs.groupby('month')[['input_cost', 'output_cost']].sum().reset_index()
    naive_monthly['naive_total_cost'] = naive_monthly['input_cost'] + naive_monthly['output_cost']
    naive_monthly = naive_monthly.rename(columns={
        'input_cost': 'naive_input_cost',
        'output_cost': 'naive_output_cost'
    })

    # --- API EMULATION costs ---
    api_df = emulate_true_api_chat_cost(df)
    api_monthly = api_df.groupby('month')[['api_input_token_cost', 'api_output_token_cost']].sum().reset_index()
    api_monthly['api_total_cost'] = api_monthly['api_input_token_cost'] + api_monthly['api_output_token_cost']

    # --- Merge both on 'month' ---
    combined = pd.merge(naive_monthly, api_monthly, on='month', how='outer').sort_values('month')

    # --- Add gain/loss vs Plus for both ---
    combined['naive_gain_loss'] = combined['naive_total_cost'] - MONTHLY_PLUS
    combined['naive_gain_loss_pct'] = (combined['naive_gain_loss'] / MONTHLY_PLUS * 100).round(1)
    combined['api_gain_loss'] = combined['api_total_cost'] - MONTHLY_PLUS
    combined['api_gain_loss_pct'] = (combined['api_gain_loss'] / MONTHLY_PLUS * 100).round(1)

    # --- Add summary row ---
    summary = {
        'month': 'TOTAL',
        'naive_input_cost': combined['naive_input_cost'].sum(),
        'naive_output_cost': combined['naive_output_cost'].sum(),
        'naive_total_cost': combined['naive_total_cost'].sum(),
        'api_input_token_cost': combined['api_input_token_cost'].sum(),
        'api_output_token_cost': combined['api_output_token_cost'].sum(),
        'api_total_cost': combined['api_total_cost'].sum(),
        'naive_gain_loss': combined['naive_gain_loss'].sum(),
        'naive_gain_loss_pct': (combined['naive_gain_loss'].sum() / (MONTHLY_PLUS * len(combined))) * 100,
        'api_gain_loss': combined['api_gain_loss'].sum(),
        'api_gain_loss_pct': (combined['api_gain_loss'].sum() / (MONTHLY_PLUS * len(combined))) * 100
    }
    combined = pd.concat([combined, pd.DataFrame([summary])], ignore_index=True)

    # --- Save combined CSV ---
    combined.to_csv(COSTS_COMBINED_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ Saved combined monthly cost comparison to {COSTS_COMBINED_CSV}")
    print(combined)

    # --- Plot NAIVE ---
    plot_naive = combined[combined['month'] != 'TOTAL']
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(plot_naive['month'], plot_naive['naive_total_cost'],
            color=['#2ca02c' if g > 0 else '#d62728' for g in plot_naive['naive_gain_loss']],
            label='Naive (no API emulation)')
    ax1.axhline(MONTHLY_PLUS, color='grey', linestyle='--', linewidth=2, label='$21 GPT Plus Subscription')
    for i, (cost, gain) in enumerate(zip(plot_naive['naive_total_cost'], plot_naive['naive_gain_loss'])):
        sign = '+' if gain > 0 else ''
        ax1.text(i, cost + 0.5, f"{sign}{gain:.1f}", ha='center', va='bottom', fontsize=9)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Cost (USD)')
    ax1.set_title('Monthly Token Cost (Naive, No API Emulation) vs $21 GPT Plus')
    ax1.legend()
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_NAIVE, bbox_inches='tight')
    print(f"✅ Saved naive cost plot to {PLOT_NAIVE}")

    # --- Plot API EMULATION ---
    plot_emu = combined[combined['month'] != 'TOTAL']
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(plot_emu['month'], plot_emu['api_total_cost'],
            color=['#2ca02c' if g > 0 else '#d62728' for g in plot_emu['api_gain_loss']],
            label='API Emulation')
    ax2.axhline(MONTHLY_PLUS, color='grey', linestyle='--', linewidth=2, label='$21 GPT Plus Subscription')
    for i, (cost, gain) in enumerate(zip(plot_emu['api_total_cost'], plot_emu['api_gain_loss'])):
        sign = '+' if gain > 0 else ''
        ax2.text(i, cost + 0.5, f"{sign}{gain:.1f}", ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Total Cost (USD)')
    ax2.set_title('Monthly Token Cost (API Emulation) vs $21 GPT Plus')
    ax2.legend()
    ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_EMU, bbox_inches='tight')
    print(f"✅ Saved API emulation cost plot to {PLOT_EMU}")

if __name__ == "__main__":
    main()
