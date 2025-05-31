import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cm
import os

# --- 0) Set up paths and output directory ---
DATA_DIR = "data"
MSG_CSV = os.path.join(DATA_DIR, "merged_conversations_filled.csv")
TOK_CSV = os.path.join(DATA_DIR, "token_counts.csv")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def plot_monthly_summary(merged_csv_path="data/merged_conversations_filled.csv", output_dir="data"):

    # === 1. Monthly message and conversation plots ===
    msg_df = pd.read_csv(MSG_CSV, dtype=str)

    # Unique conversation and message counts
    unique_conversations = msg_df['conversation_id'].nunique()
    unique_messages = msg_df['message_id'].nunique()
    print(f"Number of unique conversation_id: {unique_conversations}")
    print(f"Number of unique message_id: {unique_messages}")

    # Parse datetime columns
    msg_df['conversation_create_time'] = pd.to_datetime(
        msg_df['conversation_create_time'],
        format='%Y%m%d_%H%M%S.%f',
        errors='coerce'
    )
    msg_df['create_time'] = pd.to_datetime(
        msg_df['create_time'],
        format='%Y%m%d_%H%M%S.%f',
        errors='coerce'
    )

    # Monthly aggregates
    conv_df = msg_df[['conversation_id', 'conversation_create_time']].drop_duplicates()
    conv_df['month'] = conv_df['conversation_create_time'].dt.to_period('M')
    monthly_conversations = conv_df.groupby('month').size()
    msg_df['month'] = msg_df['create_time'].dt.to_period('M')
    monthly_messages = msg_df.groupby('month').size()
    monthly_ratio = (monthly_messages / monthly_conversations).fillna(0)

    # Plot monthly conversations
    fig1 = plt.figure(figsize=(10, 5))
    plt.bar(monthly_conversations.index.astype(str), monthly_conversations.values)
    plt.title('Historical Monthly Conversation Count')
    plt.xlabel('Month')
    plt.ylabel('Number of Conversations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "monthly_conversations.png"), bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ Saved: {os.path.join(DATA_DIR, 'monthly_conversations.png')}")

    # Plot monthly messages
    fig2 = plt.figure(figsize=(10, 5))
    plt.bar(monthly_messages.index.astype(str), monthly_messages.values)
    plt.title('Historical Monthly Message Count')
    plt.xlabel('Month')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "monthly_messages.png"), bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Saved: {os.path.join(DATA_DIR, 'monthly_messages.png')}")

    # Plot monthly messages per conversation
    fig3 = plt.figure(figsize=(10, 5))
    plt.bar(monthly_ratio.index.astype(str), monthly_ratio.values)
    plt.title('Historical Monthly Messages per Conversation')
    plt.xlabel('Month')
    plt.ylabel('Messages per Conversation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "monthly_messages_per_conversation.png"), bbox_inches='tight')
    plt.close(fig3)
    print(f"✅ Saved: {os.path.join(DATA_DIR, 'monthly_messages_per_conversation.png')}")

    # === 2. Monthly token usage (input/output, stacked by model) ===
    df = pd.read_csv(TOK_CSV, dtype=str)

    # Parse date and numeric columns
    df['conversation_create_time'] = pd.to_datetime(
        df['conversation_create_time'],
        format='%Y%m%d_%H%M%S.%f',
        errors='coerce'
    )
    df['month'] = df['conversation_create_time'].dt.to_period('M').astype(str)
    df['input_tokens'] = pd.to_numeric(df['input_tokens'], errors='coerce').fillna(0).astype(int)
    df['output_tokens'] = pd.to_numeric(df['output_tokens'], errors='coerce').fillna(0).astype(int)

    # Pivot tables: month x model
    pivot_in = df.pivot_table(index='month', columns='model', values='input_tokens', aggfunc='sum', fill_value=0)
    pivot_out = df.pivot_table(index='month', columns='model', values='output_tokens', aggfunc='sum', fill_value=0)
    models = sorted(set(pivot_in.columns) | set(pivot_out.columns))
    months = list(pivot_in.index)
    x = range(len(months))
    width = 0.4

    # Generate colors for each model
    cmap = cm.get_cmap('hsv', len(models))
    colors = [cmap(i) for i in range(len(models))]

    fig4, ax = plt.subplots(figsize=(14, 7))
    bottom_in = [0] * len(months)
    bottom_out = [0] * len(months)

    for i, model in enumerate(models):
        color = colors[i]
        vals_in = pivot_in.get(model, [0] * len(months))
        ax.bar([xi - width/2 for xi in x], vals_in, width,
            bottom=bottom_in, color=color, alpha=0.5)
        bottom_in = [bot + v for bot, v in zip(bottom_in, vals_in)]
        
        vals_out = pivot_out.get(model, [0] * len(months))
        ax.bar([xi + width/2 for xi in x], vals_out, width,
            bottom=bottom_out, color=color, alpha=1.0, label=model)
        bottom_out = [bot + v for bot, v in zip(bottom_out, vals_out)]

    # Y-axis formatting in millions
    max_in = pivot_in.sum(axis=1).max()
    max_out = pivot_out.sum(axis=1).max()
    y_max = max(max_in, max_out) * 1.1
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}M'))

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_xlabel('Month')
    ax.set_ylabel('Token Count (Millions)')
    ax.set_title('Monthly Token Usage by Model (Input vs Output)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    fig4.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "monthly_token_usage_by_model.png"), bbox_inches='tight')
    plt.close(fig4)
    print(f"✅ Saved: {os.path.join(DATA_DIR, 'monthly_token_usage_by_model.png')}")

    pass

if __name__ == "__main__":
    plot_monthly_summary()                # <-- keep demo call if you like
    print("🎉 All summary plots are saved in the data/ folder!")