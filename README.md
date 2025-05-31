# ChatGPT History Analyzer

A **Python-based pipeline** to analyze and visualize your exported ChatGPT conversation data, compare API usage costs, and generate reports—with modular, scriptable steps and easy email delivery of your results.

---

### Features

* **Automatic import of ChatGPT exports** (from exported ZIP file).
* **Comprehensive flattening** of message data, web searches, and image generations.
* **Accurate model name recovery** using metadata.
* **Per-message token counting** (using OpenAI’s tiktoken).
* **True API emulation** (context window, pricing logic).
* **Monthly summaries and cost visualizations**:

  * Monthly conversations, messages, and messages per conversation
  * Per-model token usage (input/output, stacked bars)
  * Compare your API-equivalent token costs with ChatGPT Plus subscription
* **Automated report delivery via email**:

  * Plots, logs, and model usage summary as attachments
* **Logs all steps to `logs.txt`** for transparency and reproducibility
### What You Get in the Email

At the end of the analysis, you’ll receive an email with **all your monthly summary plots (as PNG images), a detailed model usage frequency table (CSV), and the full processing log (logs.txt)**. This gives you a clear, visualized breakdown of your ChatGPT usage, estimated API-equivalent costs (versus your subscription), and all processing steps for transparency and reproducibility—making it easy to decide if a subscription or pay-as-you-go API use best fits your needs.

### Why Token Counting and API Simulation?

**OpenAI charges by tokens for API usage, not by message.** Counting tokens for every message—split into user (input) and assistant (output) parts—lets you see your true usage patterns and how much an equivalent workflow would cost on the API. The simulation goes further: when you chat with the API, every message includes previous context, up to a fixed “context window” (number of tokens the model can “remember”). This means **your API cost is often much higher than just the sum of individual messages**—the pipeline faithfully simulates this by accumulating context and pruning as needed, using each model’s actual context window and pricing.

**The purpose of token counting and cost simulation is to help you evaluate whether you are better off continuing your ChatGPT Plus subscription, or switching to the pay-as-you-go API model.** The tool provides a direct, month-by-month comparison of your estimated API cost versus the \$21 monthly subscription.


---

## Directory Structure

```
chatgpt-history-analysis/
│
├─ main.py                # The main pipeline entrypoint
├─ requirements.txt       # Required packages
├─ .env                   # Email credentials (see below)
├─ /src/                  # All helper modules (see below)
├─ /data/                 # All outputs (CSVs, PNGs, logs)
│    ├─ chatgpt-YYYYMMDD-HHMM/   # Your extracted ChatGPT export
│    ├─ merged_conversations.csv
│    ├─ merged_conversations_filled.csv
│    ├─ token_counts.csv
│    ├─ token_costs_true_api_emulated.csv
│    ├─ model_usage_frequency.csv
│    ├─ monthly_conversations.png
│    ├─ monthly_messages.png
│    ├─ monthly_messages_per_conversation.png
│    ├─ monthly_token_usage_by_model.png
│    ├─ logs.txt
│    └─ ...etc
```

---

## Setup

1. **Clone this repo**

   ```
   git clone https://github.com/YOUR_USERNAME/chatgpt-history-analysis.git
   cd chatgpt-history-analysis
   ```

2. **Install requirements**

   ```
   pip install -r requirements.txt
   ```

3. **Prepare your `.env` file**
   Create a file named `.env` in the project root:

   ```
   EMAIL_USER=your_gmail@gmail.com
   EMAIL_PASS=your_app_password   # Use an App Password, not your main Gmail password!
   ```

4. **(Optional) Edit/Review configuration paths in `main.py` or modules if needed**

---

## Usage

### 1. **Export your ChatGPT history**

Go to [chat.openai.com](https://chat.openai.com), request an export, and download the `.zip` file.

### 2. **Run the pipeline**

```
python main.py
```

* The script will **prompt for your export zip path** (e.g., `C:\Users\you\Downloads\chatgpt-20250528-xxxx.zip`)
* It will **extract, organize, and process** the export, saving all results to the `data/` directory.
* At the end, it prompts for your **recipient email address** and sends you a full report (with attachments).

---

## Outputs

* **CSV files** with flattened, merged, and tokenized data
* **PNG plots** for monthly usage and per-model token stats
* **Comprehensive logs** (`logs.txt`) for all steps
* **Email report** (all outputs attached)

---

## Key Modules (in `/src/`)

* `import_export_zip.py` – Decompress and prep export ZIP
* `flatten_messages.py` – Flatten conversations to rows
* `flatten_websearch.py` – Extract search/thought/code records
* `flatten_images.py` – Extract image generations/uploads
* `merge_flattened.py` – Merge all sources, resolve conflicts
* `fill_model_names.py` – Backfill unknown models using JSON metadata
* `token_counter.py` – Count tokens per message (OpenAI-style)
* `emulate_api_chat_costs.py` – Simulate API chat costs and context windows
* `plot_monthly_summary.py` – Generate monthly summary plots
* `plot_token_costs_comparison.py` – Plot naive vs API-emulated costs
* `send_email_report.py` – Send all outputs via Gmail

---

## Security and Privacy

* **No data leaves your machine** except for the optional email step (and that is only to your chosen address).
* **Logs are saved** and attached to your email for full transparency.

---

## Requirements

See `requirements.txt`. Typical packages include:

```
pandas
numpy
matplotlib
tiktoken
python-dotenv
```

---

## Troubleshooting

* **ZIP not found**: Check your input path.
* **Email not sent**: Ensure you use an [App Password for Gmail](https://support.google.com/accounts/answer/185833?hl=en).
* **Missing plots or CSVs**: Review the logs in `data/logs.txt`.
* **Error: Argument mismatch**: Ensure your function calls match the module definitions.

---

## To Do

* [ ] Add more flexible plotting options
* [ ] Support for non-Gmail SMTP providers
* [ ] CLI for headless mode (no interactive prompts)
* [ ] Advanced analytics: topic classification, time-of-day heatmaps, etc.

---

## License

MIT

---

**Pull requests and suggestions welcome!**
If you found this useful, [star ⭐️ the repo](#) or open an issue!

Author: Kuan-Yuan Chen, M.D.
Email: galen147258369@gmail.com

---

*Generated with help from ChatGPT and iteratively debugged by the project author!*

---


