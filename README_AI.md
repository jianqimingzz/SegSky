## AI-assisted Experiment Analysis

SegSky provides an AI-assisted module to analyze query evaluation results after the core Stage A/B/C pipeline finishes.

The AI module does not change the index construction or query algorithm. It only reads the generated CSV results and produces a structured analysis report, parameter summaries, and visualization figures.

### Purpose

The module is used to analyze:

- Recall and QPS under different query ranges;
- the influence of `ef` on Recall, QPS, and average visited nodes;
- the best configuration under target Recall thresholds such as 0.90 and 0.95;
- parameter tuning suggestions;
- possible risks and limitations of the current experiment.

Because the CSV contains results from different query ranges, the analysis is performed range by range. The script groups results by `[T_START, T_END]` before computing statistics, instead of mixing all ranges together.

### Script

```bash
scripts/analyze_range_results_ai.py
```

Run with the latest CSV in `demo_outputs/results`:

```bash
python scripts/analyze_range_results_ai.py \
  --result_dir demo_outputs/results \
  --output_dir demo_outputs/ai_analysis \
  --config config/ai_config.json
```

Or specify one CSV file directly:

```bash
python scripts/analyze_range_results_ai.py \
  --input demo_outputs/results/grid_results_top10_base_10k_ws-10_qExternal.csv \
  --output_dir demo_outputs/ai_analysis \
  --config config/ai_config.json
```

### Configuration

The AI configuration file is:

```text
config/ai_config.json
```

Example configuration without API key:

```json
{
  "provider": "mock",
  "model": "deepseek-chat",
  "api_key": "",
  "base_url": "https://api.deepseek.com/chat/completions",
  "temperature": 0.3,
  "max_rounds": 3,
  "target_recalls": [0.9, 0.95],
  "language": "zh"
}
```

In `mock` mode, no external API is required, so the demo can be reproduced directly.

To use DeepSeek, set:

```json
{
  "provider": "deepseek",
  "model": "deepseek-chat",
  "api_key": "YOUR_DEEPSEEK_API_KEY",
  "base_url": "https://api.deepseek.com/chat/completions",
  "temperature": 0.3,
  "max_rounds": 3,
  "target_recalls": [0.9, 0.95],
  "language": "zh"
}
```

### Outputs

The analysis results are saved in:

```text
demo_outputs/ai_analysis/
```

Main output files:

```text
clean_results.csv        # cleaned result table
range_summary.csv        # range-wise best configurations
ai_range_report.md       # AI-assisted structured report
all_ranges_qps_vs_recall.png
range_xxx_recall_vs_ef.png
range_xxx_qps_vs_ef.png
range_xxx_avg_visited_vs_ef.png
```

### Notes

The AI module is designed as an experiment-analysis assistant. It helps users understand the relationship between parameters and performance, but it does not replace the SegSky algorithm itself.

Do not upload real API keys to GitHub. Add the following files to `.gitignore`:

```gitignore
config/ai_config.json
.env
```
