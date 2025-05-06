# Murphy Brown Impact Analysis

A modular Python system for analyzing the impact of Murphy Brown on verification processes.

## Overview

This system is a refactored and modularized version of a monolithic analysis script. It's designed to be more maintainable, testable, and easier to extend. The system is split into four main components:

1. **metrics.py** - Core metric calculation functions
2. **analysis.py** - Statistical analysis and comparison functions 
3. **viz.py** - Visualization and report generation
4. **main.py** - Command-line interface that ties everything together

## Requirements

- Python 3.6+
- pandas
- numpy 
- matplotlib
- seaborn
- scipy

You can install these dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Usage

### Command Line Interface

The simplest way to run the full analysis is through the command-line interface:

```bash
python main.py --input_file path/to/history_data.csv
```

### Command Line Options

- `--input_file`: (Required) Path to the input CSV file with history data
- `--output_dir`: Directory for output files (default: ./Output)
- `--filter_searchtype`: Comma-separated list of search types to include (default: empv,eduv)
- `--mb_agent`: Identifier for Murphy Brown agent (default: murphy.brown)
- `--skip_dashboard`: Skip dashboard creation (default: False)
- `--skip_summary`: Skip executive summary creation (default: False)

### Example

```bash
python main.py --input_file data/history_export.csv --output_dir results --filter_searchtype empv
```

## Using the Modules Separately

You can also use each module separately for more granular control:

```python
import pandas as pd
import metrics
import analysis
import viz

# Load your data
df_history = pd.read_csv('data/history_export.csv')

# Calculate metrics
all_metrics = metrics.calculate_all_metrics(df_history)

# Perform analysis
analysis_results = analysis.perform_full_analysis(all_metrics, df_history)

# Create visualizations
dashboard_dir = viz.create_dashboard(analysis_results, 'results/dashboard')

# Generate an executive summary
summary_path = viz.generate_executive_summary(analysis_results, 'results/reports')
```

## Output

The system generates several types of output:

1. **CSV Data Files** - Raw metrics and analysis results
2. **JSON Verdict Files** - Analysis verdicts in machine-readable format
3. **Visualizations** - PNG files showing key metrics and comparisons
4. **Dashboard** - Interactive HTML dashboard
5. **Executive Summary** - Markdown report summarizing key findings

## Module Details

### metrics.py

Contains functions for calculating basic metrics from history data:
- TAT (turnaround time) calculations
- Attempt counts
- MB capability detection (contact research, email handling)
- Autonomy metrics
- Contact plan detection

### analysis.py

Contains functions for analyzing the impact of MB:
- Merging metrics
- Comparative analysis between groups
- Statistical tests (t-tests, ANOVA)
- Time efficiency analysis
- Contact efficiency analysis
- FTE savings estimation

### viz.py

Contains functions for creating visualizations and reports:
- Dashboard creation
- Plots for key metrics
- Executive summary generation

### main.py

Command-line interface that:
- Parses arguments
- Loads and filters data
- Calls the appropriate functions from other modules
- Saves results
- Creates visualizations and reports

## Extending the System

You can extend the system by:

1. Adding new metric calculations to metrics.py
2. Adding new analysis functions to analysis.py
3. Adding new visualization types to viz.py
4. Updating main.py to include your new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.