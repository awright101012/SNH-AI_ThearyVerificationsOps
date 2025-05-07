#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
murphy_brown_impact_analysis_fullGeminiRefacv7empv15minsextraV18gemiy

A comprehensive analysis script to quantify Murphy Brown's impact on the verification process,
focusing on both contact research and email handling capabilities.

Usage:
murphy_brown_impact_analysis_fullGeminiRefacv7empv15minsextraV18gemiy
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import traceback
import json # Add json import for saving verdicts
from pathlib import Path

# ------------------------------------------------------------------
# Helper: dump any verdict-dict to Output/verdicts/<stem>_YYYYMMDD.json
from pathlib import Path

def save_verdict(obj: dict, stem: str, output_dir: str | None = None) -> None:
    """Persist a verdict dictionary; silently skip if obj is falsy."""
    if not obj:
        return

    if output_dir is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:            # interactive fallback
            script_dir = os.getcwd()
        output_dir = os.path.join(script_dir, "Output", "verdicts")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{stem}_{datetime.now():%Y%m%d}.json"
    with open(Path(output_dir, fname), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
# ------------------------------------------------------------------

# --- Configuration Settings ---
VERBOSITY_LEVEL = 3  # 1 - standard output (key metrics and steps)
                     # 2 - detailed output (all metrics and statistics)

# --- Rich Library for Formatted Console Output ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.box import ROUNDED
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    # print("Rich library not available. Install with: pip install rich")
    # print("Falling back to standard output formatting.")
    RICH_AVAILABLE = False
    console = None # Define console as None if rich is not available

# --- Helper function for verbosity-controlled output ---
def print_info(message, level=1):
    """
    Print information based on verbosity level.
    Level 1: Basic information (key steps and metrics)
    Level 2: Detailed information (all metrics and statistics)
    """
    if VERBOSITY_LEVEL >= level:
        if RICH_AVAILABLE and console is not None:
            console.print(message)
        else:
            print(message)

# --- Formatted table for TAT comparison ---
def print_tat_comparison(tat_stats):
    """
    Print a nicely formatted table comparing MB vs Non-MB TAT metrics.
    """
    if tat_stats is None or tat_stats.empty:
        print_info("TAT stats are not available for comparison.", level=1)
        return

    if not RICH_AVAILABLE or console is None:
        # Fallback formatting
        print_info("\n--- TAT Comparison: MB vs Non-MB ---", level=1)
        for _, row in tat_stats.iterrows():
            group = "MB" if row['mb_touched'] == 1 else "Non-MB"
            print_info(f"{group}: Count={row.get('search_count', 'N/A')}, Avg TAT (Calendar)={row.get('avg_tat_days', float('nan')):.2f}, Avg TAT (Business)={row.get('avg_tat_business', float('nan')):.2f}", level=1)
        return
        
    # Rich table implementation
    table = Table(title="TAT Comparison: MB vs Non-MB")
    
    table.add_column("Group", style="cyan")
    table.add_column("Search Count", justify="right", style="green")
    table.add_column("Avg TAT (Calendar)", justify="right")
    table.add_column("Avg TAT (Business)", justify="right")
    table.add_column("Median TAT (Calendar)", justify="right")
    table.add_column("Median TAT (Business)", justify="right")
    
    for _, row in tat_stats.iterrows():
        group = "MB" if row['mb_touched'] == 1 else "Non-MB"
        table.add_row(
            group,
            str(row.get('search_count', 'N/A')),
            f"{row.get('avg_tat_days', float('nan')):.2f}",
            f"{row.get('avg_tat_business', float('nan')):.2f}",
            f"{row.get('median_tat_days', float('nan')):.2f}",
            f"{row.get('median_tat_business', float('nan')):.2f}"
        )
    
    console.print(table)

# --- Dashboard for key metrics ---
def print_metric_dashboard(all_results):
    """
    Print a dashboard with key metrics from the analysis.
    """
    # Extract metrics from results
    autonomy = all_results.get('autonomy', {}).get('summary_metrics', {})
    time_savings = all_results.get('time_savings', {})
    
    if not RICH_AVAILABLE or console is None:
        # Fallback formatting
        print_info("\n--- Murphy Brown Impact Dashboard ---", level=1)
        print_info(f"End-to-End Autonomy Rate: {autonomy.get('autonomy_rate', 'N/A'):.1f}%", level=1)
        print_info(f"Rework Rate: {autonomy.get('rework_rate', 'N/A'):.1f}%", level=1)
        print_info(f"Average Human Touches: {autonomy.get('avg_human_touches', 'N/A'):.1f}", level=1)
        print_info(f"Total Hours Saved: {time_savings.get('total_hours_saved', 'N/A'):.1f}", level=1)
        print_info(f"FTE Equivalent: {time_savings.get('fte_equivalent', 'N/A'):.2f}", level=1)
        return
    
    # Create dashboard with Rich
    dashboard = Table(title="Murphy Brown Impact Dashboard", show_header=False, box=ROUNDED)
    dashboard.add_column("Metric", style="cyan")
    dashboard.add_column("Value", style="green")
    
    # Add autonomy metrics
    dashboard.add_row("End-to-End Autonomy Rate", f"{autonomy.get('autonomy_rate', float('nan')):.1f}%")
    dashboard.add_row("Rework Rate", f"{autonomy.get('rework_rate', float('nan')):.1f}%")
    dashboard.add_row("Average Human Touches", f"{autonomy.get('avg_human_touches', float('nan')):.1f}")
    
    # Add time savings
    dashboard.add_row("Total Hours Saved", f"{time_savings.get('total_hours_saved', float('nan')):.1f}")
    dashboard.add_row("FTE Equivalent", f"{time_savings.get('fte_equivalent', float('nan')):.2f}")
    
    # Print dashboard
    console.print(dashboard)

# --- Console Executive summary ---
def print_console_executive_summary(results): # Renamed to avoid conflict
    """
    Print an executive summary of the analysis results to the console.
    """
    # Extract key metrics
    autonomy = results.get('autonomy', {}).get('summary_metrics', {})
    time_savings = results.get('time_savings', {})
    tat_stats_data = results.get('time_efficiency', {}).get('tat_stats') # Use a different variable name
    
    # Calculate key differences
    tat_diff_str = "N/A" # Renamed to avoid conflict
    if tat_stats_data is not None and not tat_stats_data.empty and \
       'mb_touched' in tat_stats_data.columns and \
       1 in tat_stats_data['mb_touched'].values and 0 in tat_stats_data['mb_touched'].values and \
       'avg_tat_days' in tat_stats_data.columns:
        mb_tat_series = tat_stats_data.loc[tat_stats_data['mb_touched'] == 1, 'avg_tat_days']
        non_mb_tat_series = tat_stats_data.loc[tat_stats_data['mb_touched'] == 0, 'avg_tat_days']
        if not mb_tat_series.empty and not non_mb_tat_series.empty:
            mb_tat = mb_tat_series.values[0]
            non_mb_tat = non_mb_tat_series.values[0]
            if pd.notna(mb_tat) and pd.notna(non_mb_tat):
                tat_val_diff = non_mb_tat - mb_tat # Value difference
                tat_perc_diff = (tat_val_diff / non_mb_tat * 100) if non_mb_tat != 0 else 0
                tat_diff_str = f"{tat_val_diff:.2f} days ({tat_perc_diff:.1f}%)"
    
    # Also calculate attempts difference
    attempts_diff_str = "N/A" # Renamed
    contact_efficiency_data = results.get('contact_efficiency', {}) # Renamed
    attempts_stats_data = contact_efficiency_data.get('attempts_stats') # Renamed
    if attempts_stats_data is not None and not attempts_stats_data.empty and \
       'mb_touched' in attempts_stats_data.columns and \
       1 in attempts_stats_data['mb_touched'].values and 0 in attempts_stats_data['mb_touched'].values and \
       'avg_attempts' in attempts_stats_data.columns:
        mb_attempts_series = attempts_stats_data.loc[attempts_stats_data['mb_touched'] == 1, 'avg_attempts']
        non_mb_attempts_series = attempts_stats_data.loc[attempts_stats_data['mb_touched'] == 0, 'avg_attempts']
        if not mb_attempts_series.empty and not non_mb_attempts_series.empty:
            mb_attempts = mb_attempts_series.values[0]
            non_mb_attempts = non_mb_attempts_series.values[0]
            if pd.notna(mb_attempts) and pd.notna(non_mb_attempts):
                attempts_val_diff = non_mb_attempts - mb_attempts # Value difference
                attempts_perc_diff = (attempts_val_diff / non_mb_attempts * 100) if non_mb_attempts != 0 else 0
                attempts_diff_str = f"{attempts_val_diff:.2f} attempts ({attempts_perc_diff:.1f}%)"
    
    if not RICH_AVAILABLE or console is None:
        # Fallback formatting
        print_info("\n=== CONSOLE EXECUTIVE SUMMARY ===", level=1)
        print_info(f"Autonomy Rate: {autonomy.get('autonomy_rate', float('nan')):.1f}%", level=1)
        print_info(f"TAT Improvement: {tat_diff_str}", level=1)
        print_info(f"Attempts Reduction: {attempts_diff_str}", level=1)
        print_info(f"Total Hours Saved: {time_savings.get('total_hours_saved', float('nan')):.1f}", level=1)
        print_info(f"FTE Equivalent: {time_savings.get('fte_equivalent', float('nan')):.2f}", level=1)
        print_info(f"Dashboard available at: {results.get('dashboard_dir', 'N/A')}/dashboard.html", level=1)
        return
    
    # Create rich panel for executive summary
    summary_text = (
        f"[bold cyan]CONSOLE EXECUTIVE SUMMARY[/bold cyan]\n\n"
        f"[bold]Autonomy Rate:[/bold] {autonomy.get('autonomy_rate', float('nan')):.1f}%\n"
        f"[bold]TAT Improvement:[/bold] {tat_diff_str}\n"
        f"[bold]Attempts Reduction:[/bold] {attempts_diff_str}\n"
        f"[bold]Total Hours Saved:[/bold] {time_savings.get('total_hours_saved', float('nan')):.1f}\n"
        f"[bold]FTE Equivalent:[/bold] {time_savings.get('fte_equivalent', float('nan')):.2f}\n\n"
        f"Dashboard available at: {results.get('dashboard_dir', 'N/A')}/dashboard.html"
    )
    console.print(Panel.fit(
        summary_text,
        title="Murphy Brown Impact Analysis",
        border_style="green"
    ))

# --- Rich Table Helper Functions ---
def table_from_series(s, title):
    t = Table(title=title, box=None, show_header=False)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in s.items():
        val = f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)
        t.add_row(str(k), val)
    if RICH_AVAILABLE and console:
        console.print(t)
    elif not RICH_AVAILABLE: # Fallback if Rich is not available
        print(f"\n--- {title} ---")
        for k, v in s.items():
            val = f"{v:,.2f}" if isinstance(v, (float)) else f"{v:,}" if isinstance(v, int) else str(v)
            print(f"{str(k)}: {val}")


def table_from_df(df, title):
    t = Table(title=title, box=None, show_header=True)
    for col in df.columns:
        t.add_column(str(col), justify="right") # Ensure column names are strings
    for _, row in df.iterrows():
        # Ensure all row items are strings for Rich table
        t.add_row(*[str(x) for x in row])
    if RICH_AVAILABLE and console:
        console.print(t)
    elif not RICH_AVAILABLE: # Fallback if Rich is not available
        print(f"\n--- {title} ---")
        print(df.to_string())
# --- END Rich Table Helper Functions ---

# ──────────────────── mb_extra_slices helper block ────────────────────
"""
Utility slices that the main script can call to produce "one-off"
tables/figures that aren't worth baking into the core metric pipeline.
"""

from scipy import stats                  # local to the block → avoids polluting top-level namespace
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
import warnings

# ---------------------------------------------------------------------
# 1)  Contact-plan vs No-plan  ── TAT comparison
# ---------------------------------------------------------------------
def contact_plan_tat(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Returns mean / median TAT for searches *with* and *without* an MB
    contact-plan flag.  Expects `contact_plan_provided`, `tat_calendar_days`
    and `tat_business_days` to exist in df_summary.
    """
    needed = {'contact_plan_provided', 'tat_calendar_days', 'tat_business_days'}
    if not needed.issubset(df_summary.columns):
        return pd.DataFrame()            # silent fail – caller will skip

    out = (
        df_summary
        .groupby('contact_plan_provided')[['tat_calendar_days', 'tat_business_days']]
        .agg(['count', 'mean', 'median'])
        .rename_axis(index=None)
    )
    # flatten the MultiIndex columns for readability
    out.columns = ['_'.join(col) for col in out.columns]
    return out.reset_index()


# ---------------------------------------------------------------------
# 2)  Kaplan-Meier survival curves  (TAT until completion)
# ---------------------------------------------------------------------
def km_curves(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Fits two Kaplan–Meier curves (MB-touched vs non-MB) on calendar-day TAT
    and returns their survival probabilities at each day as a tidy DataFrame.
    """
    if not {'mb_touched', 'tat_calendar_days', 'is_completed'}.issubset(df_summary.columns):
        return pd.DataFrame()

    kmf = KaplanMeierFitter()
    curves = []

    for label, grp in df_summary.groupby('mb_touched'):
        # completion flag is the "event" indicator for lifelines
        kmf.fit(
            durations=grp['tat_calendar_days'],
            event_observed=grp['is_completed'],
            label='MB' if label == 1 else 'Non-MB'
        )
        surv = kmf.survival_function_.reset_index().rename(
            columns={'index': 'days', kmf.survival_function_.columns[0]: 'survival_prob'}
        )
        surv['group'] = kmf._label
        curves.append(surv)

    return pd.concat(curves, ignore_index=True)


# ---------------------------------------------------------------------
# 3)  Wrapper that the main script calls
# ---------------------------------------------------------------------
def run_extra_slices(df_summary: pd.DataFrame,
                     df_history: pd.DataFrame) -> dict:
    """
    Execute all "extra slice" generators and hand back a dict whose keys
    can be merged straight into `all_results` in the main pipeline.
    Safely swallows errors so a bad slice never breaks a full run.
    """
    slices = {}

    # --- contact-plan slices ----------------------------------------
    try:
        slices['contact_plan_tat'] = contact_plan_tat(df_summary)
    except Exception as e:
        warnings.warn(f"contact_plan_tat failed: {e}")

    # --- survival curves -------------------------------------------
    try:
        slices['km_curves'] = km_curves(df_summary)
    except Exception as e:
        warnings.warn(f"km_curves failed: {e}")

    # --- queues & SLA buckets (re-use helpers already in your file) --
    try:
        slices['queue_depth_weekly'] = queue_depth_weekly(df_history)
    except Exception as e:
        warnings.warn(f"queue_depth_weekly failed: {e}")

    try:
        slices['open_queue_age'] = open_queue_age(df_history)
    except Exception as e:
        warnings.warn(f"open_queue_age failed: {e}")

    try:
        slices['first_day_sla'] = first_day_sla_rate(df_summary)
    except Exception as e:
        warnings.warn(f"first_day_sla failed: {e}")

    return slices
# ────────────────── end mb_extra_slices helper block ──────────────────

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

# --- Configuration & Constants ---
CSV_FILE_PATH_DEFAULT = r"C:\Users\awrig\Downloads\Fullo3Query.csv"
MB_AGENT_IDENTIFIER = "murphy.brown"  # Agent identifier (case-insensitive)
# --- EXCLUDED_OFFICE = "SAT-HEX"  # Office to exclude for productivity

# ─── add near top, right after MB_AGENT_IDENTIFIER ────────────────────────────
EXCLUDED_USERNAMES = {
    'system', 'talx', 'auto approve',        # bots / integrations
    'murphy.brown', 'postthirdpartyfees', 'lookuporbitakas'
}
# ──────────────────────────────────────────────────────────────────────────────

# Patterns to identify MB work in notes
MB_CONTACT_PATTERN = r"(?i)contacts found from research"        # drop the parenthetical
MB_EMAIL_RECEIVED_PATTERN = r"<div style=\"color:red\">Email received from" # Renamed from MB_EMAIL_PATTERN
MB_INBOUND_PARSE_PATTERN = r"parsed-inbound|auto-apply|doc parsed" # New pattern for V19 inbound flag

CONTACT_TYPE_PATTERNS = {
    'email': r"Email: ([^\s]+@[^\s]+)",
    'phone': r"Phone: ([+\d\-\(\)]+)",
    'fax': r"Fax: ([+\d\-\(\)]+)"
}

# US Federal Holidays for business days calculation
HOLIDAYS = np.array([
    "2024-01-01","2024-01-15","2024-02-19","2024-05-27","2024-06-19",
    "2024-07-04","2024-09-02","2024-10-14","2024-11-11","2024-11-28",
    "2024-12-25","2025-01-01","2025-01-20","2025-02-17","2025-05-26",
    "2025-06-19","2025-07-04","2025-09-01","2025-10-13","2025-11-11",
    "2025-11-27","2025-12-25"
], dtype="datetime64[D]")

# Constants for contact plan analysis
CONTACT_PLAN_PHRASE = 'contacts found from research' # More specific for time saving flag
CONTACT_KEYWORDS = ['phone:', 'email:', 'fax:'] # Keywords to count as contacts

# --- Time saving constants ---
MIN_PER_CONTACT_PLAN = 20 # minutes
MIN_PER_INBOUND_PARSE = 20 # minutes
MIN_PER_OUTBOUND_SEND = 10 # minutes
INBOUND_VERIFIED_PHRASE = 'verified_complete'      # For inbound auto-close

# inbound auto-close IDs
INB_COMMENT_IDS = {188, 97, 181, 161}
INB_RESULT_IDS = {9, 10, 3, 1}
INB_VERIFIED_PHRASE = r"verified[\s_-]*complete"

SIGNIFICANCE_LEVEL = 0.05

# --- Helper Functions ---

###############################################################################
# FINAL-FORM COMPLETION / TAT CALCULATOR
###############################################################################
def add_completion_and_tat(
        df_hist: pd.DataFrame,
        holidays: np.ndarray | None = None,
        status_col: str = "searchstatus",
        ts_col: str = "historydatetime",
        finished_status: str = "REVIEW",
) -> pd.DataFrame:
    """
    Returns a tiny DataFrame keyed by searchid that contains:
      • first_attempt_time      – earliest historydatetime
      • completiondate          – latest historydatetime whose status == finished_status
                                   (falls back to very last historydatetime if none)
      • is_completed            – 1/0 flag
      • tat_calendar_days       – inclusive, calendar-day diff
      • tat_business_days       – inclusive, Mon-Fri minus supplied holidays
    The caller can simply merge it back to the main summary.
    """

    required = {"searchid", ts_col, status_col}
    missing  = required.difference(df_hist.columns)
    if missing:
        raise ValueError(f"History data is missing required columns: {missing}")

    # --- make sure timestamp column is datetime --------------------------------
    df = df_hist.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # ---------------------------------------------------------------------------
    # 1️⃣  core timestamps   <— UPDATED
    # ---------------------------------------------------------------------------
    grp = df.groupby("searchid", as_index=False) # Ensure grp is defined as per user's note

    # ❶ first outbound attempt ⟶ min(historydatetime) where attemptnumber == 1
    first_attempt = (
        df[df['attemptnumber'] == 1]              # keep only first-attempt rows
          .groupby('searchid', as_index=False)[ts_col]
          .min()
          .rename(columns={ts_col: 'first_attempt_time'})
    )

    # ❷ last attempt ⟶ overall max(historydatetime)
    last_attempt = (
        grp[ts_col]                               # `grp` is df.groupby('searchid')
          .max()
          .rename(columns={ts_col: 'last_attempt_time'})
          .reset_index()
    )

    # ❸ completion date ⟶ latest REVIEW row (same as before)
    completion = (
        df[df[status_col].astype(str).str.upper() == finished_status]
          .groupby('searchid', as_index=False)[ts_col]
          .max()
          .rename(columns={ts_col: 'completiondate'})
    )

    # ❹ assemble core table
    core = (
        first_attempt
          .merge(last_attempt, on='searchid', how='outer')
          .merge(completion,    on='searchid', how='left')
    )

    # ------------ fallback: if nothing marked REVIEW, use last attempt ------
    mask_na = core["completiondate"].isna()
    core.loc[mask_na, "completiondate"] = core.loc[mask_na, "last_attempt_time"]

    # ---------------------------------------------------------------------------
    # 2️⃣  completion flag
    # ---------------------------------------------------------------------------
    core["is_completed"] = core["completiondate"].notna().astype(int)

    # ---------------------------------------------------------------------------
    # 3️⃣  Calendar TAT  (inclusive)
    # ---------------------------------------------------------------------------
    # Use dt.normalize() to compare dates only, add 1 for inclusivity, clip negative
    core["tat_calendar_days"] = np.where(
        core["is_completed"] == 1,
        (core["completiondate"].dt.normalize() 
         - core["first_attempt_time"].dt.normalize()).dt.days + 1,
        np.nan
    )
    core.loc[core["tat_calendar_days"] < 0, "tat_calendar_days"] = np.nan   # Ensure non-negative
    core["tat_calendar_days"] = core["tat_calendar_days"].astype('Float64') # Allow NaNs

    # ---------------------------------------------------------------------------
    # 4️⃣  Business-day TAT  (inclusive)
    # ---------------------------------------------------------------------------
    def _busdays(row):
        if row.is_completed != 1 or pd.isna(row.first_attempt_time) or pd.isna(row.completiondate):
            return np.nan
        # np.busday_count is exclusive of the end-day, so +1
        try:
            # --- Make dates timezone-naive before calculation IF they are aware ---
            start_time = row.first_attempt_time
            end_time = row.completiondate

            if start_time.tz is not None:
                start_time = start_time.tz_localize(None)
            if end_time.tz is not None:
                end_time = end_time.tz_localize(None)

            start_date = start_time.date()
            end_date = end_time.date()
            # ----------------------------------------------------
            # Ensure dates are valid before casting
            start_dt64 = np.datetime64(start_date, 'D')
            end_dt64 = np.datetime64(end_date, 'D') + np.timedelta64(1, 'D') # Add day for inclusivity AFTER casting
            return np.busday_count(
                start_dt64,
                end_dt64, # Use the adjusted end_dt64
                holidays=holidays
            )
        except AttributeError: # Handle cases where .date() might fail (e.g., NaT)
            return np.nan
        except Exception as e: # Catch other potential errors during calculation
            print(f"Error calculating business days for row {row.name}: {e}")
            return np.nan

    core["tat_business_days"] = core.apply(_busdays, axis=1).astype("Int64")

    return core[
        ["searchid","first_attempt_time","completiondate","is_completed",
         "tat_calendar_days","tat_business_days"]
    ]
###############################################################################

###############################################################################
# GLOBAL TTV  (first outbound ➜ completed)
###############################################################################
def add_ttv(df_hist: pd.DataFrame,
            outbound_methods=('email', 'fax', 'phone'),
            completion_flag_col='is_completion_status') -> pd.DataFrame:
    """
    Returns a DataFrame keyed by searchid with:
      • ttv_hours  – float hours from 1st outbound attempt to completion
    Requires:
      • historydatetime (datetime)
      • contactmethod   (str)
      • completion_flag_col produced earlier (True where status==REVIEW)
    """
    needed = {'searchid', 'historydatetime', 'contactmethod', completion_flag_col}
    if not needed.issubset(df_hist.columns):
        raise ValueError(f"TTV calc missing cols: {needed - set(df_hist.columns)}")

    df = df_hist.copy()
    # Ensure datetime type and lowercase contact method for reliable matching
    df['historydatetime'] = pd.to_datetime(df['historydatetime'], errors='coerce')
    df['contactmethod'] = df['contactmethod'].astype(str).str.lower() # Ensure lower case
    df['is_outbound'] = df['contactmethod'].isin(outbound_methods)

    # first outbound per search
    first_out = df[df['is_outbound']] \
        .groupby('searchid')['historydatetime'].min() \
        .rename('first_outbound')

    # completion date per search
    # Find the latest timestamp where the completion flag is True
    completion = df[df[completion_flag_col]] \
        .groupby('searchid')['historydatetime'].max() \
        .rename('completiondate') # Use completiondate as name for consistency

    # Merge, calculate TTV, handle cases where completion might be before first outbound
    ttv = pd.merge(first_out, completion, on='searchid', how='inner') # Use inner join to only keep searches with both
    # Calculate raw difference
    time_diff = ttv['completiondate'] - ttv['first_outbound']
    # Calculate hours, ensuring non-negative result
    ttv['ttv_hours'] = np.maximum(0, time_diff.dt.total_seconds() / 3600)

    return ttv[['ttv_hours']].reset_index()
###############################################################################

def save_results(df: pd.DataFrame, filename: str) -> None:
    """Saves a DataFrame to a CSV file following standard naming and location conventions."""
    if df is None or df.empty:
        print(f"Skipping CSV save for '{filename}' as the DataFrame is empty or None.")
        return

    current_time = datetime.now().strftime("%Y%m%d")
    try:
        # Get the directory containing the script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the script name without the .py extension
        script_name = os.path.splitext(os.path.basename(__file__))[0]
    except NameError:
        # Fallback if __file__ is not defined (e.g., interactive session)
        print("Warning: __file__ not defined. Using current working directory and default script name 'script'.")
        script_dir = os.getcwd()
        script_name = "script" # Fallback script name

    # Construct the full output path according to the standard
    output_dir = os.path.join(script_dir, "Output", script_name)
    os.makedirs(output_dir, exist_ok=True) # Create the directory structure if it doesn't exist
    output_path = os.path.join(output_dir, f"{filename}_{current_time}.csv")

    try:
        # Save the DataFrame to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved CSV: {output_path}")
    except Exception as e:
        # Print detailed error information if saving fails
        print(f"Error saving CSV file to {output_path}: {e}")
        traceback.print_exc()


def load_history_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load verification history data from CSV file with optimizations.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with history data
    """
    print(f"Loading history from CSV: {file_path}")

    # --- Define desired columns and target dtypes EARLIER --- 
    date_cols = [
        'historydatetime', 'completiondate', 'postdate', 'resultsentdate',
        'last_update', 'last_attempt', 'qadate', 'first_attempt' # Added createddate
    ]
    # Using nullable types where appropriate
    # We will pass basic types to read_csv and handle dates/complex conversions later
    read_csv_dtypes = {
        'searchid': 'Int64',
        'historyid': 'Int64',
        'resultid': 'Int16',
        'userid': 'string',
        'username': 'string',
        'agentname': 'string',
        'note': 'string',
        'searchstatus': 'string',
        'searchtype': 'string',
        'contactmethod': 'string',
        'office': 'string', # <-- Added office column
        'commentid': 'Int16',
        'attemptnumber': 'Int16',
        'attempt_result': 'string',
        # We will handle date columns after load for robustness
    }
    all_desired_cols = list(set(date_cols + list(read_csv_dtypes.keys())))
    # --- End Dtype Definition ---

    try:
        # 1. Load data using specified dtypes where possible.
        #    Dates will be loaded as object/string initially and parsed later.
        #    Pass low_memory=False and potentially engine='pyarrow' if installed.
        print("Attempting optimized load with specified dtypes...")
        df = pd.read_csv(
            file_path,
            dtype=read_csv_dtypes, # Apply basic dtypes
            parse_dates=False,   # Parse dates manually later
            low_memory=False,
            # on_bad_lines='warn',
            # engine='pyarrow' # Optional optimization
        )
        print(f"Initial load complete. Found {len(df)} rows and columns: {list(df.columns)}")

        # 2. Standardize column names to lowercase
        original_columns = list(df.columns)
        df.columns = df.columns.str.lower().str.strip()
        standardized_columns = list(df.columns)
        if original_columns != standardized_columns:
            print(f"Standardized columns to: {standardized_columns}")
        else:
            print("Column names already seem standardized.")
        
        # Standardize potential office name variations (e.g., 'officename' or 'agentoffice' to 'office')
        if 'officename' in df.columns and 'office' not in df.columns:
            df = df.rename(columns={'officename': 'office'})
            print("Renamed 'officename' column to 'office'.")
        elif 'agentoffice' in df.columns and 'office' not in df.columns: # Add this condition
            df = df.rename(columns={'agentoffice': 'office'})
            print("Renamed 'agentoffice' column to 'office'.")

        # --- Normalize contactmethod ---
        if 'contactmethod' in df.columns:
            df['contactmethod'] = df['contactmethod'].fillna('').astype(str).str.lower().str.strip()
            method_map = {
                'voice': 'phone',
                'telephone': 'phone',
                'tel': 'phone',
                'faxsend': 'fax',
                # 'fax': 'fax', # Already maps to fax if lowercase is fax
                # 'email': 'email' # Already maps to email if lowercase is email
            }
            # Apply specific maps, then ensure common ones are also covered
            df['contactmethod'] = df['contactmethod'].replace(method_map)
            # Ensure that 'fax' and 'email' if already correct, remain so, and others are mapped or become empty
            allowed_methods = ['phone', 'fax', 'email']
            df['contactmethod'] = df['contactmethod'].apply(lambda x: x if x in allowed_methods else '')
            print("Normalized 'contactmethod' column.")
        # --- End Normalize contactmethod ---

        # 3. Identify available vs missing columns (among ALL desired, including dates)
        missing_desired_cols = [col for col in all_desired_cols if col not in df.columns]

        if missing_desired_cols:
            print(f"Warning: The following desired columns were NOT found in the CSV: {missing_desired_cols}")
        # print(f"Columns available for processing: {available_cols}") # available_cols was defined here previously

        # Keep only the available desired columns
        # df = df[available_cols].copy()      # ← keep all raw columns

        # 4. Apply remaining type conversions (especially for columns not covered by read_csv dtypes)
        #    This section is reduced as read_csv handled basic types.
        print("Verifying data types and parsing dates...")
        for col, dtype in read_csv_dtypes.items():
            if col in df.columns:
                # Verify if the dtype applied correctly, handle if necessary
                # (e.g., if read_csv failed silently for a column)
                if str(df[col].dtype) != dtype.lower(): # Check basic string representation
                    try:
                        print(f"Warning: Column '{col}' dtype is {df[col].dtype}, expected {dtype}. Attempting conversion...")
                        if dtype.startswith('Int'):
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                        else:
                            df[col] = df[col].astype(dtype)
                    except Exception as e:
                         print(f"Warning: Could not convert column '{col}' to dtype {dtype} after load. Error: {e}. Keeping {df[col].dtype}.")

        # 5. Parse date columns robustly
        print("Parsing date columns...")
        for col in date_cols:
             if col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                         print(f"Warning: Could not parse date column '{col}'. Error: {e}. Keeping original type.")
                # else: # No need to print if already datetime
                    # print(f"Column '{col}' already parsed as datetime.")


        print(f"Finished processing columns. Final DataFrame shape: {df.shape}. Columns: {list(df.columns)}")

        # 6. Validate essential columns
        required_columns_for_analysis = ['searchid'] # Minimal requirement for grouping
        missing_essential = [col for col in required_columns_for_analysis if col not in df.columns]
        if missing_essential:
             # This is critical, raise an error
             raise ValueError(f"Essential columns missing after loading and processing: {missing_essential}. Cannot proceed.")

        # Check historydatetime specifically
        if 'historydatetime' not in df.columns:
            print("Warning: 'historydatetime' column missing. Time-based metrics will be affected.")
        elif not pd.api.types.is_datetime64_any_dtype(df['historydatetime']):
            print("Warning: 'historydatetime' column exists but failed datetime conversion. Time-based metrics will be affected.")


        return df

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        raise
    except Exception as e:
        # Catch other potential pandas errors during initial load (e.g., EmptyDataError)
        print(f"Error during CSV loading or initial processing: {e}")
        traceback.print_exc()
        raise


def calculate_business_days(start_date, end_date, holidays=None):
    """
    Calculate business days between two dates (inclusive).
    
    Args:
        start_date: Start date
        end_date: End date
        holidays: Array of holiday dates to exclude
    
    Returns:
        Number of business days or np.nan if invalid dates
    """
    if pd.isna(start_date) or pd.isna(end_date) or end_date < start_date:
        return np.nan
    
    try:
        start_np = np.datetime64(start_date.date())
        end_np = np.datetime64(end_date.date())
        # +1 day to make the end date inclusive
        return np.busday_count(start_np, end_np + np.timedelta64(1, 'D'), holidays=holidays)
    except AttributeError:
        # Handle cases where start_date or end_date might not be datetime objects
        return np.nan
    except Exception as e:
        print(f"Error in calculate_business_days ({start_date}, {end_date}): {e}")
        return np.nan


def cohens_d(x, y):
    """
    Calculate Cohen's d effect size for two independent samples.
    
    Args:
        x: First sample
        y: Second sample
    
    Returns:
        Cohen's d value or np.nan if invalid
    """
    x = x.dropna()
    y = y.dropna()
    nx = len(x)
    ny = len(y)
    
    if nx < 2 or ny < 2:
        return np.nan
    
    dof = nx + ny - 2
    if dof <= 0:
        return np.nan
    
    # Use np.maximum to avoid negative variance due to floating point issues
    var_x = np.maximum(0, np.var(x, ddof=1))
    var_y = np.maximum(0, np.var(y, ddof=1))
    
    # Ensure pooled variance is non-negative
    pooled_var = np.maximum(0, ((nx - 1) * var_x + (ny - 1) * var_y) / dof)
    s_pooled = np.sqrt(pooled_var)
    mean_diff = np.mean(x) - np.mean(y)
    
    if s_pooled == 0:
        # Return NaN if std dev is 0 but means differ, 0 if means are also 0
        return 0.0 if np.isclose(mean_diff, 0) else np.nan
    
    # Handle potential division by zero if s_pooled is extremely small
    if np.isclose(s_pooled, 0):
        # Return NaN if std dev is near 0 but means differ, 0 if means are also near 0
        return 0.0 if np.isclose(mean_diff, 0) else np.nan
    
    return mean_diff / s_pooled


def count_channel_interactions(df_hist: pd.DataFrame, channel: str) -> pd.Series:
    """Counts inbound interactions for a specific channel, assuming resultid 0."""
    # Ensure contactmethod is lowercase for comparison
    mask = (df_hist['contactmethod'].astype(str).str.lower() == channel.lower()) & \
           (pd.to_numeric(df_hist['resultid'], errors='coerce') == 0)  # assume inbound rows have resultid 0 / adjust if needed
    # Group by searchid, count the matches, and rename the series
    return df_hist[mask].groupby('searchid').size().rename(f'{channel}_interaction_count')


# --- Core Metric Calculation Functions ---

def calculate_attempt_count_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate attempt count metrics per search ID
    ---------------------------------------------
    NEW LOGIC =  SQL-parity  →  search_attempts, applicant_contact_count,
                               maxminusapplicant
    """
    print("Calculating attempt count metrics (SQL-parity).")

    req = ['searchid', 'attemptnumber', 'attempt_result']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for attempt count metrics: {missing}")

    # Ensure attemptnumber is numeric and create a helper for case-insensitive comparison
    df['_attempt_result_lower'] = df['attempt_result'].astype(str).str.lower()
    df['attemptnumber'] = pd.to_numeric(df['attemptnumber'], errors='coerce')

    # keep only rows whose username is NOT in the exclusion list
    df_filt = df[~df['username'].str.lower().isin(EXCLUDED_USERNAMES)].copy()

    grp = df_filt.groupby('searchid')

    # 1️⃣  search_attempts (max_attempt renamed for clarity)
    search_attempts = grp['attemptnumber'].max().rename('search_attempts')

    # 2️⃣  applicant_contact_count (distinct attempts)
    applicant_contact_count = (
      grp.apply(lambda g: g.loc[g['_attempt_result_lower'] == 'applicant contact', 'attemptnumber'].nunique(), include_groups=False)
      .rename('applicant_contact_count')
    )

    out = pd.concat([search_attempts, applicant_contact_count], axis=1)
    
    # FillNa and ensure correct dtype for applicant_contact_count
    out['applicant_contact_count'] = out['applicant_contact_count'].fillna(0).astype('Int64')
    # Ensure search_attempts is also Int64, handling potential NaNs from max() on empty/all-NaN groups
    out['search_attempts'] = out['search_attempts'].astype('Float64').astype('Int64')


    # 3️⃣  maxminusapplicant (never negative)
    # Ensure both operands are numeric before subtraction
    out['maxminusapplicant'] = (
        pd.to_numeric(out['search_attempts'], errors='coerce').fillna(0) -
        pd.to_numeric(out['applicant_contact_count'], errors='coerce').fillna(0)
    ).clip(lower=0).astype('Int64')
    
    # Drop the helper column
    df.drop(columns=['_attempt_result_lower'], inplace=True, errors='ignore')

    print("Attempt count metrics calculation complete.")
    return out.reset_index()


# --- MB Capability Detection Functions ---

def identify_mb_contact_work(df_history):
    """
    Identify searches where Murphy Brown performed contact research work.
    
    Args:
        df_history: History DataFrame
    
    Returns:
        DataFrame with searchid and flags for contact research presence
    """
    print("Analyzing Murphy Brown contact research patterns...")
    
    # Initialize result DataFrame with all unique search IDs
    search_ids = df_history['searchid'].unique()
    contact_df = pd.DataFrame({'searchid': search_ids})
    contact_df['mb_contact_research'] = 0  # Default: no contact research
    agg_contacts = pd.DataFrame() # Initialize agg_contacts
    
    # Check notes for contact research pattern
    if 'note' in df_history.columns:
        # Find searches with MB contact research
        mask = df_history['note'].astype(str).str.contains(MB_CONTACT_PATTERN, na=False, case=False)
        research_search_ids = df_history.loc[mask, 'searchid'].unique()
        
        # Update flag for searches with contact research
        contact_df.loc[contact_df['searchid'].isin(research_search_ids), 'mb_contact_research'] = 1
        
        # Count number of contacts found by type
        def extract_contacts(row):
            if not isinstance(row['note'], str):
                return pd.Series({
                    'email_count': 0, 
                    'phone_count': 0, 
                    'fax_count': 0, 
                    'total_contacts': 0
                })
            
            note = row['note']
            email_count = len(re.findall(CONTACT_TYPE_PATTERNS['email'], note))
            phone_count = len(re.findall(CONTACT_TYPE_PATTERNS['phone'], note))
            fax_count = len(re.findall(CONTACT_TYPE_PATTERNS['fax'], note))
            
            return pd.Series({
                'email_count': email_count, 
                'phone_count': phone_count, 
                'fax_count': fax_count, 
                'total_contacts': email_count + phone_count + fax_count
            })
        
        # Apply contact extraction to all MB contact research rows
        contact_details = df_history[mask].apply(extract_contacts, axis=1)
        
        # Aggregate contact counts by search ID
        if not contact_details.empty:
            agg_contacts = contact_details.groupby(df_history[mask]['searchid']).max()
        # contact_df = pd.merge(contact_df, agg_contacts, on='searchid', how='left') # This line is removed
        
        # Fill NaN values with 0 # This loop is removed
        # for col in ['email_count', 'phone_count', 'fax_count', 'total_contacts']:
        #     if col in contact_df.columns:
        #         contact_df[col] = contact_df[col].fillna(0).astype('Int64') 
        #     else:
        #         contact_df[col] = 0
    else:
        print("Warning: 'note' column not found in history data. Cannot analyze contact research.")
        # Initialize contact count columns in contact_df with 0 if note column is missing
        contact_df['email_count'] = 0
        contact_df['phone_count'] = 0
        contact_df['fax_count'] = 0
        contact_df['total_contacts'] = 0
        # agg_contacts remains an empty DataFrame as initialized
    
    # if we found any contact details, merge them back in
    if not agg_contacts.empty:
        contact_df = contact_df.merge(
            agg_contacts.reset_index(), # Ensure searchid is a column for merging
            on='searchid',
            how='left'
        )
    # fill any missing ones with zero
    for col in ['email_count','phone_count','fax_count','total_contacts']:
        if col in contact_df.columns:
            contact_df[col] = contact_df[col].fillna(0).astype('Int64')
        else: # If merge didn't add the column (e.g., agg_contacts was empty or didn't have it)
            contact_df[col] = 0
            contact_df[col] = contact_df[col].astype('Int64')

    print(f"Found {contact_df['mb_contact_research'].sum()} searches with MB contact research.")
    return contact_df, agg_contacts


def identify_mb_email_interaction(df_history):
    """
    Identify searches where Murphy Brown handled email interactions.
    
    Args:
        df_history: History DataFrame
    
    Returns:
        DataFrame with searchid and flags for email handling
    """
    print("Analyzing Murphy Brown email interaction patterns...")
    
    # Initialize result DataFrame with all unique search IDs
    search_ids = df_history['searchid'].unique()
    email_df = pd.DataFrame({'searchid': search_ids})
    email_df['mb_email_handling'] = 0  # Default: no email handling
    
    # Check notes for email handling pattern
    if 'note' in df_history.columns:
        # Find searches with MB email handling
        mask = df_history['note'].astype(str).str.contains(MB_EMAIL_RECEIVED_PATTERN, na=False)
        email_search_ids = df_history.loc[mask, 'searchid'].unique()
        
        # Update flag for searches with email handling
        email_df.loc[email_df['searchid'].isin(email_search_ids), 'mb_email_handling'] = 1
        
        # Count number of email interactions per search
        email_counts = df_history[mask].groupby('searchid').size().reset_index(name='email_interaction_count')
        
        # Merge with main email DataFrame
        email_df = pd.merge(email_df, email_counts, on='searchid', how='left')
        email_df['email_interaction_count'] = email_df['email_interaction_count'].fillna(0).astype('Int64') # Use nullable integer
    else:
        print("Warning: 'note' column not found in history data. Cannot analyze email handling.")
        email_df['email_interaction_count'] = 0
    
    # Extract contactmethod for further analysis
    if 'contactmethod' in df_history.columns:
        email_method_counts = df_history[
            df_history['contactmethod'].astype(str).str.lower() == 'email'
        ].groupby('searchid').size().reset_index(name='email_method_count')
        
        # Merge with main email DataFrame
        email_df = pd.merge(email_df, email_method_counts, on='searchid', how='left')
        email_df['email_method_count'] = email_df['email_method_count'].fillna(0).astype('Int64') # Use nullable integer
    else:
        print("Warning: 'contactmethod' column not found. Cannot analyze contact methods.")
        email_df['email_method_count'] = 0
    
    print(f"Found {email_df['mb_email_handling'].sum()} searches with MB email handling.")
    return email_df


# --- Impact Analysis Functions ---

def analyze_mb_capability_impact(df_summary, extended_metrics=True):
    """
    Analyze the impact of different MB capabilities on verification metrics.
    
    Uses the pre-aggregated df_summary.
    
    Args:
        df_summary: Main summary DataFrame with MB capability flags
        extended_metrics: Whether to calculate additional detailed metrics
        
    Returns:
        Dictionary with comparative analysis between capability groups
    """
    print("Analyzing impact by MB capability group...")
    
    # Required base columns
    required_cols = ['searchid', 'mb_touched', 'mb_contact_research', 'mb_email_handling']
    if not all(col in df_summary.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_summary.columns]
        print(f"Error: Missing required columns for capability analysis: {missing}")
        return None
    
    # Create capability groups
    df_summary['capability_group'] = 'No MB'
    # Contact research only
    mask_contact = (df_summary['mb_contact_research'] == 1) & (df_summary['mb_email_handling'] == 0)
    df_summary.loc[mask_contact, 'capability_group'] = 'Contact Research Only'
    # Email handling only
    mask_email = (df_summary['mb_contact_research'] == 0) & (df_summary['mb_email_handling'] == 1)
    df_summary.loc[mask_email, 'capability_group'] = 'Email Handling Only'
    # Both capabilities
    mask_both = (df_summary['mb_contact_research'] == 1) & (df_summary['mb_email_handling'] == 1)
    df_summary.loc[mask_both, 'capability_group'] = 'Both Capabilities'
    
    # Show distribution of capability groups
    group_counts = df_summary['capability_group'].value_counts().reset_index()
    group_counts.columns = ['Capability Group', 'Count']
    print("\nDistribution of MB Capability Groups:")
    print(group_counts.to_markdown(index=False))
    
    # Define metrics to analyze
    base_metrics = [
        'tat_calendar_days', 'tat_business_days', 'search_attempts',
        'applicant_contact_count', 'maxminusapplicant'
    ]
    
    # Add extended metrics if available
    if extended_metrics:
        extended_metrics_list = [
            'total_contacts', 'email_count', 'phone_count', 'fax_count',
            'email_interaction_count', 'email_method_count'
        ]
        metrics_to_analyze = base_metrics + [m for m in extended_metrics_list if m in df_summary.columns]
    else:
        metrics_to_analyze = base_metrics
    
    # Filter to metrics that exist in the DataFrame
    metrics_to_analyze = [m for m in metrics_to_analyze if m in df_summary.columns]
    
    # --- Fix: Convert metrics to numeric type, coercing errors to NaN ---
    for metric in metrics_to_analyze:
        df_summary[metric] = pd.to_numeric(df_summary[metric], errors='coerce')
    # --- End Fix ---

    # Group by capability_group and calculate metrics
    grouped_metrics = df_summary.groupby('capability_group')[metrics_to_analyze].agg([
        'count', 'mean', 'median', 'std'
    ]).reset_index()
    
    # Perform ANOVA for each metric to test significance of differences
    anova_results = {}
    for metric in metrics_to_analyze:
        valid_groups = []
        for group_name_iter in df_summary['capability_group'].unique():
            group_data = df_summary[df_summary['capability_group'] == group_name_iter][metric].dropna()
            if len(group_data) > 1 and pd.Series(group_data).var(ddof=1) > 0:
                valid_groups.append(group_data)
        
        current_note = "" # Initialize note
        if len(valid_groups) >= 2:
            try:
                f_stat, p_val = stats.f_oneway(*valid_groups)
                significant = p_val < SIGNIFICANCE_LEVEL
            except Exception as e:
                print(f"Error performing ANOVA for {metric}: {e}")
                f_stat, p_val, significant, current_note = np.nan, np.nan, np.nan, f"ANOVA Error: {e}"
        else:
            f_stat, p_val, significant = np.nan, np.nan, np.nan
            current_note = "Skipped - constant input or <2 valid groups with variance"
            if VERBOSITY_LEVEL >= 2: # Print warning only if verbose enough
                print(f"Insufficient data or variance for ANOVA on {metric} (Valid Groups: {len(valid_groups)})")


        anova_results[metric] = {
            'F_statistic': f_stat,
            'p_value': p_val,
            'significant': significant, # Will be np.nan if test not run/errored or if skipped
            'note': current_note
        }
    
    # Create ANOVA summary DataFrame
    anova_df = pd.DataFrame([
        {
            'Metric': metric,
            'F_Statistic': results['F_statistic'],
            'P_Value': results['p_value'],
            'Significant': results['significant'],
            'Note': results.get('note', '') # Add note column
        }
        for metric, results in anova_results.items()
    ])
    
    if not anova_df.empty:
        print("\nANOVA Results for MB Capability Groups:")
        print(anova_df.to_markdown(index=False))
    
    return {
        'grouped_metrics': grouped_metrics,
        'anova_results': anova_df,
        'group_counts': group_counts
    }


def calculate_mb_autonomy_metrics(df_history, df_summary):
    """
    Calculate autonomy-related metrics for Murphy Brown using vectorized operations.
    """
    print("\n--- Calculating MB Autonomy Metrics ---")

    # Ensure necessary columns exist in df_history
    required_hist_cols = ['searchid', 'historydatetime', 'agentname', 'username']
    if not all(col in df_history.columns for col in required_hist_cols):
        missing = [col for col in required_hist_cols if col not in df_history.columns]
        print(f"Warning: Cannot calculate autonomy metrics – missing columns in df_history: {missing}")
        # Return a structure indicating failure or incomplete results
        return {
            'autonomy_df': pd.DataFrame({'searchid': df_summary['searchid'].unique()}),
            'summary_metrics': {}
        }

    # --- Vectorized Approach --- #

    # 1. Pre-calculate MB row and Human Touch flags ON df_history itself
    mb_pattern = MB_AGENT_IDENTIFIER.lower()
    # Ensure agentname and username are strings before using .str methods
    df_history['agentname_lower'] = df_history['agentname'].fillna('').astype(str).str.lower()
    df_history['username_lower'] = df_history['username'].fillna('').astype(str).str.lower()

    df_history['is_mb_row'] = (
        df_history['agentname_lower'].str.contains(mb_pattern, na=False) |
        df_history['username_lower'].str.contains(mb_pattern, na=False)
    )
    # Human touch is the inverse of MB row
    # df_history['is_human_touch'] = ~df_history['is_mb_row']
    df_history['is_human_touch'] = (
        ~df_history['username_lower'].isin(EXCLUDED_USERNAMES)
    )

    # Clean up temporary columns
    df_history.drop(columns=['agentname_lower', 'username_lower'], inplace=True)

    # 2. Sort data for shift operations
    # Ensure historydatetime is datetime before sorting
    if not pd.api.types.is_datetime64_any_dtype(df_history['historydatetime']):
        df_history['historydatetime'] = pd.to_datetime(df_history['historydatetime'], errors='coerce')
    df_history_sorted = df_history.sort_values(['searchid', 'historydatetime']).copy()

    # 3. Calculate human touch count per searchid
    cols_dedup = ['searchid', 'attemptnumber']
    human_touches = (
          df_history_sorted[df_history_sorted['is_human_touch']]
            .drop_duplicates(subset=cols_dedup)        # one touch per attempt#
            .groupby('searchid')
            .size()
            .rename('human_touch_count')
    )

    # 4. Identify Rework: Human touch immediately after an MB touch
    # Shift the 'is_human_touch' status within each search group
    df_history_sorted['prev_is_human'] = df_history_sorted.groupby('searchid')['is_human_touch'].shift(1)
    # Rework occurs where current is human AND previous was NOT human (i.e., was MB)
    rework_mask = df_history_sorted['is_human_touch'] & (df_history_sorted['prev_is_human'] == False)
    rework_search_ids = df_history_sorted.loc[rework_mask, 'searchid'].unique()

    # 5. Identify Fallback: Last touch in a search is human
    last_touches = df_history_sorted.groupby('searchid').tail(1)
    fallback_search_ids = last_touches.loc[last_touches['is_human_touch'], 'searchid'].unique()

    # 6. Aggregate results into a DataFrame aligned with df_summary
    autonomy_df = pd.DataFrame({'searchid': df_summary['searchid'].unique()})
    autonomy_df = pd.merge(autonomy_df, human_touches, on='searchid', how='left')
    autonomy_df['human_touch_count'] = autonomy_df['human_touch_count'].fillna(0).astype('Int64') # Use nullable integer
    autonomy_df['fully_autonomous'] = (autonomy_df['human_touch_count'] == 0).astype(int) 
    autonomy_df['has_rework'] = autonomy_df['searchid'].isin(rework_search_ids).astype(int)
    autonomy_df['had_fallback'] = autonomy_df['searchid'].isin(fallback_search_ids).astype(int)

    # --- End Vectorized Approach --- #

    # 7. Summary Metrics Calculation
    summary_metrics = {}
    mb_ids = df_summary.loc[df_summary['mb_touched'] == 1, 'searchid'].unique()

    # Calculate metrics only if the corresponding column exists
    if 'human_touch_count' in autonomy_df.columns:
        summary_metrics['avg_human_touches'] = autonomy_df['human_touch_count'].mean()
        print(f"Average Human Touches (overall): {summary_metrics['avg_human_touches']:.2f}")
    else:
        summary_metrics['avg_human_touches'] = np.nan

    if 'fully_autonomous' in autonomy_df.columns:
        summary_metrics['autonomy_rate'] = autonomy_df['fully_autonomous'].mean() * 100
        print(f"End-to-End Autonomy Rate (overall): {summary_metrics['autonomy_rate']:.2f}%")
    else:
        summary_metrics['autonomy_rate'] = np.nan

    # Calculate rates based *only* on searches MB touched
    if not autonomy_df.empty and 'searchid' in autonomy_df.columns:
        autonomy_mb_subset = autonomy_df[autonomy_df['searchid'].isin(mb_ids)]

        if not autonomy_mb_subset.empty:
            if 'has_rework' in autonomy_mb_subset.columns:
                summary_metrics['rework_rate'] = autonomy_mb_subset['has_rework'].mean() * 100
                print(f"MB Rework Rate: {summary_metrics['rework_rate']:.2f}% ({autonomy_mb_subset['has_rework'].sum()}/{len(autonomy_mb_subset)})" )
            else:
                summary_metrics['rework_rate'] = np.nan

            if 'had_fallback' in autonomy_mb_subset.columns:
                summary_metrics['fallback_rate'] = autonomy_mb_subset['had_fallback'].mean() * 100
                print(f"MB Fallback Transfer Rate: {summary_metrics['fallback_rate']:.2f}% ({autonomy_mb_subset['had_fallback'].sum()}/{len(autonomy_mb_subset)})" )
            else:
                summary_metrics['fallback_rate'] = np.nan

        else: # No MB searches in the autonomy subset
             print("No MB-touched searches found in autonomy data to calculate rework/fallback rates.")
             summary_metrics['rework_rate'] = 0.0
             summary_metrics['fallback_rate'] = 0.0

    else: # Autonomy df is empty or missing searchid
        print("Warning: Cannot calculate MB-specific rates (rework, fallback) due to missing autonomy data.")
        summary_metrics['rework_rate'] = np.nan
        summary_metrics['fallback_rate'] = np.nan

    return {
        'autonomy_df': autonomy_df, # Contains searchid, human_touch_count, fully_autonomous, has_rework, had_fallback
        'summary_metrics': summary_metrics
    }


def calculate_time_efficiency_metrics(df_summary):
    """
    Calculate time efficiency metrics including TAT, queue time, time to verification.
    
    Uses the pre-aggregated df_summary which contains TAT.
    
    Args:
        df_summary: Summary DataFrame
    
    Returns:
        Dictionary with time efficiency results
    """
    print("Calculating time efficiency statistics...")
    
    # Required columns
    required_cols = ['searchid', 'mb_touched', 'tat_calendar_days', 'tat_business_days',
                    'first_attempt_time', 'is_completed']
    
    missing_cols = [col for col in required_cols if col not in df_summary.columns]
    if missing_cols:
        print(f"Warning: Missing columns for time efficiency metrics: {missing_cols}")
        return None
    
    # Filter for completed searches only
    completed_df = df_summary[df_summary['is_completed'] == 1].copy()
    if completed_df.empty:
        print("No completed searches found for time efficiency analysis.")
        return None
    
    # Group by MB touched and calculate metrics
    tat_stats = completed_df.groupby('mb_touched').agg(
        search_count=('searchid', 'count'),
        avg_tat_days=('tat_calendar_days', 'mean'),
        median_tat_days=('tat_calendar_days', 'median'),
        avg_tat_business=('tat_business_days', 'mean'),
        median_tat_business=('tat_business_days', 'median'),
        avg_ttv_hours=('ttv_hours', 'mean'),  # Added TTV
        median_ttv_hours=('ttv_hours', 'median') # Added TTV
    ).reset_index()
    
    # Calculate time differences based on capability groups if available
    if all(col in completed_df.columns for col in ['mb_contact_research', 'mb_email_handling']):
        # Create capability groups
        completed_df['capability_group'] = 'No MB'
        # Contact research only
        mask_contact = (completed_df['mb_contact_research'] == 1) & (completed_df['mb_email_handling'] == 0)
        completed_df.loc[mask_contact, 'capability_group'] = 'Contact Research Only'
        # Email handling only
        mask_email = (completed_df['mb_contact_research'] == 0) & (completed_df['mb_email_handling'] == 1)
        completed_df.loc[mask_email, 'capability_group'] = 'Email Handling Only'
        # Both capabilities
        mask_both = (completed_df['mb_contact_research'] == 1) & (completed_df['mb_email_handling'] == 1)
        completed_df.loc[mask_both, 'capability_group'] = 'Both Capabilities'
        
        # Calculate metrics by capability group
        capability_stats = completed_df.groupby('capability_group').agg(
            search_count=('searchid', 'count'),
            avg_tat_days=('tat_calendar_days', 'mean'),
            median_tat_days=('tat_calendar_days', 'median'),
            avg_tat_business=('tat_business_days', 'mean'),
            median_tat_business=('tat_business_days', 'median'),
            avg_ttv_hours=('ttv_hours', 'mean'),  # Added TTV
            median_ttv_hours=('ttv_hours', 'median') # Added TTV
        ).reset_index()
    else:
        capability_stats = None
    
    # Statistical test for TAT differences
    # ensure dtype float for t-test BEFORE creating subsets
    # Convert relevant columns to numeric, coercing errors
    completed_df['tat_calendar_days'] = pd.to_numeric(completed_df['tat_calendar_days'], errors='coerce')
    completed_df['tat_business_days'] = pd.to_numeric(completed_df['tat_business_days'], errors='coerce')
    completed_df['ttv_hours'] = pd.to_numeric(completed_df['ttv_hours'], errors='coerce') # Added TTV

    mb_completed = completed_df[completed_df['mb_touched'] == 1]
    non_mb_completed = completed_df[completed_df['mb_touched'] == 0]
    
    stat_results = {}
    for metric in ['tat_calendar_days', 'tat_business_days', 'ttv_hours']: # Added TTV to loop
        if metric not in completed_df.columns:
            continue
            
        # Drop NA values *after* conversion and splitting
        data_mb = mb_completed[metric].dropna()
        data_non_mb = non_mb_completed[metric].dropna()
        
        if len(data_mb) >= 2 and len(data_non_mb) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(data_mb, data_non_mb, equal_var=False)
                significant = p_value < 0.05
                
                stat_results[metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significant
                }
                
                print(f"T-test for {metric}: t={t_stat:.4f}, p={p_value:.4f} {'(SIGNIFICANT)' if significant else ''}")
            except Exception as e:
                print(f"Error during t-test for {metric}: {e}")
        else:
            print(f"Insufficient data for t-test on {metric}")
    
    return {
        'tat_stats': tat_stats,
        'capability_stats': capability_stats,
        'statistical_tests': stat_results
    }


def calculate_contact_efficiency_metrics(df_summary):
    """
    Calculate contact efficiency metrics including outbound contact breadth, channel yield.
    
    Uses the pre-aggregated df_summary.
    
    Args:
        df_summary: Summary DataFrame
    
    Returns:
        Dictionary with contact efficiency results
    """
    print("Calculating contact efficiency statistics...")
    
    # Check for required columns
    required_cols = ['searchid', 'mb_touched', 'search_attempts', 'is_completed']
    contact_cols = ['email_count', 'phone_count', 'fax_count', 'total_contacts']
    
    missing_base = [col for col in required_cols if col not in df_summary.columns]
    if missing_base:
        print(f"Warning: Missing base columns for contact metrics: {missing_base}")
        return None
    
    missing_contact = [col for col in contact_cols if col not in df_summary.columns]
    if missing_contact:
        print(f"Warning: Missing contact detail columns: {missing_contact}")
        contact_metrics_available = False
    else:
        contact_metrics_available = True
    
    # Filter for completed searches
    completed_df = df_summary[df_summary['is_completed'] == 1].copy()
    if completed_df.empty:
        print("No completed searches found for contact efficiency analysis.")
        return None
    
    # 1. Calculate outbound attempts per verification (search_attempts)
    attempts_stats = completed_df.groupby('mb_touched').agg(
        search_count=('searchid', 'count'),
        avg_attempts=('search_attempts', 'mean'),
        median_attempts=('search_attempts', 'median')
    ).reset_index()
    
    # 2. Calculate contact breadth metrics if available
    if contact_metrics_available:
        contact_stats = completed_df.groupby('mb_touched').agg(
            avg_total_contacts=('total_contacts', 'mean'),
            avg_email_contacts=('email_count', 'mean'),
            avg_phone_contacts=('phone_count', 'mean'),
            avg_fax_contacts=('fax_count', 'mean')
        ).reset_index()
        
        # Calculate outbound contact breadth (avg distinct contacts per search)
        contact_breadth = contact_stats['avg_total_contacts']
        if len(contact_breadth) >= 2:
            print(f"Average outbound contact breadth: {contact_breadth.iloc[1]:.2f} (MB) vs {contact_breadth.iloc[0]:.2f} (non-MB)")
    else:
        contact_stats = None
    
    # 3. Calculate yield metrics if interaction data is available
    yield_stats = {} # Initialize dictionary for yields per channel
    stat_results_yield = {} # Initialize dict for yield t-tests

    for ch in ['email', 'fax', 'phone']:
        int_col = f'{ch}_interaction_count'
        out_col = f'{ch}_method_count'

        # --- Sanity check for outbound attempts ---
        if out_col in completed_df.columns:
            n_out = completed_df[out_col].sum()
            if n_out == 0:
                print(f"[WARN] No outbound {ch} attempts found (sum of {out_col} is 0) – {ch} yield calculation will be skipped or result in zero/NaNs.")
        else:
            print(f"[WARN] Column {out_col} not found for {ch} outbound attempts sanity check.")
        # --- End sanity check ---

        # Check if necessary columns exist for the channel
        if int_col in completed_df.columns and out_col in completed_df.columns:
            # Calculate yield for the channel
            # --- fixed: use pandas Series.where instead of np.where ---
            # Ensure counts are numeric before division
            numerator = pd.to_numeric(completed_df[int_col], errors='coerce')
            denominator = pd.to_numeric(completed_df[out_col], errors='coerce') # Ensure denominator is numeric too

            # Calculate the ratio safely, handling potential division by zero or NaNs
            # Start with NaN Series to avoid division by zero error if denominator is 0
            ratio = pd.Series(np.nan, index=completed_df.index) 
            # Calculate ratio only where denominator is > 0
            valid_denom_mask = denominator > 0
            ratio.loc[valid_denom_mask] = numerator.loc[valid_denom_mask] / denominator.loc[valid_denom_mask]

            # Use .where to replace NaN or inf resulting from division by zero, or where original denominator was 0, with 0
            # Then fill any remaining NaNs (e.g., from coercion errors in numerator) with 0
            completed_df[f'{ch}_yield'] = ratio.where(valid_denom_mask, 0).fillna(0).astype(float)
            # -----------------------------------------------------------

            # Calculate aggregated stats for the channel
            ys = completed_df.groupby('mb_touched').agg(
                attempts=(out_col, 'sum'),
                responses=(int_col, 'sum'),
                avg_yield=(f'{ch}_yield', 'mean')
            ).reset_index()
            yield_stats[ch] = ys # Store DataFrame in the dictionary

            # Print overall channel yield
            for row in ys.itertuples():
                if row.attempts > 0:
                    mb_status = "MB" if row.mb_touched == 1 else "non-MB"
                    overall_yield = row.responses / row.attempts * 100
                    print(f"{ch.capitalize()} channel yield for {mb_status}: {overall_yield:.2f}% ({row.responses}/{row.attempts})")
                else:
                    # mb_status = "MB" if row.mb_touched == 1 else "non-MB" # Redundant if attempts is 0
                    print(f"{ch.capitalize()} channel yield for {('MB' if row.mb_touched == 1 else 'non-MB')}: n/a (no attempts)")
                    continue # Skip to the next iteration of ys.itertuples() if attempts is 0

            # Perform T-test for yield difference
            mb_completed_channel = completed_df[completed_df['mb_touched'] == 1]
            non_mb_completed_channel = completed_df[completed_df['mb_touched'] == 0]
            yield_metric_name = f'{ch}_yield'
            data_mb = mb_completed_channel[yield_metric_name].dropna()
            data_non_mb = non_mb_completed_channel[yield_metric_name].dropna()

            if len(data_mb) >= 2 and len(data_non_mb) >= 2:
                try:
                    t_stat, p_value = stats.ttest_ind(data_mb, data_non_mb, equal_var=False, nan_policy='omit')
                    significant = p_value < SIGNIFICANCE_LEVEL # Use defined constant

                    stat_results_yield[yield_metric_name] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': significant
                    }

                    print(f"T-test for {yield_metric_name}: t={t_stat:.4f}, p={p_value:.4f} {'(SIGNIFICANT)' if significant else ''}")
                except Exception as e:
                    print(f"Error during t-test for {yield_metric_name}: {e}")
                    stat_results_yield[yield_metric_name] = {'t_statistic': np.nan, 'p_value': np.nan, 'significant': np.nan}
            else:
                print(f"Insufficient data for t-test on {yield_metric_name}")
                stat_results_yield[yield_metric_name] = {'t_statistic': np.nan, 'p_value': np.nan, 'significant': np.nan}
        else:
            print(f"Warning: Cannot calculate {ch} yield metrics - missing required columns ({int_col} or {out_col})")
            yield_stats[ch] = None # Indicate channel data was missing

    # 4. Statistical tests for attempts (Keep existing test)
    mb_completed = completed_df[completed_df['mb_touched'] == 1]
    non_mb_completed = completed_df[completed_df['mb_touched'] == 0]
    
    stat_results_attempts = {} # Rename to avoid clash
    for metric in ['search_attempts']:
        if metric not in completed_df.columns:
            continue
            
        data_mb = mb_completed[metric].dropna()
        data_non_mb = non_mb_completed[metric].dropna()
        
        if len(data_mb) >= 2 and len(data_non_mb) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(data_mb, data_non_mb, equal_var=False)
                significant = p_value < SIGNIFICANCE_LEVEL # Use defined constant
                
                stat_results_attempts[metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significant
                }
                
                print(f"T-test for {metric}: t={t_stat:.4f}, p={p_value:.4f} {'(SIGNIFICANT)' if significant else ''}")
            except Exception as e:
                print(f"Error during t-test for {metric}: {e}")
                stat_results_attempts[metric] = {'t_statistic': np.nan, 'p_value': np.nan, 'significant': np.nan}
        else:
             print(f"Insufficient data for t-test on {metric}")
             stat_results_attempts[metric] = {'t_statistic': np.nan, 'p_value': np.nan, 'significant': np.nan}

    # Combine statistical test results
    all_stat_results = {**stat_results_attempts, **stat_results_yield}

    return {
        'attempts_stats': attempts_stats,
        'contact_stats': contact_stats,
        'yield_stats': yield_stats, # Return the dictionary of yield DataFrames
        'statistical_tests': all_stat_results # Return combined stats
    }


def calculate_mb_category_metrics(df_summary):
    """
    Calculate metrics for different categories of MB usage.
    
    Uses the pre-aggregated df_summary.
    
    Args:
        df_summary: Summary DataFrame
    
    Returns:
        Dictionary with category metrics
    """
    print("Calculating metrics by MB capability category...")
    
    # Check for required columns
    if not all(col in df_summary.columns for col in ['mb_contact_research', 'mb_email_handling']):
        print("Error: Missing MB capability columns for category analysis")
        return None
    
    # Create capability categories
    df_summary['mb_category'] = 'No MB'
    df_summary.loc[(df_summary['mb_contact_research'] == 1) & (df_summary['mb_email_handling'] == 0), 'mb_category'] = 'Contact Research Only'
    df_summary.loc[(df_summary['mb_contact_research'] == 0) & (df_summary['mb_email_handling'] == 1), 'mb_category'] = 'Email Handling Only'
    df_summary.loc[(df_summary['mb_contact_research'] == 1) & (df_summary['mb_email_handling'] == 1), 'mb_category'] = 'Both Capabilities'
    
    # Show distribution
    category_counts = df_summary['mb_category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    print("\nMB Capability Category Distribution:")
    print(category_counts.to_markdown(index=False))
    
    # Define metrics to analyze
    metrics = [
        'tat_calendar_days', 'tat_business_days', 'search_attempts',
        'applicant_contact_count', 'maxminusapplicant'
    ]
    
    # Add contact and email metrics if available
    extended_metrics = [
        'total_contacts', 'email_count', 'phone_count', 'fax_count',
        'email_interaction_count', 'email_method_count',
        'human_touch_count', 'fully_autonomous', 'has_rework'
    ]
    
    # Filter for metrics that exist in the DataFrame
    metrics = [m for m in metrics if m in df_summary.columns]
    extended_metrics = [m for m in extended_metrics if m in df_summary.columns]
    all_metrics = metrics + extended_metrics
    
    # Group by category and calculate metrics
    if all_metrics:
        # Filter for completed searches
        if 'is_completed' in df_summary.columns:
            df_completed = df_summary[df_summary['is_completed'] == 1].copy()
            if not df_completed.empty:
                category_metrics = df_completed.groupby('mb_category').agg(
                    count=('searchid', 'count'),
                    **{f'{m}_mean': (m, 'mean') for m in all_metrics},
                    **{f'{m}_median': (m, 'median') for m in all_metrics}
                ).reset_index()
                
                print("\nMetrics by MB Capability Category (Completed Searches):")
                print(category_metrics.to_markdown(index=False))
                
                return {
                    'category_counts': category_counts,
                    'category_metrics': category_metrics
                }
            else:
                print("No completed searches found for category analysis")
        else:
            # If completion status not available, use all searches
            category_metrics = df_summary.groupby('mb_category').agg(
                count=('searchid', 'count'),
                **{f'{m}_mean': (m, 'mean') for m in all_metrics},
                **{f'{m}_median': (m, 'median') for m in all_metrics}
            ).reset_index()
            
            print("\nMetrics by MB Capability Category (All Searches):")
            print(category_metrics.to_markdown(index=False))
            
            return {
                'category_counts': category_counts,
                'category_metrics': category_metrics
            }
    else:
        print("No valid metrics found for category analysis")
        return None


def analyze_mb_contact_plan_impact(df_history: pd.DataFrame, mb_agent_id: str) -> dict:
    """
    Analyzes the impact of Murphy Brown's contact plans.
    
    Uses the raw df_history.
    
    Specifically measures:
    1. Coverage: Percentage of searches where MB provided a contact plan.
    2. Content: Average number of distinct contacts per provided plan.
    Aggregates results weekly and monthly.

    Args:
        df_history (pd.DataFrame): DataFrame containing history records with columns
                                     like 'searchid', 'userid', 'note', 'historydatetime'.
        mb_agent_id (str): The identifier for the Murphy Brown agent (e.g., 'murphy.brown').

    Returns:
        dict: A dictionary containing two DataFrames:
              'weekly_summary': Aggregated results by week.
              'monthly_summary': Aggregated results by month.
              Returns empty dict if required columns are missing.
    """
    print("\n--- Analyzing Murphy Brown Contact Plan Impact ---")

    required_cols = ['searchid', 'userid', 'note', 'historydatetime']
    if not all(col in df_history.columns for col in required_cols):
        print(f"Error: Missing required columns for contact plan analysis. Need: {required_cols}")
        return {}

    # --- Preparation ---
    df = df_history.copy()
    # Ensure correct types and handle missing values safely
    df['userid_lower'] = df['userid'].fillna('').astype(str).str.lower()
    df['note_clean'] = df['note'].fillna('').astype(str).str.lower()
    # Ensure historydatetime is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['historydatetime']):
        try:
            df['historydatetime'] = pd.to_datetime(df['historydatetime'], errors='coerce')
            if df['historydatetime'].isnull().any():
                 print("Warning: Some 'historydatetime' values could not be converted and were set to NaT.")
        except Exception as e:
            print(f"Error converting 'historydatetime' to datetime: {e}. Cannot proceed with time aggregation.")
            return {}


    # --- Identify MB Contact Plan Rows --- 
    is_mb_user = (df['userid_lower'] == mb_agent_id.lower())
    # Use a more specific check if needed, e.g., note *starts with* the phrase
    has_contact_phrase = df['note_clean'].str.contains(CONTACT_PLAN_PHRASE, case=False, na=False)
    df['is_mb_contact_plan_row'] = is_mb_user & has_contact_phrase
    plan_rows_mask = df['is_mb_contact_plan_row'] # Cache the mask

    # --- Count Contacts within Plan Notes (Using Regex) --- 
    def count_contacts_in_note_regex(note_text):
        """Counts occurrences of contact keywords in a note using regex."""
        if pd.isna(note_text) or not note_text:
            return 0
        count = 0
        for pattern in CONTACT_TYPE_PATTERNS.values(): # Use defined patterns
            # Use re.findall to count all occurrences
            count += len(re.findall(pattern, note_text, re.IGNORECASE))
        return count

    # Only calculate counts for rows identified as MB contact plans
    df['contacts_in_plan'] = 0 # Initialize
    df.loc[plan_rows_mask, 'contacts_in_plan'] = df.loc[plan_rows_mask, 'note_clean'].apply(count_contacts_in_note_regex)

    # --- Aggregate by Search ID --- 
    # Get the timestamp of the *first* MB contact plan note for each search
    plan_dates = df.loc[plan_rows_mask].groupby('searchid')['historydatetime'].min().rename('contact_plan_date')
    
    # Group by searchid to determine if *any* history row was an MB contact plan
    # and sum the contacts found across all such rows for that search.
    search_agg = df.groupby('searchid', as_index=False).agg(
        contact_plan_provided=('is_mb_contact_plan_row', 'max'), # True if any row was a contact plan
        total_contacts_provided=('contacts_in_plan', 'sum'),   # Sum contacts from all MB plan rows for this search
        # first_history_date=('historydatetime', 'min') # No longer needed for bucketing
    )
    
    # Merge the actual plan date
    search_agg = pd.merge(search_agg, plan_dates, on='searchid', how='left')

    # Ensure contacts are 0 if no plan was provided (max was False)
    search_agg.loc[~search_agg['contact_plan_provided'], 'total_contacts_provided'] = 0

    if search_agg['contact_plan_date'].isnull().all():
        print("Warning: Cannot determine time periods as 'contact_plan_date' is all NaT.")
        return {'weekly_summary': pd.DataFrame(), 'monthly_summary': pd.DataFrame()}

    # Drop rows where plan date couldn't be determined for reliable time aggregation
    search_agg = search_agg.dropna(subset=['contact_plan_date'])
    if search_agg.empty:
        print("Warning: No valid data remaining after removing rows with missing plan dates.")
        return {'weekly_summary': pd.DataFrame(), 'monthly_summary': pd.DataFrame()}


    # --- Aggregate by Time Period (Weekly/Monthly) using contact_plan_date --- 
    search_agg['week_period'] = search_agg['contact_plan_date'].dt.to_period('W-MON').astype(str)
    search_agg['month_period'] = search_agg['contact_plan_date'].dt.to_period('M').astype(str)

    results = {}
    for period_col, period_name in [('week_period', 'Weekly'), ('month_period', 'Monthly')]:
        period_summary = search_agg.groupby(period_col).agg(
            total_searches=('searchid', 'nunique'),
            searches_with_contact_plans=('contact_plan_provided', 'sum'), # Summing boolean True counts them
            total_contacts_in_plans=('total_contacts_provided', 'sum')
        ).reset_index()

        # Calculate derived metrics
        period_summary['percentage_with_contact_plans'] = (
            period_summary['searches_with_contact_plans'] / period_summary['total_searches'] * 100
        ).round(2)

        # Calculate average contacts per plan, handle division by zero
        period_summary['avg_contacts_per_plan'] = np.where(
            period_summary['searches_with_contact_plans'] > 0,
            (period_summary['total_contacts_in_plans'] / period_summary['searches_with_contact_plans']).round(2),
            0 # Assign 0 if no searches had contact plans in that period
        )

        # Rename columns for clarity
        period_summary = period_summary.rename(columns={
            period_col: f'{period_name} Period',
            'total_searches': 'Total Searches',
            'searches_with_contact_plans': 'Searches w/ MB Plan',
            'percentage_with_contact_plans': '% Searches w/ MB Plan',
            'avg_contacts_per_plan': 'Avg Contacts per Plan'
        })

        # Select and order final columns
        final_cols = [f'{period_name} Period', 'Total Searches', 'Searches w/ MB Plan',
                      '% Searches w/ MB Plan', 'Avg Contacts per Plan']
        results[f'{period_name.lower()}_summary'] = period_summary[final_cols]
        print(f"Generated {period_name} contact plan summary.")

    print("--- Finished Murphy Brown Contact Plan Impact Analysis ---")

    # keep a per-search flag table so we can join back later
    results['plan_flags'] = search_agg[['searchid', 'contact_plan_provided']] # <-- ADDED

    return results


# ─── NEW: delta-metrics for searches WITH vs WITHOUT a contact-plan ────────────
def contact_plan_delta(df_summary: pd.DataFrame,
                       contact_plan_flags: pd.DataFrame) -> pd.DataFrame: # Added type hints for clarity
    """
    Compare searches with vs. without an MB contact-plan using the pre-merged df_summary.
    
    Args:
        df_summary: Summary DataFrame which *already includes* the 'contact_plan_provided' flag.
        contact_plan_flags: This argument is now unused but kept for signature consistency if needed elsewhere.

    Returns:
        DataFrame with delta metrics.
    """
    # --- FIX: Remove internal merge, work directly with df_summary ---
    # The 'contact_plan_provided' column is assumed to exist and be filled in df_summary already.
    merged = df_summary.copy() 
    # Ensure the flag is boolean type
    if 'contact_plan_provided' in merged.columns:
         merged['contact_plan_provided'] = merged['contact_plan_provided'].fillna(False).astype(bool)
    else:
        # This case should not happen due to early initialization, but handle defensively
        print("Error in contact_plan_delta: 'contact_plan_provided' column unexpectedly missing from input df_summary.")
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'contact_plan_provided', 'search_count', 'completion_rate', 
            'avg_tat_days', 'avg_search_attempts', 'delta_completion_rate', 
            'delta_tat_days', 'delta_attempts'
        ])
    # --- END FIX ---

    # Ensure required columns exist before grouping
    required_cols = ['searchid', 'is_completed', 'tat_calendar_days', 'search_attempts', 'contact_plan_provided']
    if not all(col in merged.columns for col in required_cols):
        missing = [col for col in required_cols if col not in merged.columns]
        print(f"Warning: Missing required columns for delta calculation in df_summary: {missing}")
        # Return an empty DataFrame or handle appropriately
        return pd.DataFrame(columns=[
            'contact_plan_provided', 'search_count', 'completion_rate', 
            'avg_tat_days', 'avg_search_attempts', 'delta_completion_rate', 
            'delta_tat_days', 'delta_attempts'
        ])

    if True not in merged['contact_plan_provided'].values or \
       False not in merged['contact_plan_provided'].values:
        print("contact_plan_delta: no with-plan vs without-plan contrast – returning None")
        return None

    # --- Convert metrics to numeric before aggregation ---
    merged['is_completed'] = pd.to_numeric(merged['is_completed'], errors='coerce')
    merged['tat_calendar_days'] = pd.to_numeric(merged['tat_calendar_days'], errors='coerce')
    merged['search_attempts'] = pd.to_numeric(merged['search_attempts'], errors='coerce')
    # ------------------------------------------------------

    grp = merged.groupby('contact_plan_provided')
    out = grp.agg(
        search_count           = ('searchid', 'size'),
        completion_rate        = ('is_completed', 'mean'),
        avg_tat_days           = ('tat_calendar_days', 'mean'),
        avg_search_attempts    = ('search_attempts', 'mean')
    ).reset_index()

    # Δ columns (plan – no-plan)
    # Check if both True and False groups exist in the output 'out'
    if True in out['contact_plan_provided'].values and False in out['contact_plan_provided'].values:
        row_with = out[out['contact_plan_provided'] == True]
        row_without = out[out['contact_plan_provided'] == False]

        # Safely access values only if rows exist (redundant check, but safe)
        if not row_with.empty and not row_without.empty:
             # Check if means are valid before calculating delta
             delta_completion_rate = np.nan
             delta_tat_days = np.nan
             delta_attempts = np.nan
             if pd.notna(row_with['completion_rate'].values[0]) and pd.notna(row_without['completion_rate'].values[0]):
                 delta_completion_rate = row_with['completion_rate'].values[0]  - row_without['completion_rate'].values[0]
             if pd.notna(row_with['avg_tat_days'].values[0]) and pd.notna(row_without['avg_tat_days'].values[0]):
                 delta_tat_days        = row_with['avg_tat_days'].values[0]     - row_without['avg_tat_days'].values[0]
             if pd.notna(row_with['avg_search_attempts'].values[0]) and pd.notna(row_without['avg_search_attempts'].values[0]):
                 delta_attempts        = row_with['avg_search_attempts'].values[0] - row_without['avg_search_attempts'].values[0]
             
             # --- FIX: Write delta only to the 'True' row ---
             out.loc[out.contact_plan_provided == True, 'delta_completion_rate'] = delta_completion_rate
             out.loc[out.contact_plan_provided == True, 'delta_tat_days'] = delta_tat_days
             out.loc[out.contact_plan_provided == True, 'delta_attempts'] = delta_attempts
             # --- END FIX ---
        else:
            # This case should ideally not happen if both True/False are in values
            print("Warning: Could not find both 'with plan' and 'without plan' groups for delta calculation despite checks.")
            # Initialize delta columns as NaN if calculation fails for the True row
            out['delta_completion_rate'] = np.nan
            out['delta_tat_days']        = np.nan
            out['delta_attempts']        = np.nan
    else:
         print("Warning: Missing either 'with plan' or 'without plan' group for delta calculation.")
         # Initialize delta columns even if calculation fails entirely
         out['delta_completion_rate'] = np.nan
         out['delta_tat_days']        = np.nan
         out['delta_attempts']        = np.nan


    return out
# ───────────────────────────────────────────────────────────────────────────────


def create_mb_impact_dashboard(result_sets, output_dir=None):
    """
    Create a set of visualizations summarizing MB impact metrics.
    
    Args:
        result_sets: Dictionary of analysis results
        output_dir: Directory to save visualization files
    
    Returns:
        Path to dashboard directory
    """
    print("Creating dashboard visualizations...")
    
    # Set up output directory
    if output_dir is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        output_dir = os.path.join(script_dir, "Output", "mb_impact_dashboard")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-darkgrid') # Changed from 'seaborn-darkgrid'
    
    # Create visualizations based on available result sets
    
    # 1. Capability distribution plot
    if 'capability_impact' in result_sets and result_sets['capability_impact'] is not None and 'group_counts' in result_sets['capability_impact']:
        group_counts = result_sets['capability_impact']['group_counts']
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Capability Group', y='Count', data=group_counts)
        plt.title('Distribution of MB Capability Groups', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add count labels
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'bottom', fontsize=11)
        
        plt.savefig(os.path.join(output_dir, 'capability_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. TAT comparison plot
    if 'time_efficiency' in result_sets and result_sets['time_efficiency'] is not None and 'tat_stats' in result_sets['time_efficiency']:
        tat_stats = result_sets['time_efficiency']['tat_stats']
        
        # Reshape for plotting
        tat_melt = pd.melt(
            tat_stats, 
            id_vars=['mb_touched', 'search_count'],
            value_vars=['avg_tat_days', 'median_tat_days', 'avg_tat_business', 'median_tat_business'],
            var_name='metric', value_name='value'
        )
        
        tat_melt['mb_status'] = tat_melt['mb_touched'].map({0: 'Non-MB', 1: 'MB'})
        
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='metric', y='value', hue='mb_status', data=tat_melt)
        plt.title('Turnaround Time (TAT) Comparison: MB vs Non-MB', fontsize=14)
        plt.xlabel('')
        plt.ylabel('Days')
        
        # Rename x-tick labels
        new_labels = {
            'avg_tat_days': 'Avg Calendar Days',
            'median_tat_days': 'Median Calendar Days',
            'avg_tat_business': 'Avg Business Days',
            'median_tat_business': 'Median Business Days'
        }
        ax.set_xticklabels([new_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()])
        plt.xticks(rotation=30, ha='right')
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tat_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Autonomy metrics plot
    if 'autonomy' in result_sets and result_sets['autonomy'] is not None and 'summary_metrics' in result_sets['autonomy']:
        autonomy_metrics = result_sets['autonomy']['summary_metrics']
        
        if autonomy_metrics:
            metric_df = pd.DataFrame([
                {'Metric': 'Autonomy Rate (%)', 'Value': autonomy_metrics.get('autonomy_rate', 0)},
                {'Metric': 'Rework Rate (%)', 'Value': autonomy_metrics.get('rework_rate', 0)},
                {'Metric': 'Avg Human Touches', 'Value': autonomy_metrics.get('avg_human_touches', 0)}
            ])
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Metric', y='Value', data=metric_df)
            plt.title('MB Autonomy Metrics', fontsize=14)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'autonomy_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Time savings plot
    if 'time_savings' in result_sets and result_sets['time_savings'] is not None:
        time_savings = result_sets['time_savings']
        
        if time_savings:
            # Create comparison bar chart for touches
            touch_df = pd.DataFrame([
                {'Group': 'Non-MB', 'Touches': time_savings.get('avg_touches_non_mb', 0)},
                {'Group': 'MB', 'Touches': time_savings.get('avg_touches_mb', 0)}
            ])
            
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x='Group', y='Touches', data=touch_df)
            plt.title('Average Human Touches Comparison', fontsize=14)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'touch_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create time savings summary
            fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
            
            # Hide axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            
            # Add text summary
            plt.text(0.5, 0.8, 'Time Savings Summary', fontsize=18, weight='bold', ha='center')
            plt.text(0.5, 0.7, f"Total Hours Saved: {time_savings.get('total_hours_saved', 0):.1f}", fontsize=14, ha='center')
            plt.text(0.5, 0.6, f"Weekly Hours Saved: {time_savings.get('weekly_hours_saved', 0):.1f}", fontsize=14, ha='center')
            plt.text(0.5, 0.5, f"FTE Equivalent: {time_savings.get('fte_equivalent', 0):.2f}", fontsize=14, ha='center')
            plt.text(0.5, 0.4, f"Hours Saved Per Verification: {time_savings.get('hours_saved_per_verification', 0):.2f}", fontsize=14, ha='center')
            plt.text(0.5, 0.3, f"Touch Reduction: {time_savings.get('touch_reduction', 0):.2f} touches per search", fontsize=14, ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'time_savings_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. Contact efficiency plot (if available)
    if 'contact_efficiency' in result_sets and result_sets['contact_efficiency'] is not None and 'attempts_stats' in result_sets['contact_efficiency']:
        attempts_stats = result_sets['contact_efficiency']['attempts_stats']
        
        plt.figure(figsize=(10, 6))
        
        attempts_melt = pd.melt(
            attempts_stats,
            id_vars=['mb_touched', 'search_count'],
            value_vars=['avg_attempts', 'median_attempts'],
            var_name='metric', value_name='value'
        )
        
        attempts_melt['mb_status'] = attempts_melt['mb_touched'].map({0: 'Non-MB', 1: 'MB'})
        
        ax = sns.barplot(x='metric', y='value', hue='mb_status', data=attempts_melt)
        plt.title('Outbound Attempts Comparison: MB vs Non-MB', fontsize=14)
        plt.xlabel('')
        plt.ylabel('Number of Attempts')
        
        # Rename x-tick labels
        new_labels = {
            'avg_attempts': 'Average Attempts',
            'median_attempts': 'Median Attempts'
        }
        ax.set_xticklabels([new_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()])
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attempts_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Create HTML dashboard index
    html_output = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Murphy Brown Impact Analysis Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            h1, h2 {{ color: #333366; }}
            .dashboard-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .dashboard-item {{ flex: 1 1 45%; min-width: 300px; margin-bottom: 20px; background: #f9f9f9; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
            .timestamp {{ color: #666; font-size: 0.8em; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <h1>Murphy Brown Impact Analysis Dashboard</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="dashboard-container">
            <div class="dashboard-item">
                <h2>Capability Distribution</h2>
                <img src="capability_distribution.png" alt="MB Capability Distribution" />
            </div>
            
            <div class="dashboard-item">
                <h2>Turnaround Time Comparison</h2>
                <img src="tat_comparison.png" alt="TAT Comparison" />
            </div>
            
            <div class="dashboard-item">
                <h2>Autonomy Metrics</h2>
                <img src="autonomy_metrics.png" alt="Autonomy Metrics" />
            </div>
            
            <div class="dashboard-item">
                <h2>Human Touch Comparison</h2>
                <img src="touch_comparison.png" alt="Human Touch Comparison" />
            </div>
            
            <div class="dashboard-item">
                <h2>Time Savings</h2>
                <img src="time_savings_summary.png" alt="Time Savings Summary" />
            </div>
            
            <div class="dashboard-item">
                <h2>Outbound Attempts</h2>
                <img src="attempts_comparison.png" alt="Attempts Comparison" />
            </div>
        </div>
        
        <div class="timestamp">
            <p>Analysis performed using murphy_brown_impact_analysis.py</p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'dashboard.html'), 'w') as f:
        f.write(html_output)
    
    print(f"Dashboard visualizations and HTML index saved to: {output_dir}")
    return output_dir


# --- NEW: Verdict Functions ---

def email_performance_verdict(df_summary, min_pp=5, alpha=0.05):
    """
    Return True if MB email yield is materially better than control.
    
    Args:
        df_summary: Summary DataFrame.
        min_pp: Minimum difference in percentage points required for verdict.
        alpha: Significance level for t-test.
    
    Returns:
        Dictionary with verdict and supporting metrics.
    """
    # Check required columns first
    required_cols = ['searchid', 'mb_touched', 'email_method_count', 'email_interaction_count']
    if not all(col in df_summary.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_summary.columns]
        print(f"Warning: Cannot calculate email verdict. Missing columns: {missing}")
        return {'verdict': False, 'reason': f'missing columns: {missing}', 
                'mb_yield': np.nan, 'ctl_yield': np.nan, 'diff_pp': np.nan, 'p_value': np.nan}

    # Convert counts to numeric, coercing errors, before filtering
    # Use .copy() to avoid SettingWithCopyWarning if df_summary is used later
    df_summary_copy = df_summary.copy()
    df_summary_copy['email_method_count'] = pd.to_numeric(df_summary_copy['email_method_count'], errors='coerce')
    df_summary_copy['email_interaction_count'] = pd.to_numeric(df_summary_copy['email_interaction_count'], errors='coerce')

    # Keep searches with at least one outbound email so denominator is valid
    subset = df_summary_copy[df_summary_copy['email_method_count'] > 0] # No .copy needed here as it's a new slice
    
    if subset.empty:
        print("Info: No searches found with email_method_count > 0 for email verdict.")
        return {'verdict': False, 'reason': 'no searches with email method count > 0', 
                'mb_yield': np.nan, 'ctl_yield': np.nan, 'diff_pp': np.nan, 'p_value': np.nan}
                
    subset = subset.copy() # Explicitly copy the subset now to avoid SettingWithCopyWarning on yield calculation
    subset['email_yield'] = subset['email_interaction_count'] / subset['email_method_count']

    mb  = subset[subset['mb_touched'] == 1]['email_yield'].dropna()
    ctl = subset[subset['mb_touched'] == 0]['email_yield'].dropna()

    # Calculate means regardless of t-test possibility
    mb_mean = mb.mean() if not mb.empty else np.nan
    ctl_mean = ctl.mean() if not ctl.empty else np.nan
    # Calculate difference in percentage points (since yields are 0-1)
    diff_pp = (mb_mean - ctl_mean) * 100 if pd.notna(mb_mean) and pd.notna(ctl_mean) else np.nan

    if len(mb) < 2 or len(ctl) < 2:
        print(f"Info: Insufficient data for email verdict t-test (MB: {len(mb)}, CTL: {len(ctl)}). Need at least 2 in each group.")
        return {'verdict': False, 'reason': f'insufficient data (MB: {len(mb)}, CTL: {len(ctl)})', 
                'mb_yield': mb_mean, 
                'ctl_yield': ctl_mean, 
                'diff_pp': diff_pp,
                'p_value': np.nan}

    # Welch t-test (unequal var)
    try:
        t, p = stats.ttest_ind(mb, ctl, equal_var=False, nan_policy='omit') # Added nan_policy
    except Exception as e:
        print(f"Error during email yield t-test: {e}")
        return {'verdict': False, 'reason': 't-test error', 
                'mb_yield': mb_mean, 
                'ctl_yield': ctl_mean, 
                'diff_pp': diff_pp,
                'p_value': np.nan}

    verdict = (diff_pp >= min_pp) and (p < alpha) # Compare diff_pp calculated earlier
    return {
        'verdict'    : verdict,
        'mb_yield'   : mb_mean,
        'ctl_yield'  : ctl_mean,
        'diff_pp'    : diff_pp,
        'p_value'    : p,
        'reason'     : 'OK'
    }
# --- END Verdict Functions ---


# --- Additional Analysis Functions ---

def agent_completion_uplift(df_summary, agent_col='agentname', min_searches_per_group=5):
    """Calculate completion rate uplift per agent when a contact plan is provided."""
    print(f"Calculating agent completion uplift using column: {agent_col}")
    required_cols = [agent_col, 'contact_plan_provided', 'is_completed', 'searchid']
    if not all(col in df_summary.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_summary.columns]
        print(f"Warning: Cannot calculate agent uplift. Missing columns: {missing}")
        return pd.DataFrame()

    # Ensure agent column is suitable for grouping (e.g., handle NAs)
    df_summary[agent_col] = df_summary[agent_col].fillna('Unknown Agent').astype(str)
    # Ensure boolean flag is boolean
    df_summary['contact_plan_provided'] = df_summary['contact_plan_provided'].astype(bool)
    # Ensure completion is numeric (0/1)
    df_summary['is_completed'] = pd.to_numeric(df_summary['is_completed'], errors='coerce')

    # Calculate mean completion rate per agent, per plan status
    try:
        # Group by agent and plan status, calculate mean completion, then unstack
        g = df_summary.groupby([agent_col, 'contact_plan_provided'])['is_completed'].mean().unstack()
        # Calculate counts as well for filtering
        counts = df_summary.groupby([agent_col, 'contact_plan_provided']).size().unstack(fill_value=0)
        g = g.join(counts, rsuffix='_count')
        
        # Filter agents: must have data for both plan=True and plan=False
        g = g.dropna(subset=[True, False]) 
        # Filter agents: must have minimum number of searches in both groups
        # Use string keys for the count columns generated by the join
        g = g[(g['True_count'] >= min_searches_per_group) & (g['False_count'] >= min_searches_per_group)]
        
    except KeyError as e:
        print(f"Error during grouping/unstacking for agent uplift (likely bad agent column '{agent_col}' or missing True/False in contact_plan_provided after filtering): {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error during initial agent uplift grouping: {e}")
        return pd.DataFrame()

    if g.empty:
        print("No agents found meeting the criteria for completion uplift analysis.")
        return pd.DataFrame()

    g['delta_completion_pp'] = (g[True] - g[False]) * 100
    
    # Flag significance (Welch t-test per agent)
    sig = []
    p_values = []
    agent_list = g.index # Get the list of agents remaining after filtering
    
    # Group the original summary df ONLY by the agents we are analyzing
    df_filtered_agents = df_summary[df_summary[agent_col].isin(agent_list)]
    
    for agent, sub in df_filtered_agents.groupby(agent_col):
        # Ensure we are only processing agents still in our filtered group 'g'
        if agent not in agent_list:
            continue
            
        plan = sub[sub.contact_plan_provided]['is_completed'].dropna() # is_completed is already numeric
        nolp = sub[~sub.contact_plan_provided]['is_completed'].dropna()
        
        # Check counts again (redundant but safe)
        if len(plan) >= min_searches_per_group and len(nolp) >= min_searches_per_group:
            try:
                # Perform t-test only if variance is not zero in both groups
                if plan.var(ddof=0) > 0 and nolp.var(ddof=0) > 0:
                    _, p = stats.ttest_ind(plan, nolp, equal_var=False, nan_policy='omit')
                    sig.append(p < 0.05)
                    p_values.append(p)
                else:
                    # Cannot perform t-test if one group has zero variance (all same outcome)
                    sig.append(False)
                    p_values.append(np.nan) # Indicate test not performed
                    print(f"Info: Skipping t-test for agent {agent} due to zero variance in one group.")
            except Exception as e:
                print(f"Error during t-test for agent {agent}: {e}")
                sig.append(False)
                p_values.append(np.nan)
        else:
            # This case should ideally not be reached due to prior filtering, but as a safeguard
            sig.append(False)
            p_values.append(np.nan)
            
    # Add results back to the filtered DataFrame 'g'
    # Important: Ensure the length matches and indices align if needed, though direct assignment should work if order is preserved
    if len(sig) == len(g):
        g['significant'] = sig
        g['p_value'] = p_values
    else:
        print(f"Warning: Mismatch in significance result length ({len(sig)}) and agent group length ({len(g)}). Significance not added.")
        g['significant'] = False # Default value
        g['p_value'] = np.nan

    print(f"Agent completion uplift calculation complete for {len(g)} agents.")
    return g.reset_index()

def open_queue_age(df_history):
    """Calculate age buckets for open searches, grouped by MB touch."""
    print("Calculating open queue age distribution...")
    required_cols = ['searchstatus', 'historydatetime', 'is_mb_row', 'searchid']
    if not all(col in df_history.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_history.columns]
        print(f"Warning: Cannot calculate queue age. Missing columns: {missing}")
        return pd.DataFrame()
        
    # Ensure correct types
    df_history['searchstatus'] = df_history['searchstatus'].fillna('').astype(str).str.upper()
    if not pd.api.types.is_datetime64_any_dtype(df_history['historydatetime']):
        df_history['historydatetime'] = pd.to_datetime(df_history['historydatetime'], errors='coerce')
    if 'is_mb_row' not in df_history.columns or not pd.api.types.is_bool_dtype(df_history['is_mb_row']):
        # Attempt to recalculate or convert if missing/wrong type, assuming MB_AGENT_IDENTIFIER is global
        print("Warning: 'is_mb_row' flag missing or invalid type. Attempting recalculation.")
        mb_pattern = MB_AGENT_IDENTIFIER.lower()
        df_history['is_mb_row'] = (
            df_history['agentname'].fillna('').astype(str).str.lower().str.contains(mb_pattern, na=False) |
            df_history['username'].fillna('').astype(str).str.lower().str.contains(mb_pattern, na=False)
        )
        
    df_copy = df_history[required_cols].copy()
    
    # Use timezone-aware current time, then normalize
    # Use pandas current time function which respects system timezone better potentially
    # --- MODIFIED: Robust timezone handling ---
    source_tz = None
    if not df_copy['historydatetime'].dropna().empty:
        first_valid_dt = df_copy['historydatetime'].dropna().iloc[0]
        if hasattr(first_valid_dt, 'tz'): # Check if it's a timezone-aware Timestamp
            source_tz = first_valid_dt.tz

    today = pd.Timestamp.now(tz=source_tz).normalize() if source_tz else pd.Timestamp.today().normalize()
    # --- END MODIFIED ---
    
    # Filter for open searches (most recent status per search is NOT REVIEW)
    # Get the last record per searchid
    last_records = df_copy.loc[df_copy.groupby('searchid')['historydatetime'].idxmax()]
    open_searches = last_records[last_records['searchstatus'] != 'REVIEW']
    
    if open_searches.empty:
        print("No open searches found (last status != 'REVIEW').")
        return pd.DataFrame()
        
    # Calculate age based on the LATEST historydatetime of the open search
    open_searches['age_days'] = (today - open_searches['historydatetime'].dt.normalize()).dt.days
    
    # Handle potential negative ages if historydatetime is in the future
    open_searches['age_days'] = open_searches['age_days'].clip(lower=0)
    
    bins = [-1, 2, 5, 10, 20, float('inf')] # Use -1 for edge case <=0, inf for upper bound
    labels = ['0-2 days', '3-5 days', '6-10 days', '11-20 days', '>20 days'] # Adjusted labels
    open_searches['age_bucket'] = pd.cut(open_searches['age_days'], bins=bins, labels=labels, right=True)
    
    # Group by MB status and age bucket
    by_mb = (
        open_searches.groupby(['is_mb_row', 'age_bucket'], observed=False) # observed=False includes empty buckets
                     .size().rename('count')
                     .reset_index()
    )
    print("Open queue age calculation complete.")
    return by_mb

# ──────────────────────────────────────────────────────────────
def weekly_verified_per_agent(df_summary, agent_col='agentname'):
    """
    Returns one row per (agent, ISO week) with # completed searches.
    Requires df_summary to hold:
        • agent_col (e.g., 'agentname') – string
        • completiondate            – datetime
        • is_completed              – 1/0
    """
    print(f"Calculating weekly verified count per agent ({agent_col})...")
    required = {agent_col, 'completiondate', 'is_completed'}
    if not required.issubset(df_summary.columns):
        print(f"Skipping weekly throughput – missing {required.difference(df_summary.columns)}")
        return pd.DataFrame()

    # Ensure types are correct before filtering/grouping
    df_summary[agent_col] = df_summary[agent_col].fillna('Unknown Agent').astype(str)
    df_summary['is_completed'] = pd.to_numeric(df_summary['is_completed'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df_summary['completiondate']):
        df_summary['completiondate'] = pd.to_datetime(df_summary['completiondate'], errors='coerce')

    tmp = df_summary[(df_summary['is_completed'] == 1) & df_summary['completiondate'].notna()].copy()
    if tmp.empty:
        print("No completed searches with valid completion dates found for weekly throughput.")
        return pd.DataFrame()
        
    tmp['week'] = tmp['completiondate'].dt.to_period('W-MON').astype(str)
    try:
        out = (
            tmp.groupby([agent_col, 'week'], as_index=False)
                .size()
                .rename(columns={'size': 'verified_count'})
        )
        print("Weekly verified count calculation complete.")
        return out
    except Exception as e:
        print(f"Error during weekly verified grouping: {e}")
        return pd.DataFrame()
# ──────────────────────────────────────────────────────────────

def first_day_sla_rate(df_summary):
    """
    Calculates % of searches closed in ≤1 calendar day, MB vs non-MB.
    Adds column 'first_day_close' for possible downstream use.
    """
    print("Calculating first day SLA rate...")
    required = {'tat_calendar_days', 'mb_touched'}
    if not required.issubset(df_summary.columns):
        print(f"Skipping first day SLA rate – missing {required.difference(df_summary.columns)}")
        # Return None as per original code structure, caller handles it
        return None 
        
    # Ensure TAT is numeric
    # Use .copy() to avoid modifying the original df_summary inplace unnecessarily
    df_copy = df_summary.copy()
    df_copy['tat_calendar_days'] = pd.to_numeric(df_copy['tat_calendar_days'], errors='coerce')

    # Add flag (handle NAs in TAT)
    df_copy['first_day_close'] = np.where(
        df_copy['tat_calendar_days'].notna(),
        (df_copy['tat_calendar_days'] <= 1).astype(int),
        0 # Treat NA TAT as not meeting SLA
    )
    
    try:
        out = (
            df_copy.groupby('mb_touched')['first_day_close']
                      .mean()
                      .rename('sla_rate')
                      .reset_index()
        )
        print("First day SLA rate calculation complete.")
        # Add the 'first_day_close' column back to the original df_summary if needed downstream
        # df_summary['first_day_close'] = df_copy['first_day_close'] # Uncomment if needed
        return out
    except Exception as e:
        print(f"Error during first day SLA grouping: {e}")
        return None # Match original return type on error
# ──────────────────────────────────────────────────────────────

def queue_depth_weekly(df_history, sla_days=5):
    """
    Weekly % of open (non-REVIEW) searches older than sla_days business days.
    Calculates age based on the *last* history entry for each search.
    """
    print("Calculating weekly queue depth...")
    need = {'searchid', 'historydatetime', 'searchstatus'}
    if not need.issubset(df_history.columns):
        print(f"Queue-depth calc skipped – missing {need.difference(df_history.columns)}")
        return pd.DataFrame()
        
    # Ensure types
    if not pd.api.types.is_datetime64_any_dtype(df_history['historydatetime']):
        df_history['historydatetime'] = pd.to_datetime(df_history['historydatetime'], errors='coerce')
    df_history['searchstatus'] = df_history['searchstatus'].fillna('').astype(str).str.upper()
    
    df_copy = df_history[list(need)].dropna(subset=['historydatetime']).copy()
    if df_copy.empty:
        print("No valid history records found for queue depth calculation.")
        return pd.DataFrame()

    # Get the last known status and timestamp for each search
    last_records = df_copy.loc[df_copy.groupby('searchid')['historydatetime'].idxmax()]
    open_searches = last_records[last_records['searchstatus'] != 'REVIEW'].copy()

    if open_searches.empty:
        print("No open searches (last status != REVIEW) found for queue depth.")
        return pd.DataFrame()

    # Use timezone-aware current time if possible, then normalize
    source_tz = open_searches['historydatetime'].dt.tz
    today = pd.Timestamp.now(tz=source_tz).normalize()
    
    # Calculate business day age using the date part (timezone-naive)
    start_dates_naive = open_searches['historydatetime'].dt.tz_localize(None).dt.normalize()
    today_naive = today.tz_localize(None).normalize()
    
    try:
        # Vectorized business day calculation
        # Ensure dates are valid numpy datetimes before passing
        start_dt64 = start_dates_naive.values.astype('datetime64[D]')
        # Add 1 day to today to make the range inclusive for busday_count
        # Ensure today_naive is a Timestamp before adding timedelta
        if isinstance(today_naive, pd.Timestamp):
            end_dt64 = np.datetime64(today_naive + pd.Timedelta(days=1), 'D')
        else: # Should not happen, but fallback
             end_dt64 = np.datetime64(pd.Timestamp(today_naive) + pd.Timedelta(days=1), 'D')
        
        open_searches['age_bus'] = np.busday_count(
            start_dt64,
            end_dt64, 
            holidays=HOLIDAYS
        )
    except Exception as e:
        print(f"Error calculating business days for queue depth: {e}")
        # Add traceback for more detail
        traceback.print_exc()
        return pd.DataFrame()
        
    # Handle potential negative ages if history is in the future (clip at 0)
    open_searches['age_bus'] = open_searches['age_bus'].clip(lower=0)

    # Assign week based on the last history date
    open_searches['week'] = start_dates_naive.dt.to_period('W-MON').astype(str)
    
    try:
        weekly = (
            open_searches.groupby('week')
                     .agg(total_open=('searchid','nunique'),
                          open_gt_sla=('age_bus', lambda x: (x > sla_days).sum()))
                     .reset_index()
        )
        
        # Calculate percentage, handle division by zero
        weekly['pct_open_gt_sla'] = np.where(
            weekly['total_open'] > 0,
            (weekly['open_gt_sla'] / weekly['total_open'] * 100).round(2),
            0
        )
        print("Weekly queue depth calculation complete.")
        return weekly
    except Exception as e:
        print(f"Error during weekly queue depth grouping: {e}")
        return pd.DataFrame()
# ──────────────────────────────────────────────────────────────

# monthly  (just swap the .dt accessor)
def monthly_verified_per_agent(df_summary, agent_col='agentname'):
    """
    Returns one row per (agent, month) with # completed searches.
    """
    print(f"Calculating monthly verified count per agent ({agent_col})...")
    required = {agent_col, 'completiondate', 'is_completed'}
    if not required.issubset(df_summary.columns):
        print(f"Skipping monthly throughput – missing {required.difference(df_summary.columns)}")
        return pd.DataFrame()

    # Ensure types are correct before filtering/grouping
    df_summary[agent_col] = df_summary[agent_col].fillna('Unknown Agent').astype(str)
    df_summary['is_completed'] = pd.to_numeric(df_summary['is_completed'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df_summary['completiondate']):
        df_summary['completiondate'] = pd.to_datetime(df_summary['completiondate'], errors='coerce')

    tmp = df_summary[(df_summary['is_completed'] == 1) & df_summary['completiondate'].notna()].copy()
    if tmp.empty:
        print("No completed searches with valid completion dates found for monthly throughput.")
        return pd.DataFrame()

    tmp['month'] = tmp['completiondate'].dt.to_period('M').astype(str)
    try:
        out = (
            tmp.groupby([agent_col, 'month'], as_index=False)
               .size()
               .rename(columns={'size': 'verified_count'})
        )
        print("Monthly verified count calculation complete.")
        return out
    except Exception as e:
        print(f"Error during monthly verified grouping: {e}")
        return pd.DataFrame()


# daily
def daily_verified_per_agent(df_summary, agent_col='agentname'):
    """
    Returns one row per (agent, day) with # completed searches.
    """
    print(f"Calculating daily verified count per agent ({agent_col})...")
    required = {agent_col, 'completiondate', 'is_completed'}
    if not required.issubset(df_summary.columns):
        print(f"Skipping daily throughput – missing {required.difference(df_summary.columns)}")
        return pd.DataFrame()

    # Ensure types are correct before filtering/grouping
    df_summary[agent_col] = df_summary[agent_col].fillna('Unknown Agent').astype(str)
    df_summary['is_completed'] = pd.to_numeric(df_summary['is_completed'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df_summary['completiondate']):
        df_summary['completiondate'] = pd.to_datetime(df_summary['completiondate'], errors='coerce')

    tmp = df_summary[(df_summary['is_completed'] == 1) & df_summary['completiondate'].notna()].copy()
    if tmp.empty:
        print("No completed searches with valid completion dates found for daily throughput.")
        return pd.DataFrame()

    tmp['day'] = tmp['completiondate'].dt.date        # or .strftime('%Y-%m-%d')
    try:
        out = (
            tmp.groupby([agent_col, 'day'], as_index=False)
               .size()
               .rename(columns={'size': 'verified_count'})
        )
        print("Daily verified count calculation complete.")
        return out
    except Exception as e:
        print(f"Error during daily verified grouping: {e}")
        return pd.DataFrame()

# --- END Additional Analysis Functions ---


# --- Main Analysis Function ---

def perform_mb_impact_analysis(df_history):
    """
    Perform comprehensive analysis of Murphy Brown's impact using consolidated aggregations.

    Args:
        df_history: History DataFrame, pre-filtered for relevant search types.

    Returns:
        Dictionary with all analysis results
    """
    print_info("\n=== Starting Murphy Brown Impact Analysis ===", level=1)

    # --- Initialize results dictionary EARLY --- 
    all_results = {}
    # --- END INIT ---

    # --- Call identify_mb_contact_work to get agg_contacts_for_patch ---
    # This function is not part of the original script's main flow for perform_mb_impact_analysis
    # but is being added here to fulfill the user's request for patching contact counts.
    # We need to ensure it has df_history. It also returns a contact_df which we might not use here.
    print_info("Step 0: Getting agg_contacts from identify_mb_contact_work for potential patching...", level=1)
    try:
        # Assuming identify_mb_contact_work is defined in the global scope
        # Pass a copy of df_history if identify_mb_contact_work modifies it, though it shouldn't with the recent changes.
        _contact_df_temp, agg_contacts_for_patch = identify_mb_contact_work(df_history)
        if agg_contacts_for_patch.empty:
            print_info("Warning: agg_contacts_for_patch is empty. Fallback counts may not be available.", level=2)
    except Exception as e:
        print_info(f"Warning: Could not run identify_mb_contact_work to get agg_contacts_for_patch. Error: {e}", level=1)
        agg_contacts_for_patch = pd.DataFrame() # Ensure it's an empty DataFrame on error
    # --- End call to identify_mb_contact_work ---

    # --- Check essential columns early ---
    required_cols = {
        'searchid', 'historydatetime', 'agentname', 'username', 'note',
        'searchstatus', 'resultid', 'contactmethod', 'resultsentdate', # Added resultsentdate for outbound logic
        'commentid', 'attemptnumber', 'attempt_result'
    }
    if not required_cols.issubset(df_history.columns):
        missing = required_cols.difference(df_history.columns)
        print(f"Error: Missing essential columns in df_history: {missing}. Aborting analysis.")
        return None
    # --- END CHECK ---

    # --- 1. Pre-calculations and Flagging on df_history ---
    print_info("Step 1: Pre-calculating flags on history data...", level=1)

    # Ensure necessary columns have correct types
    df_history['note'] = df_history['note'].fillna('').astype(str)
    df_history['agentname'] = df_history['agentname'].fillna('').astype(str)
    df_history['username'] = df_history['username'].fillna('').astype(str)
    df_history['resultid'] = pd.to_numeric(df_history['resultid'], errors='coerce').astype('Int16')
    df_history['contactmethod'] = df_history['contactmethod'].fillna('').astype(str).str.lower()
    df_history['searchstatus'] = df_history['searchstatus'].fillna('').astype(str).str.upper()
    # Ensure historydatetime is datetime for the TAT helper
    df_history['historydatetime'] = pd.to_datetime(df_history['historydatetime'], errors='coerce')
    df_history['resultsentdate'] = pd.to_datetime(df_history['resultsentdate'], errors='coerce') # Ensure resultsentdate is datetime

    # --- STEP 0: normalise username & contactmethod once, right after load (already done above for contactmethod) ---
    df_history['username'] = df_history['username'].fillna('').str.lower()
    # contactmethod already normalized and lowercased earlier
    df_history['contactmethod'] = df_history['contactmethod'].fillna('').str.lower()


    # --- END ADDED HELPER ---

    # Cache MB agent pattern
    MB = MB_AGENT_IDENTIFIER.lower()          # keep it terse


    # --- Base Flags (before V19 specific flags) ---
    df_history['is_mb_row'] = (df_history['username'] == MB).astype('UInt8')
    df_history['is_mb_contact_research_note'] = (
        df_history['note'].str.contains(MB_CONTACT_PATTERN, na=False, case=False)
    ).astype('UInt8')
    df_history['is_mb_email_handling_note'] = (
        (df_history['username'] == MB) &
        df_history['note'].str.contains(MB_EMAIL_RECEIVED_PATTERN, na=False, case=False) # Using renamed pattern
    ).astype('UInt8')
    # ──────────────────────────────────────────────────────────────
    # ### V19 FLAG CREATION ########################################
    # Contact-plan = 20 min   | Inbound parse = 20 min   | Outbound send = 10 min
    # ----------------------------------------------------------------
    # df_history['_note_lower'] = df_history['note'].str.lower().fillna('') # _note_lower already created if needed or use .str.lower() directly

    # ------------------------------------------------------------------
    # ① MB contact-plan note  (20 min once per search)
    df_history['is_mb_contact_plan'] = (
        (df_history['username'] == MB) &
        df_history['note'].str.contains(CONTACT_PLAN_PHRASE, case=False, na=False)
    ).astype('UInt8')

    # ② MB inbound parse that closes the search (20 min once per search)
    df_history['is_mb_inbound_autoclose'] = (
        (df_history['username'] == MB) &
        (df_history['contactmethod'] == 'email') &
        (
            df_history['commentid'].isin(INB_COMMENT_IDS) |
            df_history['resultid'].isin(INB_RESULT_IDS)
        ) &
        df_history['note'].str.contains(INB_VERIFIED_PHRASE, case=False, na=False)
    ).astype('UInt8')

    # ③ MB outbound auto-send (10 min each — cap 2 per search)
    df_history['is_mb_outbound_send'] = (
        (df_history['username'] == MB) &
        df_history['contactmethod'].isin(['email','fax']) &
        df_history['resultsentdate'].notna()
    ).astype('UInt8')
    # ------------------------------------------------------------------
    # ──────────────────────────────────────────────────────────────

    # Event-specific minutes saved (on df_history)
    df_history['min_saved_inbound_event']  = np.where(df_history['is_mb_inbound_autoclose'] == 1, MIN_PER_INBOUND_PARSE,  0)
    df_history['min_saved_outbound_event'] = np.where(df_history['is_mb_outbound_send'] == 1,    MIN_PER_OUTBOUND_SEND,  0)
    df_history['min_saved_plan_event']     = np.where(df_history['is_mb_contact_plan'] == 1,     MIN_PER_CONTACT_PLAN,   0)

    # a) MB row flag (Updated to use only username)
    # df_history['is_mb_row'] = df_history['username'].str.contains(MB, na=False) # This is now created above more robustly

    # b) MB contact research flag (using regex) - This is existing, MB_CONTACT_PATTERN is used for is_mb_contact_plan
    # df_history['is_mb_contact_research_note'] = df_history['note'].str.contains(MB_CONTACT_PATTERN, na=False) # Created above

    # c) MB email handling flag (using regex) - This is existing, MB_EMAIL_PATTERN is used for is_mb_inbound_autoclose
    # df_history['is_mb_email_handling_note'] = df_history['note'].str.contains(MB_EMAIL_PATTERN, na=False) # Created above with renamed pattern



    # d) Applicant contact flag (Result ID 16)
    df_history['is_applicant_contact'] = (df_history['resultid'] == 16)

    # e) Email contact method flag
    df_history['is_email_contact_method'] = (df_history['contactmethod'] == 'email')

    # f) Completion status flag (used by TAT helper)
    df_history['is_completion_status'] = (df_history['searchstatus'] == 'REVIEW') # Kept for TAT helper

    # g) Extract contact details from MB contact research notes (Vectorized)
    print_info("Step 1b: Extracting contact details from notes (vectorized)...", level=1)
    contact_extract_pattern = r'(?i)(?:email:\s*([^\s@]+@[^\s@]+))|(?:phone:\s*([+\d\-\(\)]+))|(?:fax:\s*([+\d\-\(\)]+))'
    # Apply only to rows flagged as contact research notes
    mask_contact_notes = df_history['is_mb_contact_research_note']
    # Use extractall, which returns a multi-index (original_index, match_number)
    contacts_extracted = df_history.loc[mask_contact_notes, 'note'].str.extractall(contact_extract_pattern)

    if not contacts_extracted.empty:
        # Rename columns for clarity (match groups correspond to email, phone, fax)
        contacts_extracted.columns = ['email_match', 'phone_match', 'fax_match']
        # Count non-NA matches per original index (searchid will be linked via index)
        contact_counts_per_note = contacts_extracted.notna().groupby(level=0).sum()
        # Rename columns for aggregation
        contact_counts_per_note = contact_counts_per_note.rename(columns={
            'email_match': 'emails_in_note',
            'phone_match': 'phones_in_note',
            'fax_match': 'faxes_in_note'
        })
        # Add a total column
        contact_counts_per_note['total_contacts_in_note'] = contact_counts_per_note.sum(axis=1)

        # Join these counts back to the original df_history based on index
        df_history = df_history.join(contact_counts_per_note)
        # Fill NaNs resulting from non-matching rows with 0
        contact_count_cols = ['emails_in_note', 'phones_in_note', 'faxes_in_note', 'total_contacts_in_note']
        df_history[contact_count_cols] = df_history[contact_count_cols].fillna(0).astype(int)
    else:
        print("No contacts found matching pattern in MB contact notes.")
        # Ensure columns exist even if no contacts found
        df_history['emails_in_note'] = 0
        df_history['phones_in_note'] = 0
        df_history['faxes_in_note'] = 0
        df_history['total_contacts_in_note'] = 0

    # h) Count actual email interaction instances within notes
    df_history['email_interaction_instances'] = df_history['note'].str.count(MB_EMAIL_RECEIVED_PATTERN)
        
    # --- 1c. Calculate TAT Block using Helper --- 
    print_info("Step 1c: Calculating completion and TAT metrics using helper...", level=1)
    tat_block = add_completion_and_tat(
        df_history,
        holidays=HOLIDAYS,
        status_col="searchstatus",
        finished_status="REVIEW" # <-- Use REVIEW as the completion status
    )

    # --- 2a. Compute SQL-parity attempt metrics via helper function ---
    print_info("Step 2a: Calculating attempt count metrics via helper...", level=1)
    attempt_metrics = calculate_attempt_count_metrics(df_history.copy()) # Pass a copy to avoid side effects from _attempt_result_lower creation/deletion

    # --- 2b. Consolidated Aggregation of everything else (no attempt counts) ---
    print_info("Step 2b: Performing consolidated aggregation of flags and contacts...", level=1)
    
    # Define aggregation dictionary (REMOVED time/completion AND attempt count fields)
    agg_dict = {
        # MB Touch & Capability Flags (take max, as True > False)
        'mb_touched': ('is_mb_row', 'max'),
        'mb_contact_research': ('is_mb_contact_research_note', 'max'),
        'mb_email_handling': ('is_mb_email_handling_note', 'max'),
        
        # ── V19 AGGREGATION ─────────
        'contact_plan_provided' : ('is_mb_contact_plan', 'max'), # Changed from is_mb_contact_plan
        'mb_inbound_handled'    : ('is_mb_inbound_autoclose', 'max'), # Changed from is_mb_inbound_autoclose
        'mb_outbound_sent'      : ('is_mb_outbound_send',     'max'), # Changed from is_mb_outbound_send
        # ── END V19 AGGREGATION ─────

        # Aggregated minutes saved from history events
        'min_inbound_total_hist': ('min_saved_inbound_event', 'sum'),
        'min_outbound_total_hist': ('min_saved_outbound_event', 'sum'),
        'min_plan_total_hist': ('min_saved_plan_event', 'sum'),

        # Contact Counts (Sum counts extracted per note)
        'email_count': ('emails_in_note', 'sum'),
        'phone_count': ('phones_in_note', 'sum'),
        'fax_count': ('faxes_in_note', 'sum'),
        'total_contacts': ('total_contacts_in_note', 'sum'),
        
        # Email Metrics
        'email_interaction_count': ('email_interaction_instances', 'sum'), 
        'email_method_count': ('is_email_contact_method', 'sum'), # Count rows where method was email
        
        # --- FIX: Add agentname aggregation ---
        'agentname': ('agentname', 'first'), # Keep the first agentname associated with the searchid
        # --- END FIX ---
        # --- ADDED: Add office aggregation ---
        'office': ('office', 'first'), # Keep the first office associated with the searchid
        # --- END ADDED ---
        # --- ADDED: Add searchtype aggregation ---
        'searchtype': ('searchtype', 'first') # Keep the searchtype
        # --- END ADDED ---
    }

    # Remove aggregations if the source column doesn't exist in df_history
    # -- Ensure agentname check is included if added --
    required_agg_cols = [v[0] for v in agg_dict.values()]
    agg_dict_filtered = {k: v for k, v in agg_dict.items() if v[0] in df_history.columns}
    missing_agg_cols = set(required_agg_cols) - set(df_history.columns)
    if missing_agg_cols:
        print(f"Warning: Columns required for aggregation not found in history: {missing_agg_cols}")
    # ---------------------------------------------

    # Perform the aggregation
    df_summary = df_history.groupby('searchid').agg(**agg_dict_filtered).reset_index()
    print_info(f"Aggregation complete. Summary table shape (before attempt/TAT merge): {df_summary.shape}", level=1)
   
    # --- Drop _note_lower from df_history after aggregation ---
    # df_history.drop(columns=['_note_lower'], inplace=True, errors='ignore') # _note_lower was not added globally here
    # --- END ADDED HELPER --- (This comment seems misplaced, but instruction was to drop _note_lower here)

    # --- 2c. Merge attempt metrics back in ---
    print_info("Step 2c: Merging SQL-parity attempt metrics...", level=1)
    df_summary = df_summary.merge(
        attempt_metrics, # Contains searchid, search_attempts, applicant_contact_count, maxminusapplicant
        on='searchid',
        how='left'
    )
    print_info(f"Merged attempt metrics. Summary table shape: {df_summary.shape}", level=1)

    # --- Merge the TAT block --- (Moved after attempt metrics merge)
    df_summary = df_summary.merge(tat_block, on="searchid", how="left")
    print_info(f"Merged TAT block. Summary table shape (after TAT merge): {df_summary.shape}", level=1)

    # --- Merge the TTV block ---
    # Requires is_completion_status to be calculated in df_history (Step 1f)
    ttv_block = add_ttv(df_history, completion_flag_col='is_completion_status')
    df_summary = df_summary.merge(ttv_block, on='searchid', how='left')
    print_info(f"Merged TTV block. Summary table shape (after TTV merge): {df_summary.shape}", level=1)
    # Fill NA for ttv_hours as it's an outer join
    if 'ttv_hours' in df_summary.columns:
        df_summary['ttv_hours'] = df_summary['ttv_hours'].astype('Float64')

    # Calculate total minutes saved per search using the new method from aggregated history events
    df_summary['minutes_saved'] = df_summary['min_inbound_total_hist'] + \
                                  df_summary['min_outbound_total_hist'] + \
                                  df_summary['min_plan_total_hist']

    # --- Patch df_summary with agg_contacts_for_patch if total_contacts is 0 ---
    print_info("Step 2d: Patching contact counts in df_summary if vectorized method resulted in zero...", level=1)
    if not agg_contacts_for_patch.empty and all(col in agg_contacts_for_patch.columns for col in ['email_count', 'phone_count', 'fax_count', 'total_contacts']):
        cols_to_patch = ['email_count', 'phone_count', 'fax_count', 'total_contacts']
        # Create a temporary summary indexed by searchid for easier update
        temp_summary_for_patch = df_summary.set_index('searchid')
        
        # Identify searchids in summary where total_contacts is 0
        mask_zero_in_summary = temp_summary_for_patch['total_contacts'] == 0
        searchids_to_patch = temp_summary_for_patch[mask_zero_in_summary].index

        if not searchids_to_patch.empty:
            print_info(f"Found {len(searchids_to_patch)} searches in df_summary with 0 total_contacts from vectorized method. Attempting patch.", level=2)
            # Align agg_contacts_for_patch with the searchids that need patching
            # agg_contacts_for_patch is already indexed by searchid
            fallback_values = agg_contacts_for_patch.reindex(searchids_to_patch)[cols_to_patch].fillna(0)

            # Update the temporary summary
            for col_to_patch in cols_to_patch:
                temp_summary_for_patch.loc[searchids_to_patch, col_to_patch] = fallback_values[col_to_patch].values
            
            # Restore df_summary from the patched temporary version
            df_summary = temp_summary_for_patch.reset_index()
            print_info("Patching of contact counts complete.", level=1)
        else:
            print_info("No searches in df_summary required patching for contact counts.", level=2)
    else:
        print_info("Skipping contact count patching: agg_contacts_for_patch is empty or missing required columns.", level=2)
    # --- End Patch --- 

    # --- 3. Post-aggregation Calculations & Clean-up ---
    print_info("Step 3: Performing post-aggregation calculations...", level=1)

    # === EXTRA MB IMPACT SLICES ======================================
    extra_slices = run_extra_slices(df_summary, df_history)
    all_results.update(extra_slices)

    # optional – drop every DataFrame slice to CSV for inspection
    for name, obj in extra_slices.items():
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            save_results(obj, name)
    # =================================================================

    # a) Calculate maxminusapplicant
    # This is now redundant as it's calculated by calculate_attempt_count_metrics and merged in.
    # Keeping the check for existence, but should not recalculate.
    if 'maxminusapplicant' not in df_summary.columns:
        print("Warning: 'maxminusapplicant' not found after merging attempt_metrics. This should not happen.")
        if 'search_attempts' in df_summary.columns and 'applicant_contact_count' in df_summary.columns:
            print("Recalculating 'maxminusapplicant' as a fallback.")
            df_summary['maxminusapplicant'] = (
                df_summary['search_attempts'] - df_summary['applicant_contact_count']
            ).clip(lower=0).astype('Int64')
        else:
            df_summary['maxminusapplicant'] = pd.NA # Indicate it couldn't be calculated
    else:
        # Ensure correct dtype if it was merged
        df_summary['maxminusapplicant'] = df_summary['maxminusapplicant'].astype('Int64')

    # b) REMOVED: Completion status and date calculation (handled by add_completion_and_tat)
    # c) REMOVED: TAT metric calculation (handled by add_completion_and_tat)

    # d) Convert boolean flags (Max gives True/False) to int (1/0)
    bool_flags = [
        'mb_touched', 'mb_contact_research', 'mb_email_handling',
        'is_mb_contact_plan', 'is_mb_inbound_autoclose', 'is_mb_outbound_send'
    ]
    for flag in bool_flags:
        if flag in df_summary.columns:
            df_summary[flag] = df_summary[flag].astype(int)

    # e) Fill NA values in count columns resulting from aggregation (if any)
    count_cols = ['search_attempts', 'applicant_contact_count', 
                  'email_count', 'phone_count', 'fax_count', 'total_contacts',
                  'email_interaction_count', 'email_method_count']
    for col in count_cols:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].fillna(0).astype('Int64') # Use nullable integer
            
    # --- ADDED: Fill NA for office column --- 
    if 'office' in df_summary.columns:
        df_summary['office'] = df_summary['office'].fillna('Unknown Office').astype(str)
    # --- END ADDED ---

    # --- ADDED: Safety check for searchtype column ---
    if 'searchtype' in df_summary.columns:
        df_summary['searchtype'] = (
            df_summary['searchtype']
            .fillna('Unknown Type')
            .astype(str)
            .str.lower()  # Ensure lowercase for consistency
        )
    else:
        # This case should be less likely now that it's in agg_dict,
        # but as a fallback, create it if it's still missing.
        print("Warning: 'searchtype' column was still missing after aggregation. Initializing.")
        df_summary['searchtype'] = 'Unknown Type'
    # --- END ADDED ---

    # --- Add Fax/Phone interaction and method counts ---
    print_info("Step 3b: Adding fax/phone interaction and method counts...", level=1)
    for ch in ['fax', 'phone']:
        # Interaction counts (using helper function)
        inter = count_channel_interactions(df_history, ch)
        df_summary = df_summary.merge(inter.reset_index(), on='searchid', how='left') # Use reset_index() before merge
        interaction_col = f'{ch}_interaction_count'
        if interaction_col in df_summary.columns:
            df_summary[interaction_col] = df_summary[interaction_col].fillna(0).astype('Int64')
        else:
            df_summary[interaction_col] = 0 # Ensure column exists even if merge fails
            df_summary[interaction_col] = df_summary[interaction_col].astype('Int64')

        # Outbound method counts (direct calculation from history)
        method_col = f'{ch}_method_count'
        attempts = df_history[df_history['contactmethod'].astype(str).str.lower() == ch] \
                       .groupby('searchid').size().rename(method_col)
        df_summary = df_summary.merge(attempts.reset_index(), on='searchid', how='left') # Use reset_index() before merge
        if method_col in df_summary.columns:
            df_summary[method_col] = df_summary[method_col].fillna(0).astype('Int64')
        else:
            df_summary[method_col] = 0 # Ensure column exists
            df_summary[method_col] = df_summary[method_col].astype('Int64')

    # *** NEW: Initialize contact_plan_provided column early ***
    # df_summary['contact_plan_provided'] = False # This is now handled by the aggregation logic and type casting. And removed as per user instruction.
    # print_info("Initialized 'contact_plan_provided' column in df_summary.", level=1)
    # *********************************************************

    # --- Step 4: Calculate Downstream Metrics using df_summary --- 
    print_info("Step 4: Calculating downstream analysis metrics...", level=1)

    # a) Calculate autonomy metrics (requires df_history with flags and the new df_summary)
    autonomy_results = calculate_mb_autonomy_metrics(df_history, df_summary)
    all_results['autonomy'] = autonomy_results # Store results
    if autonomy_results and 'autonomy_df' in autonomy_results and not autonomy_results['autonomy_df'].empty:
        # Merge autonomy metrics into summary
        autonomy_cols_to_merge = ['searchid', 'human_touch_count', 'fully_autonomous', 'has_rework', 'had_fallback']
        existing_autonomy_cols = [col for col in autonomy_cols_to_merge if col in autonomy_results['autonomy_df'].columns]
        if existing_autonomy_cols:
            df_summary = pd.merge(
                 df_summary,
                 autonomy_results['autonomy_df'][existing_autonomy_cols],
                 on='searchid', how='left'
             )
            # Fill NA for autonomy metrics
            for col in existing_autonomy_cols:
                if col != 'searchid' and col in df_summary.columns and df_summary[col].isnull().any():
                    df_summary[col] = df_summary[col].fillna(0).astype('Int64') 
            # --- SAVE AUTONOMY DETAILS ---
            save_results(autonomy_results['autonomy_df'], 'mb_autonomy_details')
            # -----------------------------
        else:
            print("Warning: No autonomy columns found to merge.")
    else:
        print("Warning: Autonomy calculation failed or returned no results. Skipping merge and save.")

    # b) Calculate time efficiency metrics
    time_efficiency_results = calculate_time_efficiency_metrics(df_summary)
    all_results['time_efficiency'] = time_efficiency_results # Store results
    # ---- Print TAT stats ----
    if time_efficiency_results and 'tat_stats' in time_efficiency_results and time_efficiency_results['tat_stats'] is not None:
        print_info("\n*** ΔTAT (calendar) MB-vs-Non-MB ***", level=1)
        # print(time_efficiency_results['tat_stats'].to_markdown(index=False)) # This can be removed or kept based on preference
        print_tat_comparison(time_efficiency_results['tat_stats'])
        # --- Save TAT stats --- 
        save_results(time_efficiency_results['tat_stats'], 'tat_stats_every_run')
        # ----------------------
    # ------------------------

    # c) Calculate contact efficiency metrics
    contact_efficiency_results = calculate_contact_efficiency_metrics(df_summary)
    all_results['contact_efficiency'] = contact_efficiency_results # Store results
    # ---- Print Attempts stats ----
    if contact_efficiency_results and 'attempts_stats' in contact_efficiency_results and contact_efficiency_results['attempts_stats'] is not None:
        print_info("\n*** ΔAttempts MB-vs-Non-MB ***", level=1)
        print(contact_efficiency_results['attempts_stats'].to_markdown(index=False))
        # --- Save Attempts stats --- 
        save_results(contact_efficiency_results['attempts_stats'], 'attempts_stats_every_run')
        # ---------------------------
    # ----------------------------

    # --- Outbound Email/Fax Automation Rate (Phase 3A) ---
    print_info("Step 4d: Calculating outbound automation rate for email & fax...", level=1)
    phase_3a_metrics = {} # Initialize dict for metrics
    try:
        # 1. Filter MB outbound attempts over email or fax
        mask_out = (
            df_history['is_mb_row'] &
            df_history['contactmethod'].isin(['email','fax'])
        )
        out = (
            df_history[mask_out]
            .sort_values(['searchid','historydatetime'])
        )
        # 2. Only first 48 hours per search, then take up to 2 attempts
        if not out.empty:
            first_ts = out.groupby('searchid')['historydatetime'].first().rename('t0')
            out = out.join(first_ts, on='searchid')
            out = out.loc[out['historydatetime'] <= out['t0'] + pd.Timedelta(hours=48)]
            out = out.groupby('searchid').head(2)

            # 3. Pull first "REVIEW" (success) time per search
            resp = (
                df_history[df_history['searchstatus'] == 'REVIEW']
                .groupby('searchid', as_index=False)['historydatetime']
                .min()
                .rename(columns={'historydatetime':'resp_time'})
            )

            # 4. Build summary and flag auto-completes
            summary_out = (
                out.groupby('searchid')
                   .agg(attempts=('historydatetime','count'),
                        end_time=('historydatetime','max'))
                   .reset_index()
                   .merge(resp, on='searchid', how='left')
            )
            # Success if resp_time <= end_time of 2nd attempt
            summary_out['auto_complete'] = (
                summary_out['resp_time'] <= summary_out['end_time']
            ).fillna(False)

            # --- SAVE OUTBOUND SUMMARY DATA ---
            save_results(summary_out, 'mb_outbound_automation_summary')
            # --------------------------------

            # --- Minimal Key Metrics (Phase 3A) ---
            total = len(summary_out)
            automated = summary_out['auto_complete'].sum()

            # 1. Automation Rate
            automation_rate = automated / total if total > 0 else 0
            phase_3a_metrics['automation_rate'] = automation_rate

            # 2. Reassignment Rate
            reassignment_rate = 1 - automation_rate
            phase_3a_metrics['reassignment_rate'] = reassignment_rate

            # 3. Time to First Response (hours)
            if not out.empty:
                first_times = out.groupby('searchid')['historydatetime'].first().rename('first_time')
                resp_times  = summary_out.set_index('searchid')['resp_time']
                resp_times, first_times = resp_times.align(first_times, join='inner')
                if not resp_times.empty and not first_times.empty:
                    response_hours = (resp_times - first_times).dt.total_seconds() / 3600
                    avg_response_time = response_hours.mean()
                else:
                    print("Warning: Could not calculate response time.")
                    avg_response_time = np.nan
            else:
                print("Warning: 'out' DataFrame is empty. Cannot calculate response time.")
                avg_response_time = np.nan
            phase_3a_metrics['avg_response_time_hours'] = avg_response_time

            # 4. Channel Success Rate
            email_ids = out.loc[out['contactmethod']=='email', 'searchid'].unique()
            fax_ids   = out.loc[out['contactmethod']=='fax',   'searchid'].unique()

            email_success_rate = np.nan
            if len(email_ids) > 0:
                valid_email_ids = summary_out.set_index('searchid').index.intersection(email_ids)
                if len(valid_email_ids) > 0:
                    email_success_rate = summary_out.set_index('searchid').loc[valid_email_ids, 'auto_complete'].mean()

            fax_success_rate = np.nan
            if len(fax_ids) > 0:
                valid_fax_ids = summary_out.set_index('searchid').index.intersection(fax_ids)
                if len(valid_fax_ids) > 0:
                    fax_success_rate = summary_out.set_index('searchid').loc[valid_fax_ids, 'auto_complete'].mean()
            
            phase_3a_metrics['email_success_rate'] = email_success_rate
            phase_3a_metrics['fax_success_rate'] = fax_success_rate

            # Print all four
            def _nice(pct):
                return "n/a" if pct is np.nan else f"{pct:.1%}"

            print(f"Automation Rate:         {_nice(automation_rate)}")
            print(f"Reassignment Rate:       {_nice(reassignment_rate)}")
            print(f"Avg Time to Response:    {avg_response_time:.1f} hours")
            print(f"Email Success Rate:      {_nice(email_success_rate)}")
            print(f"Fax  Success Rate:       {_nice(fax_success_rate)}")
        else:
            print("Skipping Phase 3A automation calculation: No MB email/fax attempts found.")
            # Ensure summary_out exists for later merge attempt, even if empty
            summary_out = pd.DataFrame(columns=['searchid', 'attempts', 'end_time', 'resp_time', 'auto_complete'])
            
    except Exception as e:
        print(f"Error during Phase 3A outbound automation calculation: {e}")
        traceback.print_exc()
        # Ensure summary_out exists
        summary_out = pd.DataFrame(columns=['searchid', 'attempts', 'end_time', 'resp_time', 'auto_complete'])
        
    all_results['phase_3a_automation_summary'] = phase_3a_metrics # Store metrics dict
    # --- end of Phase 3A automation check ---

    # e) Estimate agent hours saved - REPLACED WITH DETERMINISTIC CALCULATION
    # time_savings_results = estimate_agent_hours_saved(
    #     df_summary, 
    #     autonomy_results,
    #     df_history, 
    #     time_per_touch=20 # Changed from 5 to 20
    # )
    # all_results['time_savings'] = time_savings_results # Store results

    # --- Deterministic time-savings math (User Step 3) ---
    print("Calculating deterministic time savings...")
    # MIN_PER_PLAN    = 20
    # MIN_PER_INBOUND = 20
    # MIN_PER_SEND    = 10

    # df_summary['minutes_saved'] = (
    #       df_summary['is_mb_contact_plan']   * MIN_PER_PLAN
    #     + df_summary['is_mb_inbound_autoclose'] * MIN_PER_INBOUND
    #     + df_summary['is_mb_outbound_send']  * MIN_PER_SEND
    # )
    
    # Minutes saved per search (deterministic)
    df_summary['minutes_saved'] = (
            df_summary['contact_plan_provided'] * 20 +
            df_summary['mb_inbound_handled'] * 20 +
            df_summary['mb_outbound_sent'] * 10
    )

    total_hours_saved_val = 0.0 # Initialize with a float
    fte_equivalent_val = 0.0    # Initialize with a float
    weeks_observed_val = 1.0 # Default to 1 week if calculation fails

    if 'minutes_saved' in df_summary.columns and df_summary['minutes_saved'].sum() > 0:
        total_hours_saved_val = df_summary['minutes_saved'].sum() / 60.0
        if 'historydatetime' in df_history.columns and df_history['historydatetime'].notna().any():
            min_date_hist = df_history['historydatetime'].min()
            max_date_hist = df_history['historydatetime'].max()
            if pd.notna(min_date_hist) and pd.notna(max_date_hist) and max_date_hist > min_date_hist:
                weeks_observed_val    = max(1.0, (max_date_hist - min_date_hist).days / 7.0)
                fte_equivalent_val = total_hours_saved_val / (weeks_observed_val * 40.0)
            elif pd.notna(min_date_hist) and pd.notna(max_date_hist) and max_date_hist == min_date_hist: # Single day of data
                weeks_observed_val = 1.0/7.0 # part of a week
                fte_equivalent_val = total_hours_saved_val / (weeks_observed_val * 40.0)
            else:
                print("Warning: Could not determine valid date range from df_history for FTE calculation. Using default 1 week observed.")
        else:
            print("Warning: 'historydatetime' not available or empty in df_history for FTE calculation. Using default 1 week observed.")
    elif 'minutes_saved' in df_summary.columns and df_summary['minutes_saved'].sum() <= 0:
        print_info("Total minutes saved is zero or less. Hours saved and FTE will be zero.", level=1)
        total_hours_saved_val = 0.0
        fte_equivalent_val = 0.0
        # weeks_observed_val remains default or previously calculated
    else:
        print("Warning: 'minutes_saved' column not created. Skipping FTE calculation.")
        # Values remain 0.0

    print_info(f"Total hours saved: {total_hours_saved_val:,.1f}  →  FTE ≈ {fte_equivalent_val:.2f} (based on {weeks_observed_val:.2f} weeks)", level=1)
    
    # Store in all_results, similar to how estimate_agent_hours_saved would have
    current_time_savings_results = {
        'total_hours_saved': total_hours_saved_val,
        'fte_equivalent': fte_equivalent_val,
        'weeks_observed': weeks_observed_val,
        'minutes_saved_per_search_avg': df_summary['minutes_saved'].mean() if ('minutes_saved' in df_summary.columns and df_summary['minutes_saved'].sum() > 0) else 0.0
    }
    
    mb_completed_for_touches = pd.DataFrame()
    non_mb_completed_for_touches = pd.DataFrame()

    if 'human_touch_count' in df_summary.columns and 'mb_touched' in df_summary.columns:
        mb_completed_for_touches = df_summary[df_summary['mb_touched'] == 1]
        non_mb_completed_for_touches = df_summary[df_summary['mb_touched'] == 0]

    avg_touches_mb_val = mb_completed_for_touches['human_touch_count'].mean() if not mb_completed_for_touches.empty else 0.0
    avg_touches_non_mb_val = non_mb_completed_for_touches['human_touch_count'].mean() if not non_mb_completed_for_touches.empty else 0.0
    
    touch_reduction_val = 0.0
    if pd.notna(avg_touches_non_mb_val) and pd.notna(avg_touches_mb_val): # Check for NaN before subtraction
        touch_reduction_val = avg_touches_non_mb_val - avg_touches_mb_val
    
    weekly_hours_saved_val = 0.0
    if pd.notna(total_hours_saved_val) and pd.notna(weeks_observed_val) and weeks_observed_val > 0: # Check for NaN
        weekly_hours_saved_val = total_hours_saved_val / weeks_observed_val
        
    hours_saved_per_verification_val = 0.0
    if 'minutes_saved' in df_summary and not df_summary['minutes_saved'].empty and df_summary['minutes_saved'].sum() > 0:
         hours_saved_per_verification_val = df_summary['minutes_saved'].mean() / 60.0
    
    current_time_savings_results.update({
        'avg_touches_mb'           : avg_touches_mb_val,
        'avg_touches_non_mb'       : avg_touches_non_mb_val,
        'weekly_hours_saved'       : weekly_hours_saved_val,
        'hours_saved_per_verification': hours_saved_per_verification_val,
        'touch_reduction'          : touch_reduction_val
    })
    all_results['time_savings'] = current_time_savings_results
    # --- End Deterministic time-savings math ---


    # f) Analyze impact by MB capability category
    capability_impact_results = analyze_mb_capability_impact(df_summary, extended_metrics=True)
    all_results['capability_impact'] = capability_impact_results # Store results

    # g) Calculate metrics by MB category
    category_results = calculate_mb_category_metrics(df_summary)
    all_results['category_metrics'] = category_results # Store results

    # h) Analyze MB contact plan impact
    contact_plan_results = analyze_mb_contact_plan_impact(df_history, MB_AGENT_IDENTIFIER)
    plan_flags_df = None # Initialize
    if contact_plan_results:
        all_results['mb_contact_plan_weekly'] = contact_plan_results.get('weekly_summary')
        all_results['mb_contact_plan_monthly'] = contact_plan_results.get('monthly_summary')
        if 'plan_flags' in contact_plan_results and not contact_plan_results['plan_flags'].empty:
            plan_flags_df = contact_plan_results['plan_flags'] # Assign for later use
            save_results(plan_flags_df, 'mb_contact_plan_flags')

    # --- Calculate and Save Agent Throughput --- 
    print_info("Calculating and saving agent throughput...", level=1)
    weekly_throughput = weekly_verified_per_agent(df_summary)
    monthly_throughput = monthly_verified_per_agent(df_summary) # Assumes function exists
    daily_throughput = daily_verified_per_agent(df_summary)     # Assumes function exists
    
    all_results['verified_by_agent_week'] = weekly_throughput
    all_results['verified_by_agent_month'] = monthly_throughput
    all_results['verified_by_agent_day'] = daily_throughput
    
    if weekly_throughput is not None and not weekly_throughput.empty:
        save_results(weekly_throughput, 'verified_by_agent_week')
    if monthly_throughput is not None and not monthly_throughput.empty:
        save_results(monthly_throughput, 'verified_by_agent_month')
    if daily_throughput is not None and not daily_throughput.empty:
        save_results(daily_throughput, 'verified_by_agent_day')
    # -----------------------------------------

    # i) Merge contact plan flag into df_summary (if available, overwrites initial False)
    if plan_flags_df is not None:
        # --- MODIFIED MERGE ---
        # Drop the placeholder column first to avoid suffixes _x, _y
        # df_summary = df_summary.drop(columns=['contact_plan_provided'], errors='ignore') # contact_plan_provided is now from agg_dict
        # Merge plan_flags_df which has the specific 'contact_plan_provided' from analyze_mb_contact_plan_impact
        # This specific 'contact_plan_provided' might be different from the one from V19 flags if logic differs.
        # The user's instruction for C changed 'is_mb_contact_plan' to 'contact_plan_provided' in agg_dict.
        # So df_summary already has 'contact_plan_provided'.
        # The plan_flags_df from analyze_mb_contact_plan_impact also has 'contact_plan_provided'.
        # To avoid conflict or if they represent different things, we need to be careful.
        # The user's intent for D.2 (moving time savings) was after the merge that brings in plan_flags_df.
        # This implies plan_flags_df['contact_plan_provided'] is important.
        # Let's rename the one from plan_flags_df if it's distinct.
        # Original analyze_mb_contact_plan_impact uses CONTACT_PLAN_PHRASE, which is 'contacts found from research'.
        # The V19 flag 'is_mb_contact_plan' also uses CONTACT_PLAN_PHRASE. So they should be the same.
        # The merge below will update 'contact_plan_provided' if plan_flags_df has searchids not in summary or vice-versa.
        # A left merge is usually sufficient if df_summary is the master.
        # Given the user removed the `df_summary['contact_plan_provided'] = False` and agg_dict now creates it,
        # we should ensure this merge correctly updates it based on the more detailed `plan_flags_df` if necessary.
        # However, if both flags are derived from the same logic (CONTACT_PLAN_PHRASE on MB notes), a simple merge should be fine,
        # but we need to handle potential column name clashes if not careful.
        # Let's assume the `contact_plan_provided` from `agg_dict` (derived from `is_mb_contact_plan`) is the primary one.
        # The `plan_flags_df` from `analyze_mb_contact_plan_impact` is based on `is_mb_contact_plan_row` which is the same.
        # So, the existing `contact_plan_provided` in `df_summary` should be correct.
        # The instruction to "move time calc after merge with plan_flags_df" might be for other reasons or if plan_flags_df contained other crucial info.
        # For now, I will assume the `contact_plan_provided` created by agg_dict is sufficient and consistent.
        # The original merge was:
        # df_summary = df_summary.drop(columns=['contact_plan_provided'], errors='ignore') 
        # df_summary = pd.merge(df_summary, plan_flags_df, on='searchid', how='left')
        # df_summary['contact_plan_provided'] = df_summary['contact_plan_provided'].fillna(False).astype(bool)
        # This explicitly uses plan_flags_df's version. Given section C changes agg_dict to directly create `contact_plan_provided`,
        # this merge from `plan_flags_df` might be redundant or intended to overwrite.
        # The key `contact_plan_provided` in `agg_dict` sources from `is_mb_contact_plan`.
        # `is_mb_contact_plan` is `(df_history['username'] == MB) & df_history['note'].str.contains(CONTACT_PLAN_PHRASE, case=False, na=False)`.
        # `analyze_mb_contact_plan_impact` also derives its `contact_plan_provided` from `is_mb_contact_plan_row` which is based on the same logic.
        # So, they should be identical. I will keep the merge as it was, to ensure `plan_flags_df` (if it had subtle differences or more complete searchid list for this flag) is authoritative for this specific flag.

        if plan_flags_df is not None and 'contact_plan_provided' in plan_flags_df.columns:
             if 'contact_plan_provided' in df_summary.columns:
                 df_summary = df_summary.drop(columns=['contact_plan_provided'])
             df_summary = pd.merge(df_summary, plan_flags_df[['searchid', 'contact_plan_provided']], on='searchid', how='left')
             df_summary['contact_plan_provided'] = df_summary['contact_plan_provided'].fillna(False).astype('UInt8') # Changed to UInt8
             print("Merged/updated 'contact_plan_provided' from contact_plan_results into df_summary.")
        else:
             # Ensure column exists and is UInt8 if not merged
             if 'contact_plan_provided' not in df_summary.columns:
                 df_summary['contact_plan_provided'] = False # Should be created by agg_dict
             df_summary['contact_plan_provided'] = df_summary['contact_plan_provided'].fillna(False).astype('UInt8')
             print("Warning: Contact plan flags (plan_flags_df) not available or missing 'contact_plan_provided' column. 'contact_plan_provided' in df_summary relies on initial aggregation.")

        # --- END MODIFICATION ---
        print("Merged actual contact plan flags into df_summary.") # This print might be redundant now or needs context adjustment
    else:
        # If flags couldn't be generated, the column remains as created by agg_dict.
        if 'contact_plan_provided' not in df_summary.columns: # Should exist from agg_dict
            df_summary['contact_plan_provided'] = False
        df_summary['contact_plan_provided'] = df_summary['contact_plan_provided'].fillna(False).astype('UInt8')
        print("Warning: Contact plan flags not available. 'contact_plan_provided' column relies on aggregation.")

    # j) Calculate delta metrics for searches with vs without MB contact-plan
    plan_delta = None 
    # Check if the necessary flag column now exists (it should always exist now)
    if 'contact_plan_provided' in df_summary.columns:
        required_summary_cols = ['searchid', 'is_completed', 'tat_calendar_days', 'search_attempts']
        if all(col in df_summary.columns for col in required_summary_cols):
             # Pass the potentially *empty* plan_flags_df here; delta function handles internal merge
             plan_delta = contact_plan_delta(df_summary, plan_flags_df if plan_flags_df is not None else pd.DataFrame({'searchid': [], 'contact_plan_provided': []})) 
             all_results['contact_plan_delta'] = plan_delta
             if plan_delta is not None and not plan_delta.empty:
                save_results(plan_delta, 'mb_contact_plan_delta') 
        else:
            missing_cols = [col for col in required_summary_cols if col not in df_summary.columns]
            print(f"Warning: Cannot calculate contact plan delta. Missing summary columns: {missing_cols}")
    else:
        # This case should not be reached due to early initialization
        print("Error: 'contact_plan_provided' column unexpectedly missing. Cannot calculate contact plan delta.")
        
    # k) Calculate Contact Plan Verdict (using plan_delta and df_summary)
    contact_verdict = {'overall_good': False, 'reason': 'Prerequisites not met'} # Default
    if plan_delta is not None and not plan_delta.empty and 'contact_plan_provided' in plan_delta.columns:
        try:
            print_info("Calculating contact plan verdict...", level=1)
            # Ensure required columns exist in df_summary for the t-test
            # These checks should be more robust now due to early initialization and checks within delta function
            if 'tat_calendar_days' in df_summary.columns and 'contact_plan_provided' in df_summary.columns and 'is_completed' in df_summary.columns and 'search_attempts' in df_summary.columns:
                verdict = {}
                # Use the boolean column directly from plan_delta to find the row
                row_with_mask = plan_delta['contact_plan_provided'] == True
                # Check if the 'True' group exists in the *delta* output
                if True in plan_delta['contact_plan_provided'].values:
                    row_with = plan_delta.loc[row_with_mask]
                    
                    # Check if delta calculations were successful before proceeding
                    if row_with[['delta_tat_days', 'delta_completion_rate', 'delta_attempts']].notna().all(axis=None):
                        
                        # 1. TAT must be lower by at least 1 day and p<0.05
                        tat_diff = row_with['delta_tat_days'].iat[0]
                        # Perform t-test using the main df_summary
                        # Ensure `contact_plan_provided` is treated as boolean mask (0/1 for UInt8)
                        group_with_plan = df_summary[df_summary['contact_plan_provided'] == 1]['tat_calendar_days'].dropna()
                        group_without_plan = df_summary[df_summary['contact_plan_provided'] == 0]['tat_calendar_days'].dropna()
                        
                        if len(group_with_plan) >= 2 and len(group_without_plan) >= 2:
                             t_stat, p_val = stats.ttest_ind(
                                group_with_plan,
                                group_without_plan,
                                equal_var=False, nan_policy='omit'
                             )
                             verdict['tat_ok'] = pd.notna(tat_diff) and (tat_diff < -1) and (p_val < 0.05)
                             verdict['tat_p_value'] = p_val
                        else:
                             verdict['tat_ok'] = False
                             verdict['tat_p_value'] = np.nan
                             print_info("Info: Insufficient data for contact plan TAT t-test.", level=1)

                        # 2. Completion-rate must be higher by >=3 pp
                        compl_diff = row_with['delta_completion_rate'].iat[0] * 100 # Already checked for NA
                        verdict['completion_ok'] = compl_diff > 3
                        verdict['completion_diff_pp'] = compl_diff

                        # 3. Attempts must be lower (direction only)
                        att_diff = row_with['delta_attempts'].iat[0] # Already checked for NA
                        verdict['attempts_ok'] = att_diff < 0
                        verdict['attempts_diff'] = att_diff

                        # Combine verdicts, ensuring keys exist before checking values
                        verdict['overall_good'] = all(verdict.get(k, False) for k in ['tat_ok', 'completion_ok', 'attempts_ok'])
                        verdict['reason'] = 'OK'
                    else:
                         verdict = {'overall_good': False, 'reason': 'Delta calculations in plan_delta failed (returned NaN)'}
                         print_info(f"Contact Plan Verdict: {verdict}", level=1)

                    contact_verdict = verdict # Update the main dict
                    print_info(f"Contact Plan Verdict: {contact_verdict}", level=1)
                else:
                     contact_verdict = {'overall_good': False, 'reason': 'No rows with contact_plan_provided=True in delta table'}
                     print_info(f"Contact Plan Verdict: {contact_verdict}", level=1)
            else:
                contact_verdict = {'overall_good': False, 'reason': 'Missing required columns in df_summary for verdict calculation'}
                print_info(f"Contact Plan Verdict: {contact_verdict}", level=1)
        except Exception as e:
            print(f"Error calculating contact plan verdict: {e}")
            traceback.print_exc()
            contact_verdict = {'overall_good': False, 'reason': f'Error: {e}'}
            
    all_results['contact_plan_verdict'] = contact_verdict
    save_verdict(contact_verdict, 'mb_contact_plan_verdict') # Save as JSON

    # l) Calculate Email Performance Verdict
    print_info("Calculating email performance verdict...", level=1)
    email_verdict = email_performance_verdict(df_summary)
    all_results['email_performance_verdict'] = email_verdict
    print_info(f"Email Performance Verdict: {email_verdict}", level=1)
    save_verdict(email_verdict, 'mb_email_performance_verdict') # Save as JSON

    # --- 5. Compile Results & Save/Visualize --- 
    print_info("Step 5: Compiling results and generating outputs...", level=1)
    # Results are already compiled in all_results dictionary
    
    # Add df_summary to all_results AFTER all merges are done
    all_results['df_summary'] = df_summary

    # Save key dataframes
    save_results(df_summary, "mb_analysis_summary")
    if 'mb_contact_plan_weekly' in all_results and all_results['mb_contact_plan_weekly'] is not None and not all_results['mb_contact_plan_weekly'].empty:
        save_results(all_results['mb_contact_plan_weekly'], "mb_contact_plan_weekly")
    if 'mb_contact_plan_monthly' in all_results and all_results['mb_contact_plan_monthly'] is not None and not all_results['mb_contact_plan_monthly'].empty:
        save_results(all_results['mb_contact_plan_monthly'], "mb_contact_plan_monthly")
    
    # --- SAVE STATISTICAL TEST RESULTS --- 
    if capability_impact_results and 'anova_results' in capability_impact_results and capability_impact_results['anova_results'] is not None and not capability_impact_results['anova_results'].empty:
        save_results(capability_impact_results['anova_results'], 'mb_capability_anova_results')
    
    if time_efficiency_results and 'statistical_tests' in time_efficiency_results and time_efficiency_results['statistical_tests']:
        try:
            # Convert dict to DataFrame before saving
            ttest_df = pd.DataFrame.from_dict(time_efficiency_results['statistical_tests'], orient='index')
            ttest_df.index.name = 'Metric'
            save_results(ttest_df.reset_index(), 'mb_time_efficiency_ttests')
        except Exception as e:
            print(f"Error saving time efficiency t-tests: {e}")
            
    if contact_efficiency_results and 'statistical_tests' in contact_efficiency_results and contact_efficiency_results['statistical_tests']:
        try:
            # Convert dict to DataFrame before saving
            ttest_df = pd.DataFrame.from_dict(contact_efficiency_results['statistical_tests'], orient='index')
            ttest_df.index.name = 'Metric'
            save_results(ttest_df.reset_index(), 'mb_contact_efficiency_ttests')
        except Exception as e:
            print(f"Error saving contact efficiency t-tests: {e}")
    # --- END SAVING STATS ---

    # --- NEW: Monthly completion summaries (print + single Excel workbook) ---
    print_info("Step 5b: Calculating and saving monthly completion breakdowns by searchtype to Excel...", level=1) # Modified print statement
    monthly_overall = pd.DataFrame() # Initialize empty
    monthly_office = pd.DataFrame()
    monthly_agent = pd.DataFrame()
    
    if 'is_completed' in df_summary.columns and 'completiondate' in df_summary.columns and pd.api.types.is_datetime64_any_dtype(df_summary['completiondate']) and 'searchtype' in df_summary.columns: # Added check for searchtype
        # 1) filter to completed searches and periodize
        completed = df_summary[df_summary['is_completed'] == 1].copy()
        if not completed.empty and completed['completiondate'].notna().any():
            completed['month_period'] = completed['completiondate'].dt.to_period('M').astype(str)
            # Ensure searchtype is suitable for grouping (string, handle NA)
            completed['searchtype'] = completed['searchtype'].fillna('Unknown Type').astype(str)

            # 2) overall by searchtype
            monthly_overall = (
                completed
                .groupby(['month_period', 'searchtype'])['searchid'] # Added searchtype
                .count()
                .reset_index(name='completions')
            )

            # 3) by office and searchtype (if you carried 'office' into df_summary)
            if 'office' in completed.columns:
                monthly_office = (
                    completed
                    .groupby(['month_period', 'office', 'searchtype'])['searchid'] # Added searchtype
                    .count()
                    .reset_index(name='completions')
                )
            else:
                print("Warning: 'office' column not found in summary for monthly Excel breakdown.")
                monthly_office = pd.DataFrame(columns=['month_period','office','searchtype','completions'])

            # 4) by agent and searchtype (ensure agentname exists)
            if 'agentname' in completed.columns:
                monthly_agent = (
                    completed
                    .groupby(['month_period', 'agentname', 'searchtype'])['searchid'] # Added searchtype
                    .count()
                    .reset_index(name='completions')
                )
            else:
                print("Warning: 'agentname' column not found in summary for monthly Excel breakdown.")
                monthly_agent = pd.DataFrame(columns=['month_period','agentname','searchtype','completions'])

            # 5) print to terminal
            if not monthly_overall.empty and len(monthly_overall) < 100:   # or any threshold
                print("=== Monthly Completions: Overall by SearchType ===") # Modified print statement
                print(monthly_overall.to_string(index=False))
            if not monthly_office.empty and len(monthly_office) < 100:   # or any threshold
                print("=== Monthly Completions: By Office, SearchType ===") # Modified print statement
                print(monthly_office.to_string(index=False))
            if not monthly_agent.empty and len(monthly_agent) < 100:   # or any threshold
                print("=== Monthly Completions: By Agent, SearchType ===") # Modified print statement
                print(monthly_agent.to_string(index=False))

            # 6) write to a single Excel workbook with 3 sheets
            try:
                # Determine output directory similar to save_results
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    script_name = os.path.splitext(os.path.basename(__file__))[0]
                except NameError:
                    script_dir = os.getcwd()
                    script_name = "script"
                output_dir_standard = os.path.join(script_dir, "Output", script_name)
                os.makedirs(output_dir_standard, exist_ok=True)
                
                current_time_str = datetime.now().strftime("%Y%m%d")
                excel_path = os.path.join(output_dir_standard, f"monthly_completions_{current_time_str}.xlsx")

                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    if not monthly_overall.empty:
                         monthly_overall.to_excel(writer, sheet_name='Overall', index=False)
                    if not monthly_office.empty:
                         monthly_office.to_excel(writer, sheet_name='By Office', index=False)
                    if not monthly_agent.empty:
                         monthly_agent.to_excel(writer, sheet_name='By Agent', index=False)
                print(f"Saved Excel workbook: {excel_path}")
            except Exception as e:
                 print(f"Error saving monthly completions Excel workbook to {excel_path}: {e}")
                 traceback.print_exc()
        else:
             print("No completed searches found for monthly Excel breakdown calculations.")
    else:
        missing_reqs = []
        if 'is_completed' not in df_summary.columns: missing_reqs.append('is_completed')
        if 'completiondate' not in df_summary.columns: missing_reqs.append('completiondate')
        if not pd.api.types.is_datetime64_any_dtype(df_summary.get('completiondate')): missing_reqs.append('completiondate (invalid type)') # Check type safely
        if 'searchtype' not in df_summary.columns: missing_reqs.append('searchtype') # Check for searchtype
        print(f"Warning: Missing required columns/types ({', '.join(missing_reqs)}) in summary. Skipping monthly Excel breakdown.") # Updated warning
    # --- END NEW Excel Block ---

    # --- REMOVE attempt_bucket_stats check ---
    # (The block checking for 'attempt_bucket_stats' has been removed)
    # --- END REMOVAL ---
    # --- EXISTING CODE CONTINUES BELOW ---
    # --- Detailed 15-day sample of completed EMPV searches ---------------------------
    max_hist_date = df_history['historydatetime'].max()
    if pd.notna(max_hist_date):
        end_date   = max_hist_date - pd.Timedelta(days=7)
        start_date = end_date      - pd.Timedelta(days=15)

        completed_ids = df_summary[df_summary['is_completed'] == 1]['searchid'].unique()

        df_hist_15d = df_history[
            (df_history['historydatetime'] >= start_date) &
            (df_history['historydatetime'] <= end_date) &
            (df_history['searchid'].isin(completed_ids)) &
            (df_history['searchtype'].str.lower() == 'empv')
        ].copy()

        full_sample = df_hist_15d.merge(df_summary, on='searchid', how='left')

        save_results(full_sample, 'detailed_sample_15d_fullcols')
        print_info(f"Saved EMPV-only 15-day sample: {len(full_sample)} rows.", level=1)
    else:
        print_info("Could not determine max history date – skipped 15-day sample.", level=1)
    # ---------------------------------------------------------------------------

    # Create visualizations
    dashboard_dir = create_mb_impact_dashboard(all_results)
    all_results['dashboard_dir'] = dashboard_dir

    # New console reporting additions
    print_info("\n--- Key Performance Metrics ---", level=1)
    print_metric_dashboard(all_results)

    print_info("\n", level=1) # For spacing
    print_console_executive_summary(all_results) # New console executive summary

    print_info("\n=== Murphy Brown Impact Analysis Complete ===", level=1)
    return all_results


# --- Reporting Functions ---

def generate_executive_summary(results, output_dir=None):
    """
    Generate an executive summary of the MB impact analysis.
    
    Args:
        results: Dictionary with analysis results
        output_dir: Directory to save the report
    
    Returns:
        Path to the report file
    """
    print_info("\n--- Generating Executive Summary (Markdown File) ---", level=1)
    
    if output_dir is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        output_dir = os.path.join(script_dir, "Output", "mb_impact_reports")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Format timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Get key metrics, safely checking for None
    autonomy_metrics = {}
    if 'autonomy' in results and results['autonomy'] is not None and 'summary_metrics' in results['autonomy']:
        autonomy_metrics = results['autonomy']['summary_metrics'] or {}
    
    time_savings = {}
    if 'time_savings' in results and results['time_savings'] is not None:
        time_savings = results['time_savings'] or {}
    
    time_efficiency = {}
    if 'time_efficiency' in results and results['time_efficiency'] is not None:
        time_efficiency = results['time_efficiency'] or {}
    
    contact_efficiency = {}
    if 'contact_efficiency' in results and results['contact_efficiency'] is not None:
        contact_efficiency = results['contact_efficiency'] or {}
    
    # Prepare report content
    report_content = f"""
    # Murphy Brown Impact Analysis: Executive Summary
    
    **Generated on:** {timestamp}
    
    ## Overview
    
    This report summarizes the impact of Murphy Brown (MB) on the verification process, focusing on both contact research and email handling capabilities.
    
    ## Key Findings
    
    ### Autonomy & Automation
    
    - **End-to-End Autonomy Rate:** {(lambda x: f"{x:.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")(autonomy_metrics.get('autonomy_rate', 'N/A'))}
    - **Rework Rate:** {(lambda x: f"{x:.1f}%" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")(autonomy_metrics.get('rework_rate', 'N/A'))}
    - **Avg. Human Touches:** {(lambda x: f"{x:.1f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")(autonomy_metrics.get('avg_human_touches', 'N/A'))}
    
    ### Efficiency Improvements
    
    """
    
    # Add TAT comparison if available
    if time_efficiency and 'tat_stats' in time_efficiency:
        tat_stats = time_efficiency.get('tat_stats') # Get the DataFrame
        # --- MODIFIED GUARD ---
        if tat_stats is not None and not tat_stats.empty and {'avg_tat_days','mb_touched'}.issubset(tat_stats.columns):
        # --- END GUARD ---
            # Find MB and non-MB rows safely
            mb_row = tat_stats[tat_stats['mb_touched'] == 1]
            non_mb_row = tat_stats[tat_stats['mb_touched'] == 0]
            
            # Calculate diff only if both rows exist and have data
            # --- MODIFIED GUARD for TAT diff ---
            if (not mb_row.empty and not non_mb_row.empty
                    and pd.notna(mb_row['avg_tat_days'].iat[0])
                    and pd.notna(non_mb_row['avg_tat_days'].iat[0])):
            # --- END MODIFIED GUARD ---
                mb_avg_tat = mb_row['avg_tat_days'].iloc[0]
                non_mb_avg_tat = non_mb_row['avg_tat_days'].iloc[0]
                tat_diff = mb_avg_tat - non_mb_avg_tat
                tat_perc_diff = (tat_diff / non_mb_avg_tat * 100) if non_mb_avg_tat != 0 else 0
                
                report_content += f"""
    - **Turnaround Time Reduction:** {abs(tat_diff):.1f} days ({tat_perc_diff:.1f}%)
    - **MB Avg. TAT:** {mb_avg_tat:.1f} days vs. **Non-MB:** {non_mb_avg_tat:.1f} days
"""
            else:
                report_content += """
    - **Turnaround Time Reduction:** Insufficient data for comparison
"""
        else:
            report_content += """
    - **Turnaround Time Reduction:** Data unavailable
"""
    
    # Add attempt reduction if available
    if contact_efficiency and 'attempts_stats' in contact_efficiency:
        attempt_stats = contact_efficiency.get('attempts_stats') # Get the DataFrame
        # --- MODIFIED GUARD ---
        if attempt_stats is not None and not attempt_stats.empty and {'avg_attempts', 'mb_touched'}.issubset(attempt_stats.columns):
        # --- END GUARD ---
             # Find MB and non-MB rows safely
            mb_row = attempt_stats[attempt_stats['mb_touched'] == 1]
            non_mb_row = attempt_stats[attempt_stats['mb_touched'] == 0]

            # Calculate diff only if both rows exist and have data
            # --- MODIFIED GUARD for attempt diff ---
            if (not mb_row.empty and not non_mb_row.empty
                    and pd.notna(mb_row['avg_attempts'].iat[0])
                    and pd.notna(non_mb_row['avg_attempts'].iat[0])):
            # --- END MODIFIED GUARD ---
                mb_avg_att = mb_row['avg_attempts'].iloc[0]
                non_mb_avg_att = non_mb_row['avg_attempts'].iloc[0]
                attempt_diff = mb_avg_att - non_mb_avg_att
                att_perc_diff = (attempt_diff / non_mb_avg_att * 100) if non_mb_avg_att != 0 else 0
                
                report_content += f"""
    - **Attempt Reduction:** {abs(attempt_diff):.1f} attempts ({att_perc_diff:.1f}%)
    - **MB Avg. Attempts:** {mb_avg_att:.1f} vs. **Non-MB:** {non_mb_avg_att:.1f}
"""
            else:
                report_content += """
    - **Attempt Reduction:** Insufficient data for comparison
"""
        else:
             report_content += """
    - **Attempt Reduction:** Data unavailable
"""
    
    # Add time savings if available
    if time_savings:
        report_content += f"""
    ### Resource Impact
    
    - **Hours Saved Per Verification:** {(lambda x: f"{x:.2f} hours" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")(time_savings.get('hours_saved_per_verification', 'N/A'))}
    - **Total Hours Saved:** {(lambda x: f"{x:.1f} hours" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")(time_savings.get('total_hours_saved', 'N/A'))}
    - **FTE Equivalent:** {(lambda x: f"{x:.2f} FTEs" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")(time_savings.get('fte_equivalent', 'N/A'))}
    """
    
    # Add capability comparison if available
    if 'capability_impact' in results and results['capability_impact'] is not None and 'group_counts' in results['capability_impact']:
        group_counts = results['capability_impact']['group_counts']
        
        report_content += f"""
    ### Capability Analysis
    
    MB capabilities breakdown:
    """
        
        for _, row in group_counts.iterrows():
            report_content += f"""
    - **{row['Capability Group']}:** {row['Count']} searches
    """
    
    # --- ADD THROUGHPUT SECTION --- 
    agent_weekly = results.get('verified_by_agent_week')
    if agent_weekly is not None and not agent_weekly.empty and 'week' in agent_weekly.columns and 'verified_count' in agent_weekly.columns:
        try:
            # Ensure week is sortable (string format might work, or convert to Period/Timestamp)
            agent_weekly['week_period'] = pd.to_datetime(agent_weekly['week'].str.split('/').str[0], errors='coerce')
            weekly_sum = agent_weekly.groupby('week_period')['verified_count'].sum().sort_index()
            if len(weekly_sum) > 1:
                top_lift = weekly_sum.pct_change().iloc[-1] * 100
                report_content += f"""
    ### Throughput

    - **Latest weekly completions growth:** {top_lift:+.1f}%
"""
            else:
                report_content += """
    ### Throughput

    - **Latest weekly completions growth:** Not enough data (>1 week) for trend
"""
        except Exception as e:
            print(f"Error calculating weekly throughput growth for summary: {e}")
            report_content += """
    ### Throughput

    - **Latest weekly completions growth:** Error calculating trend
"""
    # --- END THROUGHPUT SECTION ---
    
    # --- ADD MONTHLY THROUGHPUT --- 
    agent_monthly = results.get('verified_by_agent_month')
    if agent_monthly is not None and not agent_monthly.empty and 'month' in agent_monthly.columns and 'verified_count' in agent_monthly.columns:
        try:
            # Ensure month is sortable (string format like YYYY-MM should work)
            agent_monthly['month_period'] = pd.to_datetime(agent_monthly['month'], format='%Y-%m', errors='coerce')
            monthly_sum = agent_monthly.groupby('month_period')['verified_count'].sum().sort_index()
            if len(monthly_sum) > 1:
                top_lift_monthly = monthly_sum.pct_change().iloc[-1] * 100
                report_content += f"""
    - **Latest monthly completions growth:** {top_lift_monthly:+.1f}%
"""
            else:
                report_content += """
    - **Latest monthly completions growth:** Not enough data (>1 month) for trend
"""
        except Exception as e:
            print(f"Error calculating monthly throughput growth for summary: {e}")
            report_content += """
    - **Latest monthly completions growth:** Error calculating trend
"""
    # --- END MONTHLY THROUGHPUT ---
    # --- END THROUGHPUT SECTION ---
            
    # Add conclusion
    report_content += f"""
    ## Conclusion
    
    Murphy Brown demonstrates significant positive impact on verification processes, with notable improvements in turnaround time, reduction in human touches required, and substantial resource savings.
    
    The dual capability of contact research and email handling work together to streamline verifications and reduce agent workload.
    
    ## Next Steps
    
    1. Continue monitoring MB performance as volume increases
    2. Identify additional opportunity areas for expanding capabilities
    3. Track long-term impact on verification throughput and quality
    
    ---
    
    *Full analysis dashboard available at: {results.get('dashboard_dir', 'Output/mb_impact_dashboard')}/dashboard.html*
    """
    
    # Write report to file
    report_path = os.path.join(output_dir, f"mb_impact_executive_summary_{datetime.now().strftime('%Y%m%d')}.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print_info(f"Executive summary (Markdown file) saved to: {report_path}", level=1)
    return report_path


def save_verdict(verdict_dict, filename):
    """Saves a verdict dictionary to a JSON file following standard conventions."""
    if not isinstance(verdict_dict, dict):
        print(f"Skipping JSON save for '{filename}' as input is not a dictionary.")
        return

    current_time = datetime.now().strftime("%Y%m%d")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_name = os.path.splitext(os.path.basename(__file__))[0]
    except NameError:
        script_dir = os.getcwd()
        script_name = "script"

    output_dir = os.path.join(script_dir, "Output", script_name)
    os.makedirs(output_dir, exist_ok=True)
    # Use .json extension for verdict files
    output_path = os.path.join(output_dir, f"{filename}_{current_time}.json")

    try:
        with open(output_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_dict = {k: (v.item() if hasattr(v, 'item') else v) for k, v in verdict_dict.items()}
            json.dump(serializable_dict, f, indent=4)
        print(f"Saved JSON: {output_path}")
    except Exception as e:
        print(f"Error saving JSON file to {output_path}: {e}")
        traceback.print_exc()


# ------------------------------------------------------------------
# Helper: dump any verdict-dict to Output/verdicts/<stem>_YYYYMMDD.json
from pathlib import Path

def save_verdict(obj: dict, stem: str, output_dir: str | None = None) -> None:
    """Persist a verdict dictionary; silently skip if obj is falsy."""
    if not obj:
        return

    if output_dir is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:            # interactive fallback
            script_dir = os.getcwd()
        output_dir = os.path.join(script_dir, "Output", "verdicts")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{stem}_{datetime.now():%Y%m%d}.json"
    with open(Path(output_dir, fname), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
# ------------------------------------------------------------------

# --- Main Execution ---

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Murphy Brown Impact Analysis')
    parser.add_argument('--input_file', default=CSV_FILE_PATH_DEFAULT,
                      help='Path to input CSV file (default: %(default)s)')
    parser.add_argument('--output_dir', default=None,
                      help='Directory for output files (default: Output/murphy_brown_analysis)')
    
    args = parser.parse_args()
    
    # print(f"\n=== Murphy Brown Impact Analysis Script ===")
    print_info(f"Input file: {args.input_file}", level=1)
    print_info(f"Output directory: {args.output_dir or 'Output/murphy_brown_analysis'}", level=1)
    
    start_time = datetime.now()
    
    try:
        # 1. Load data
        df_history = load_history_from_csv(args.input_file)

        # --- ADDED: Filter for 'empv' and 'eduv' search types (Moved Up) ---
        original_count = len(df_history)
        if 'searchtype' in df_history.columns:
            # Use .str.lower() for case-insensitivity and handle potential NAs safely
            # Filter for 'empv' OR 'eduv'target_types = ['empv', 'eduv'] 
            target_types = ['empv', 'eduv'] # Updated to include both
            search_type_filter = df_history['searchtype'].astype(str).str.lower().isin(target_types)
            df_history = df_history.loc[search_type_filter].copy() # Use .loc and copy() to avoid SettingWithCopyWarning
            print_info(f"Filtered for {target_types} search types. Kept {len(df_history)} rows out of {original_count}.", level=1)
            if df_history.empty:
                print_info(f"No {target_types} records found after filtering. Exiting.", level=1)
                return # Exit if no matching records are left
        else:
            print_info("Warning: 'searchtype' column not found. Cannot filter by search type. Proceeding with all data.", level=1)
        # --- END ADDED FILTER ---

        # 2. Run comprehensive analysis
        # Ensure df_history is not empty before proceeding
        if not df_history.empty:
            analysis_results = perform_mb_impact_analysis(df_history)

            # 3. Generate executive summary
            if analysis_results:
                executive_summary_path = generate_executive_summary(analysis_results, args.output_dir) # Store path
                print_info(f"\nAnalysis Complete! Executive Summary (Markdown file) available at: {executive_summary_path}", level=1)

                if 'dashboard_dir' in analysis_results:
                    print_info(f"Interactive dashboard available at: {analysis_results['dashboard_dir']}/dashboard.html", level=1)

                # --- Aging Bucket Calculation & Verbosity Printouts ---
                df_summary_for_aging = analysis_results.get('df_summary')
                aging = pd.DataFrame() # Initialize aging DataFrame

                if df_summary_for_aging is not None and 'completiondate' in df_summary_for_aging.columns and 'first_attempt_time' in df_summary_for_aging.columns:
                    # Ensure 'completiondate' and 'first_attempt_time' are datetime objects
                    df_summary_for_aging['completiondate'] = pd.to_datetime(df_summary_for_aging['completiondate'], errors='coerce')
                    df_summary_for_aging['first_attempt_time'] = pd.to_datetime(df_summary_for_aging['first_attempt_time'], errors='coerce')

                    # Calculate age_hours, filtering out rows where dates couldn't be parsed
                    valid_dates_mask = df_summary_for_aging['completiondate'].notna() & df_summary_for_aging['first_attempt_time'].notna()
                    temp_df_for_aging = df_summary_for_aging[valid_dates_mask].copy() # Work on a copy

                    if not temp_df_for_aging.empty:
                        temp_df_for_aging['age_hours'] = (
                            temp_df_for_aging['completiondate'] - temp_df_for_aging['first_attempt_time']
                        ).dt.total_seconds().div(3600).round(1)

                        bucket_edges = [0, 24, 48, 72, 96, 168, np.inf]       # 0-24, 24-48, ..., ≥7 d
                        bucket_lbls  = ["<1 d","1-2 d","2-3 d","3-4 d","4-7 d","≥7 d"]
                        temp_df_for_aging['age_bucket'] = pd.cut(temp_df_for_aging['age_hours'],
                                                              bucket_edges, labels=bucket_lbls, right=False)

                        temp_df_for_aging['week'] = temp_df_for_aging['completiondate'].dt.to_period('W-MON').dt.start_time # Use W-MON to align with other weekly reports
                        
                        if not temp_df_for_aging.empty and 'age_bucket' in temp_df_for_aging.columns:
                            aging = (temp_df_for_aging.groupby(['week','age_bucket'], observed=False) # observed=False to include all categories
                                   .size()
                                   .reset_index(name='count')
                                   .pivot(index='week', columns='age_bucket', values='count')
                                   .fillna(0).astype(int))
                            print_info("Aging bucket calculation complete.", level=2)
                        else:
                            print_info("Warning: Aging bucket calculation resulted in an empty DataFrame or 'age_bucket' column missing.", level=1)
                    else:
                        print_info("Warning: No valid date data for aging bucket calculation after filtering NaT.", level=1)
                else:
                    print_info("Warning: df_summary or required date columns not found in analysis_results for aging buckets.", level=1)


                if VERBOSITY_LEVEL >= 2:
                    if df_summary_for_aging is not None:
                        minutes_cols = ['min_inbound_total_hist','min_outbound_total_hist','min_plan_total_hist','minutes_saved']
                        if all(col in df_summary_for_aging.columns for col in minutes_cols):
                            by_type = df_summary_for_aging[minutes_cols].sum()
                            # Rename for better display
                            by_type = by_type.rename({
                                'min_inbound_total_hist': 'Inbound Event Mins',
                                'min_outbound_total_hist': 'Outbound Event Mins',
                                'min_plan_total_hist': 'Plan Event Mins',
                                'minutes_saved': 'Total Minutes Saved'
                            })
                            table_from_series(by_type, "⏱  Minutes Saved (Detail)")
                        else:
                            print_info("Warning: Not all minutes saved columns present in df_summary for verbosity level 2 printout.", level=2)

                    if not aging.empty:
                        last_week_data = aging.index.max()
                        if pd.notna(last_week_data):
                             table_from_series(aging.loc[last_week_data],
                                              f"Aging Buckets – week of {last_week_data:%Y-%m-%d}")
                        else:
                            print_info("Warning: Could not determine last week's data for aging buckets printout.", level=2)
                    else:
                        print_info("Warning: Aging data is empty, skipping last week's aging bucket printout.", level=2)


                if VERBOSITY_LEVEL >= 3:
                    if not aging.empty:
                        table_from_df(aging.tail(8), "Aging Buckets – last 8 weeks")
                    else:
                        print_info("Warning: Aging data is empty, skipping last 8 weeks aging bucket printout.", level=3)

                if VERBOSITY_LEVEL >= 4:
                    if df_history is not None and not df_history.empty:
                        print_info("\\n--- df_history head (debug) ---", level=4)
                        if RICH_AVAILABLE and console:
                            console.print(df_history.head(20)) # Rich can print DataFrames directly
                        else:
                            print(df_history.head(20).to_markdown())
                    else:
                        print_info("Warning: df_history is empty or None, skipping debug printout.", level=4)
                # --- End Aging Bucket & Verbosity Printouts ---

        else:
            print_info("Skipping analysis as the filtered DataFrame is empty.", level=1)

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
    
    end_time = datetime.now()
    print_info(f"\nTotal execution time: {end_time - start_time}", level=1)


if __name__ == "__main__":
    main()