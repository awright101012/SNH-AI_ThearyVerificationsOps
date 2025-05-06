#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
murphy_brown_impact_refactored.py (v9.1 - Patched)

Refactored analysis script to quantify Murphy Brown's impact on the verification process,
based on the v8 script and the provided refactoring blueprint. Includes patches based
on code review.

Combines data loading, augmentation, analysis, and reporting into a streamlined pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
import traceback # Add this import
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union # Added Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
from matplotlib.container import BarContainer   # NEW

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")
warnings.filterwarnings("ignore", message=".*Desired columns not found.*")
warnings.filterwarnings("ignore", message=".*Some values in date column.*")
pd.options.mode.chained_assignment = None

# ───────────────────────── 0. CONFIGURATION ──────────────────────────

@dataclass
class Config:
    """Configuration settings for the analysis."""
    # --- File Paths & Identifiers ---
    csv_default: str = r"C:\Users\awrig\Downloads\Fullo3Query.csv" # Default input
    out_dir: str = "Output/mb_analysis_refactored" # Default output sub-directory
    mb_agent_identifier: str = "murphy.brown" # Case-insensitive agent name tag

    # --- Analysis Parameters ---
    search_types_to_include: Tuple[str, ...] = ("empv", "eduv") # Filter for these search types
    completion_status: str = "REVIEW" # Status indicating completion
    time_per_touch_min: int = 20 # Estimated minutes saved per human touch avoided
    significance_level: float = 0.05 # Alpha for statistical tests
    contact_plan_tat_reduction_days: int = 1 # Min TAT reduction for contact plan verdict (absolute value)
    contact_plan_completion_increase_pp: int = 3 # Min completion % points increase for contact plan verdict
    email_verdict_min_yield_increase_pp: int = 5 # Min email yield % points increase for email verdict
    agent_uplift_min_searches: int = 5 # Min searches per agent group for uplift analysis
    queue_depth_sla_days: int = 5 # Business days threshold for old queue items

    # --- Patterns for Note Analysis ---
    mb_email_pattern: str = r"<div style=\"color:red\">Email received from"
    mb_contact_pattern: str = r"Contacts found from research \(ranked by most viable first\)"
    contact_plan_phrase: str = "contacts found" # Phrase indicating a contact plan note
    contact_type_patterns: Dict[str, str] = field(default_factory=lambda: {
        'email': r"Email:\s*([^\s@]+@[^\s@]+)", # Added \s*
        'phone': r"Phone:\s*([+\d\-\(\)]+)", # Added \s*
        'fax': r"Fax:\s*([+\d\-\(\)]+)" # Added \s*
    })
    # Combined pattern for extracting all contact types at once
    contact_extract_pattern: str = r'(?i)(?:email:\s*([^\s@]+@[^\s@]+))|(?:phone:\s*([+\d\-\(\)]+))|(?:fax:\s*([+\d\-\(\)]+))'

    # --- US Federal Holidays (Example: 2024-2025) ---
    holidays: np.ndarray = field(default_factory=lambda: np.array([
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19",
        "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28",
        "2024-12-25", "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11",
        "2025-11-27", "2025-12-25"
    ], dtype="datetime64[D]"))

    # --- Plotting Configuration ---
    plot_style: str = 'seaborn-v0_8-darkgrid'
    plot_figsize: Tuple[int, int] = (10, 6)
    plot_dpi: int = 300


# Instantiate the config
cfg = Config()

# ─────────────────── 1. UTILITY FUNCTIONS (≈ 70 lines) ─────────────

def persist(obj: Union[pd.DataFrame, Dict, plt.Figure], stem: str, kind: str | None = None, output_dir: str = cfg.out_dir):
    """
    Persist `obj` under output_dir / script_name / stem_YYYYMMDD.*
    kind picks {'csv','json','xlsx','png'}. If None: auto-detect obj type.
    Handles DataFrame, dict (for JSON/Excel), or Matplotlib figure.
    """
    # Auto-detect kind if not provided
    if kind is None:
        if isinstance(obj, pd.DataFrame):
            kind = "csv"
        elif isinstance(obj, dict):
            kind = "json" # Default for dict, Excel needs explicit kind='xlsx'
        elif isinstance(obj, plt.Figure): # Check specifically for Figure object
             kind = "png"
        elif hasattr(obj, 'savefig'): # Fallback check for plt module (less ideal)
             print("Warning: Persisting the current Matplotlib figure (plt module passed instead of Figure object).")
             kind = "png"
        else:
            raise ValueError(f"Cannot auto-detect persistence kind for type: {type(obj)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped filename
    ts = datetime.now().strftime("%Y%m%d")
    # Use the script name as part of the path structure
    try:
        # Extract script name without extension
        script_name = os.path.splitext(os.path.basename(__file__))[0]
    except NameError: # Fallback for interactive sessions
        script_name = "refactored_analysis" # Or derive from class name if needed

    # Construct the full output path within the standard structure
    output_subdir = os.path.join(output_dir, script_name)
    os.makedirs(output_subdir, exist_ok=True) # Create script-specific subdir
    path = os.path.join(output_subdir, f"{stem}_{ts}.{kind}")

    # Persist based on kind
    try:
        if kind == "csv":
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(path, index=False)
            else:
                raise TypeError(f"Object must be a DataFrame for kind='csv', got {type(obj)}")
        elif kind == "json":
            if isinstance(obj, dict):
                # Convert numpy types and handle nested DataFrames safely
                def default_serializer(o):
                    # numpy scalars
                    if isinstance(o, (np.integer, np.floating)):
                        return o.item()
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                    # pandas Timestamp / Period
                    if isinstance(o, (pd.Timestamp, pd.Period)):
                        return str(o)

                    # DataFrame handling  ── this is where the crash happened
                    if isinstance(o, pd.DataFrame):
                        df_copy = o.copy()

                        # Convert datetime-like columns to string
                        # Use specific known types rather than broad 'datetime'
                        dt_cols = df_copy.select_dtypes(
                            include=["datetime64[ns]", "datetimetz"]
                        ).columns
                        # ‘period' dtype vanished in pandas ≥2.2; guard with try/except
                        try:
                            per_cols = df_copy.select_dtypes(include=["period"]).columns
                        except NotImplementedError:
                            per_cols = [] # Period dtype alias not supported

                        # Combine columns and convert to string
                        for col in list(dt_cols) + list(per_cols):
                            df_copy[col] = df_copy[col].astype(str)

                        # nullable numeric → object, keep None where NA
                        for col in df_copy.select_dtypes(include=["Int64", "Float64"]).columns:
                            df_copy[col] = (
                                df_copy[col]
                                .astype(object)
                                .where(df_copy[col].notna(), None)
                            )

                        return df_copy.to_dict(orient="records")

                    # pd.NA / NaT handling
                    if pd.isna(o):
                        return None

                    # fallback to string representation for other types
                    # This avoids potential errors with default JSONEncoder behavior
                    return str(o)

                with open(path, "w") as f:
                    json.dump(obj, f, indent=4, default=default_serializer)
            else:
                raise TypeError(f"Object must be a dict for kind='json', got {type(obj)}")
        elif kind == "xlsx":
            if isinstance(obj, dict): # Expects a dict of {sheet_name: DataFrame}
                with pd.ExcelWriter(path, engine='xlsxwriter') as xls:
                    for sheet, df_sheet in obj.items():
                        if isinstance(df_sheet, pd.DataFrame):
                            # Handle potential timezone issues before writing
                            df_copy = df_sheet.copy()
                            for col in df_copy.select_dtypes(include=['datetime64[ns, UTC]', 'datetimetz']).columns:
                                 df_copy[col] = df_copy[col].dt.tz_localize(None)
                            df_copy.to_excel(xls, sheet_name=sheet, index=False)
                        else:
                            print(f"Warning: Skipping sheet '{sheet}' in Excel output, expected DataFrame, got {type(df_sheet)}")
            elif isinstance(obj, pd.DataFrame): # Allow saving a single DataFrame to Excel
                 with pd.ExcelWriter(path, engine='xlsxwriter') as xls:
                     # Handle potential timezone issues
                     df_copy = obj.copy()
                     for col in df_copy.select_dtypes(include=['datetime64[ns, UTC]', 'datetimetz']).columns:
                          df_copy[col] = df_copy[col].dt.tz_localize(None)
                     df_copy.to_excel(xls, sheet_name='Sheet1', index=False)
            else:
                raise TypeError(f"Object must be a dict or DataFrame for kind='xlsx', got {type(obj)}")
        elif kind == "png":
            # PATCH: Stricter check for Figure object
            if isinstance(obj, plt.Figure):
                 obj.savefig(path, dpi=cfg.plot_dpi, bbox_inches="tight")
            elif hasattr(obj, 'savefig'): # Handle plt module case with warning
                 print("Warning: Saving current Matplotlib figure (plt module passed instead of Figure object).")
                 obj.savefig(path, dpi=cfg.plot_dpi, bbox_inches="tight")
            else:
                 raise TypeError(f"Object must be a Matplotlib figure or module for kind='png', got {type(obj)}")
        else:
            raise ValueError(f"Unknown persist kind: {kind}")

        print(f"Saved {kind.upper()}: {path}")
        return path

    except Exception as e:
        print(f"Error saving {kind.upper()} file to {path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def welch_ttest(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    """
    Performs Welch's t-test for independent samples with unequal variances.
    Returns (t-statistic, p-value), handling NaNs and insufficient data.
    Returns (None, None) if test cannot be performed.
    """
    x_clean = x.dropna()
    y_clean = y.dropna()

    # Check if sufficient data exists in both groups
    if len(x_clean) < 2 or len(y_clean) < 2:
        # print(f"Info: Insufficient data for Welch t-test (Group 1: {len(x_clean)}, Group 2: {len(y_clean)}). Need at least 2 in each.")
        return None, None

    # Check for zero variance in either group (prevents division by zero)
    # Use a small tolerance for floating point comparisons
    var_x = x_clean.var()
    var_y = y_clean.var()
    if np.isclose(var_x, 0, atol=1e-10) or np.isclose(var_y, 0, atol=1e-10):
         # Check if means are also close - if so, p=1, t=0, otherwise cannot test reliably
         if np.isclose(x_clean.mean(), y_clean.mean(), atol=1e-8):
             return 0.0, 1.0
         else:
             # print(f"Info: Skipping Welch t-test due to zero variance in one group with different means.")
             return None, None # Cannot reliably perform test

    try:
        # Use nan_policy='omit' just in case, though dropna() should handle it
        t, p = stats.ttest_ind(x_clean, y_clean, equal_var=False, nan_policy='omit')
        # Check for NaN results which can happen in edge cases
        if pd.isna(t) or pd.isna(p):
            # print("Info: Welch t-test resulted in NaN.")
            return None, None
        return t, p
    except Exception as e:
        print(f"Error during Welch t-test: {e}")
        return None, None


def business_days_inclusive(start_dates: pd.Series, end_dates: pd.Series, holidays: np.ndarray = cfg.holidays) -> pd.Series:
    """
    Calculates the number of inclusive business days between two Series of dates.
    Handles NaNs and ensures dates are timezone-naive before calculation.
    Returns a Series of integers (Int64 to allow NaNs).
    """
    # Ensure inputs are datetime objects
    start = pd.to_datetime(start_dates, errors='coerce')
    end = pd.to_datetime(end_dates, errors='coerce')

    # Create a mask for valid (non-NaN) date pairs where end >= start
    valid_mask = start.notna() & end.notna() & (end.dt.normalize() >= start.dt.normalize())

    # Initialize result Series with NaNs (using nullable Int64)
    results = pd.Series(pd.NA, index=start.index, dtype='Int64')

    if valid_mask.any():
        # Work with valid dates only
        valid_start = start[valid_mask]
        valid_end = end[valid_mask]

        # Make dates timezone-naive if they are aware (only if necessary)
        if valid_start.dt.tz is not None:
            valid_start = valid_start.dt.tz_localize(None)
        if valid_end.dt.tz is not None:
            valid_end = valid_end.dt.tz_localize(None)

        # Normalize to date level and convert to numpy datetime64[D]
        start_np = valid_start.dt.normalize().values.astype("datetime64[D]")
        # Add 1 day to end date for inclusivity *after* normalizing
        end_np = (valid_end.dt.normalize() + pd.Timedelta(days=1)).values.astype("datetime64[D]")

        # Calculate business days using numpy
        try:
            bus_days_count = np.busday_count(start_np, end_np, holidays=holidays)
            # Assign results back to the original index positions
            results.loc[valid_mask] = bus_days_count
        except Exception as e:
            print(f"Error during np.busday_count calculation: {e}")
            # Keep results as NA for affected rows

    return results


def create_barplot(df: pd.DataFrame, x: str, y: str, hue: str | None = None,
                   title: str = "", xlabel: str | None = None, ylabel: str | None = None,
                   fname: str = "barplot", add_labels: bool = True, label_fmt: str = "{:.1f}"):
    """
    Generates and saves a bar plot using Matplotlib. Handles optional hue grouping.
    """
    plt.style.use(cfg.plot_style)
    fig, ax = plt.subplots(figsize=cfg.plot_figsize)

    try:
        # PATCH: Handle hue argument - pivot data if hue is provided
        if hue:
            # Pivot the table to have hue categories as columns
            try:
                plot_data = df.pivot(index=x, columns=hue, values=y)
                plot_data.plot(kind='bar', ax=ax, legend=True)
                ax.legend(title=hue.replace('_', ' ').title()) # Add legend title
            except Exception as pivot_err:
                 print(f"Error pivoting data for bar plot '{title}' with hue '{hue}': {pivot_err}")
                 print("Falling back to plotting without hue.")
                 # Fallback: plot without hue if pivot fails
                 df[[x, y]].set_index(x).plot(kind='bar', ax=ax, legend=False)
        else:
            # Original behavior: plot directly if no hue
            # Ensure x is the index for direct plotting
            df_plot = df[[x, y]].set_index(x)
            df_plot.plot(kind='bar', ax=ax, legend=False)


        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel if xlabel else x.replace('_', ' ').title())
        ax.set_ylabel(ylabel if ylabel else y.replace('_', ' ').title())
        # Adjust rotation based on number of bars potentially
        num_bars = len(df[x].unique())
        rotation = 45 if num_bars > 5 else 0
        ha = 'right' if rotation > 0 else 'center'
        ax.tick_params(axis='x', rotation=rotation)
        for lbl in ax.get_xticklabels():
            lbl.set_ha(ha)

        if add_labels:
            try:
                for container in ax.containers:
                    # Check container type before applying labels
                    if isinstance(container, BarContainer):
                        ax.bar_label(container, fmt=label_fmt, fontsize=9, padding=3)
            except Exception as label_err:
                 print(f"Warning: Could not add labels to bar plot '{title}': {label_err}")


        fig.tight_layout()
        persist(fig, fname, "png") # Use the persist helper

    except Exception as e:
        print(f"Error creating bar plot '{title}': {e}")
    finally:
        plt.close(fig) # Ensure figure is closed


# ─────────────────── 2. ANALYSIS PIPELINE CLASS (≈ 500 lines) ─────────────

class MBAnalysis:
    """Encapsulates the Murphy Brown impact analysis pipeline."""

    def __init__(self, csv_path: str):
        """Initialize with the path to the input CSV."""
        print(f"\n--- Initializing MBAnalysis for: {csv_path} ---")
        self.csv_path = csv_path
        self.df_hist: pd.DataFrame | None = None # Raw-ish history data
        self.df_sum: pd.DataFrame | None = None # Aggregated summary data by searchid
        self.results: Dict[str, Any] = {} # Dictionary to store results (metrics, verdicts, small dfs)
        self.plot_configs: List[Dict[str, Any]] = [] # Store plot parameters

    # --------------- 1. Ingestion Step ------------------
    def ingest(self) -> MBAnalysis:
        """Loads data, applies initial filtering and basic flag calculations."""
        print("--- Step 1: Ingesting and Pre-processing History Data ---")
        # Define dtypes for efficient loading (using nullable types where appropriate)
        # Basic types handled by read_csv, dates handled separately
        read_csv_dtypes = {
            'searchid': 'Int64', 'historyid': 'Int64', 'resultid': 'Int16',
            'userid': 'string', 'username': 'string', 'agentname': 'string',
            'note': 'string', 'searchstatus': 'string', 'contactmethod': 'string',
            'searchtype': 'string', 'agentoffice': 'string', # Added agentOffice, removed office
            # Date columns will be parsed later
        }
        date_cols = ['historydatetime', 'completiondate', 'postdate', 'resultsentdate',
                     'last_update', 'last_attempt', 'qadate', 'first_attempt']

        try:
            # Load data, letting pandas infer missing columns
            print(f"Loading CSV: {self.csv_path}")
            df = pd.read_csv(
                self.csv_path,
                dtype=read_csv_dtypes, # Apply basic types
                parse_dates=False, # Parse dates manually for robustness
                low_memory=False,
                # PATCH: Removed on_bad_lines='warn' for broader pandas compatibility
                # If bad lines are expected, consider adding error handling or pre-cleaning
            )
            print(f"Initial load: {len(df)} rows, {len(df.columns)} columns.")

            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()
            # Rename 'officename' or 'office' to 'agentoffice' if present
            rename_map = {}
            if 'officename' in df.columns:
                rename_map['officename'] = 'agentoffice'
            elif 'office' in df.columns: # Handle if 'office' exists directly
                 rename_map['office'] = 'agentoffice'
            if rename_map:
                 df = df.rename(columns=rename_map)


            # Identify available columns vs missing desired columns
            all_desired_cols = list(read_csv_dtypes.keys()) + date_cols
            available_cols = [col for col in all_desired_cols if col in df.columns]
            missing_desired = [col for col in all_desired_cols if col not in df.columns]
            if missing_desired:
                print(f"Warning: Desired columns not found in CSV: {missing_desired}")

            # Keep only available desired columns
            df = df[available_cols].copy()

            # Parse date columns robustly
            print("Parsing date columns...")
            for col in date_cols:
                if col in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if df[col].isnull().any():
                             print(f"Warning: Some values in date column '{col}' failed parsing.")

            # Filter by search type
            if 'searchtype' in df.columns:
                original_count = len(df)
                df = df[df['searchtype'].astype(str).str.lower().isin(cfg.search_types_to_include)]
                print(f"Filtered for search types {cfg.search_types_to_include}. Kept {len(df)} of {original_count} rows.")
                if df.empty:
                    raise ValueError(f"No data remaining after filtering for search types: {cfg.search_types_to_include}")
            else:
                print("Warning: 'searchtype' column not found. Cannot filter by type.")

            # --- Basic Flag Calculations (on history) ---
            print("Calculating initial flags on history data...")
            # Ensure source columns are strings before using .str methods
            df['agentname'] = df['agentname'].fillna('').astype(str)
            df['username'] = df['username'].fillna('').astype(str)
            df['note'] = df['note'].fillna('').astype(str)
            df['searchstatus'] = df['searchstatus'].fillna('').astype(str)
            df['contactmethod'] = df['contactmethod'].fillna('').astype(str)
            df['resultid'] = pd.to_numeric(df['resultid'], errors='coerce') # Ensure numeric

            # MB row flag
            mb_pattern = cfg.mb_agent_identifier.lower()
            df['is_mb_row'] = (
                df['agentname'].str.lower().str.contains(mb_pattern, na=False) |
                df['username'].str.lower().str.contains(mb_pattern, na=False)
            )

            # Completion flag (for TAT calculation later)
            df['is_completion_status'] = (df['searchstatus'].str.upper() == cfg.completion_status)

            # Capability flags (from notes)
            df['is_mb_contact_research_note'] = df['note'].str.contains(cfg.mb_contact_pattern, na=False, regex=True)
            df['is_mb_email_handling_note'] = df['note'].str.contains(cfg.mb_email_pattern, na=False, regex=True)
            df['is_mb_contact_plan_note'] = (
                df['is_mb_row'] & # Must be MB user
                df['note'].str.lower().str.contains(cfg.contact_plan_phrase, na=False)
            )

            # Other flags for aggregation
            df['is_applicant_contact'] = (df['resultid'] == 16)
            df['is_email_contact_method'] = (df['contactmethod'].str.lower() == 'email')

            # Extract contact counts from contact research notes
            print("Extracting contact counts from notes...")
            contacts_extracted = df.loc[df['is_mb_contact_research_note'], 'note'].str.extractall(cfg.contact_extract_pattern)
            if not contacts_extracted.empty:
                contacts_extracted.columns = ['email_match', 'phone_match', 'fax_match']
                contact_counts_per_note = contacts_extracted.notna().groupby(level=0).sum()
                contact_counts_per_note = contact_counts_per_note.rename(columns={
                    'email_match': 'emails_in_note', 'phone_match': 'phones_in_note', 'fax_match': 'faxes_in_note'
                })
                contact_counts_per_note['total_contacts_in_note'] = contact_counts_per_note.sum(axis=1)
                df = df.join(contact_counts_per_note)
            # Ensure columns exist even if no contacts found, fill NaNs
            contact_count_cols = ['emails_in_note', 'phones_in_note', 'faxes_in_note', 'total_contacts_in_note']
            for col in contact_count_cols:
                if col not in df.columns:
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(0)
            df[contact_count_cols] = df[contact_count_cols].astype(int)

            # Count email interaction instances
            df['email_interaction_instances'] = df['note'].str.count(cfg.mb_email_pattern)

            # Store the processed history DataFrame
            self.df_hist = df
            print(f"Ingestion complete. History DataFrame shape: {self.df_hist.shape}")

        except FileNotFoundError:
            print(f"Error: Input file not found at {self.csv_path}")
            raise
        except ValueError as ve:
             print(f"ValueError during ingestion: {ve}")
             raise
        except Exception as e:
            print(f"An unexpected error occurred during ingestion: {e}")
            import traceback
            traceback.print_exc()
            raise

        return self

    # --------------- 2. Augmentation Step ------------------
    def augment(self) -> MBAnalysis:
        """Aggregates history data by searchid and calculates summary metrics."""
        if self.df_hist is None:
            raise ValueError("History data not loaded. Run ingest() first.")

        print("--- Step 2: Augmenting Data - Aggregating by Search ID ---")
        h = self.df_hist # Alias for brevity

        # --- Calculate TAT Block ---
        print("Calculating TAT block...")
        # Find first attempt time and last overall time
        tat_times = h.groupby("searchid")["historydatetime"].agg(
            first_attempt_time="min",
            last_attempt_time="max" # Keep last overall time for fallback
        ).reset_index()

        # Find the latest timestamp where status indicates completion
        comp_dates = (
            h[h['is_completion_status']] # Use pre-calculated flag
            .groupby("searchid")["historydatetime"]
            .max()
            .rename("completiondate")
        )

        # Merge completion date, fallback to last attempt if no completion status found
        tat_block = tat_times.merge(comp_dates, on="searchid", how="left")
        mask_na = tat_block["completiondate"].isna()
        tat_block.loc[mask_na, "completiondate"] = tat_block.loc[mask_na, "last_attempt_time"]

        # Calculate completion flag and TATs
        tat_block["is_completed"] = tat_block["completiondate"].notna().astype(int)
        tat_block["tat_calendar_days"] = (
            tat_block["completiondate"].dt.normalize() - tat_block["first_attempt_time"].dt.normalize()
        ).dt.days + 1
        # Set TAT to NaN if not completed or if negative (data issue)
        tat_block.loc[(tat_block["is_completed"] == 0) | (tat_block["tat_calendar_days"] < 0), "tat_calendar_days"] = pd.NA
        tat_block["tat_calendar_days"] = tat_block["tat_calendar_days"].astype('Int64')

        # Calculate business days TAT using the helper
        # PATCH: Ensure result is nullable Int64 series
        bus_days_result = business_days_inclusive(
            tat_block["first_attempt_time"],
            tat_block["completiondate"] # Use completiondate (which might be last_attempt_time if not completed)
        )
        tat_block["tat_business_days"] = pd.Series(bus_days_result, dtype="Int64")

        # Ensure business days TAT is NA if not completed
        tat_block.loc[tat_block["is_completed"] == 0, "tat_business_days"] = pd.NA

        # Select final TAT columns
        tat_block = tat_block[[
            "searchid", "first_attempt_time", "completiondate", "is_completed",
            "tat_calendar_days", "tat_business_days"
        ]]

        # --- Define Main Aggregation Dictionary ---
        print("Defining aggregation rules...")
        agg_dict = {
            # Attempt Counts
            'attempts_by_row': ('historyid', 'size'),
            'distinct_applicant_contact_count': ('is_applicant_contact', 'sum'),
            # MB Touch & Capability Flags (take max, as True > False)
            'mb_touched': ('is_mb_row', 'max'),
            'mb_contact_research': ('is_mb_contact_research_note', 'max'),
            'mb_email_handling': ('is_mb_email_handling_note', 'max'),
            'contact_plan_provided': ('is_mb_contact_plan_note', 'max'),
            # Contact Counts (Sum counts extracted per note)
            'email_count': ('emails_in_note', 'sum'),
            'phone_count': ('phones_in_note', 'sum'),
            'fax_count': ('faxes_in_note', 'sum'),
            'total_contacts': ('total_contacts_in_note', 'sum'),
            # Email Metrics
            'email_interaction_count': ('email_interaction_instances', 'sum'),
            'email_method_count': ('is_email_contact_method', 'sum'),
            # Agent/Office Info (take first non-null value)
            'agentname': ('agentname', 'first'),
            'office': ('office', 'first'),
            # Search Type (take first non-null value)
            'searchtype': ('searchtype', 'first'),
        }

        # Filter aggregations based on available columns in df_hist
        agg_dict_filtered = {k: v for k, v in agg_dict.items() if v[0] in h.columns}
        missing_agg_cols = set(v[0] for v in agg_dict.values()) - set(h.columns)
        if missing_agg_cols:
            print(f"Warning: Columns missing for aggregation: {missing_agg_cols}")

        # --- Perform Aggregation ---
        print("Performing aggregation by searchid...")
        try:
            df_agg = h.groupby("searchid").agg(**agg_dict_filtered).reset_index()
        except Exception as e:
             print(f"Error during main aggregation: {e}")
             raise

        # --- Merge TAT Block ---
        print("Merging TAT block...")
        self.df_sum = df_agg.merge(tat_block, on="searchid", how="left")

        # --- Post-aggregation Calculations & Cleanup ---
        print("Performing post-aggregation calculations...")
        s = self.df_sum # Alias

        # Calculate maxminusapplicant
        if 'attempts_by_row' in s.columns and 'distinct_applicant_contact_count' in s.columns:
            # Ensure operands are numeric before subtraction
            att = pd.to_numeric(s['attempts_by_row'], errors='coerce').fillna(0)
            app = pd.to_numeric(s['distinct_applicant_contact_count'], errors='coerce').fillna(0)
            s['maxminusapplicant'] = (att - app).clip(lower=0).astype('Int64')
        else:
            s['maxminusapplicant'] = pd.NA

        # Convert boolean flags (Max gives True/False) to int (1/0)
        bool_flags = ['mb_touched', 'mb_contact_research', 'mb_email_handling', 'contact_plan_provided']
        for flag in bool_flags:
            if flag in s.columns:
                s[flag] = s[flag].fillna(False).astype(int) # Fill NA before converting

        # Fill NA values in count columns resulting from aggregation
        count_cols = ['attempts_by_row', 'distinct_applicant_contact_count',
                      'email_count', 'phone_count', 'fax_count', 'total_contacts',
                      'email_interaction_count', 'email_method_count']
        for col in count_cols:
            if col in s.columns:
                # Ensure column is numeric before fillna/astype
                s[col] = pd.to_numeric(s[col], errors='coerce').fillna(0).astype('Int64')

        # Fill NA for agent/office/searchtype
        for col in ['agentname', 'office', 'searchtype']:
             if col in s.columns:
                 s[col] = s[col].fillna(f'Unknown {col.title()}').astype(str)

        # Ensure completion flag is Int64
        if 'is_completed' in s.columns:
             s['is_completed'] = pd.to_numeric(s['is_completed'], errors='coerce').fillna(0).astype('Int64')

        print(f"Augmentation complete. Summary DataFrame shape: {self.df_sum.shape}")
        # print(f"Summary columns: {list(self.df_sum.columns)}") # Optional: print columns for debugging
        # print(self.df_sum.head().to_markdown(index=False)) # Optional: print head for debugging

        return self

    # --------------- 3. Analysis Steps ------------------

    def _run_analysis(self, analysis_func: Callable, name: str, **kwargs) -> MBAnalysis:
        """Helper to run an analysis function and store results."""
        if self.df_sum is None:
            print(f"Skipping analysis '{name}': Summary data not available.")
            return self

        print(f"--- Running Analysis: {name} ---")
        try:
            # Pass df_sum and potentially df_hist if needed by the function
            func_args = {'df_summary': self.df_sum}
            # Check if the function expects df_history
            import inspect
            sig = inspect.signature(analysis_func)
            if 'df_history' in sig.parameters:
                 if self.df_hist is not None:
                     func_args['df_history'] = self.df_hist
                 else:
                     print(f"Warning: Analysis '{name}' requires history data, but it's not available.")
                     # Decide whether to skip or proceed without history data
                     # return self # Option: Skip if history is required but missing
                     # Option: Proceed, function must handle missing history
                     pass


            result = analysis_func(**func_args, **kwargs)
            self.results[name] = result
            print(f"Finished Analysis: {name}")
        except Exception as e:
            print(f"Error during analysis '{name}': {e}")
            import traceback
            traceback.print_exc()
            self.results[name] = {'error': str(e)} # Store error info

        return self

    def _merge_analysis_results(self, analysis_name: str, df_to_merge: pd.DataFrame | None, on_col: str = 'searchid'):
        """Merges results from an analysis step back into df_sum."""
        if df_to_merge is None or df_to_merge.empty:
            # print(f"Info: No data to merge from analysis '{analysis_name}'.")
            return

        if self.df_sum is None:
            print(f"Warning: Cannot merge results from '{analysis_name}', df_sum is None.")
            return

        if on_col not in df_to_merge.columns:
            print(f"Warning: Merge column '{on_col}' not found in results from '{analysis_name}'.")
            return

        # Select columns to merge (excluding the 'on' column)
        cols_to_merge = [col for col in df_to_merge.columns if col != on_col]
        if not cols_to_merge:
             print(f"Warning: No columns to merge (excluding '{on_col}') from analysis '{analysis_name}'.")
             return

        # Check for overlapping columns (excluding 'on_col') and drop them from df_sum before merge
        overlap_cols = [col for col in cols_to_merge if col in self.df_sum.columns]
        if overlap_cols:
            # print(f"Info: Dropping overlapping columns before merge: {overlap_cols}")
            self.df_sum = self.df_sum.drop(columns=overlap_cols)

        try:
            self.df_sum = pd.merge(
                self.df_sum,
                df_to_merge[[on_col] + cols_to_merge],
                on=on_col,
                how='left'
            )
            # print(f"Successfully merged results from '{analysis_name}'.")
        except Exception as e:
            print(f"Error merging results from '{analysis_name}': {e}")

    # --- Analysis Method Implementations (Adapting v8 logic) ---

    def calculate_basic_stats(self) -> MBAnalysis:
        """Calculates basic descriptive stats for key metrics, grouped by MB touch."""
        if self.df_sum is None: return self
        print("--- Running Analysis: Basic Stats (Grouped by MB Touch) ---")
        metrics_to_stat = [
            "tat_calendar_days", "tat_business_days",
            "attempts_by_row", "maxminusapplicant",
            "distinct_applicant_contact_count", "total_contacts",
            "email_interaction_count", "email_method_count"
        ]
        stats_results = {}
        df_sum_copy = self.df_sum.copy() # Work on a copy to avoid modifying self.df_sum here

        for metric in metrics_to_stat:
            if metric in df_sum_copy.columns:
                # Ensure metric is numeric before grouping
                df_sum_copy[metric] = pd.to_numeric(df_sum_copy[metric], errors='coerce')
                try:
                    # Filter out NaNs in the metric column before grouping
                    valid_data = df_sum_copy.dropna(subset=[metric, 'mb_touched'])
                    if not valid_data.empty:
                        g = (
                            valid_data.groupby('mb_touched')[metric]
                            .agg(["count", "mean", "median", "std"])
                            .reset_index()
                        )
                        stats_results[f"{metric}_by_mb_touch"] = g
                    else:
                         print(f"Info: No valid data for basic stats on '{metric}' after dropping NaNs.")
                         stats_results[f"{metric}_by_mb_touch"] = pd.DataFrame() # Empty DF if no data
                except Exception as e:
                    print(f"Error calculating basic stats for {metric}: {e}")
            # else: # No need to warn for every missing metric here
                # print(f"Warning: Metric '{metric}' not found for basic stats.")
        self.results['basic_stats'] = stats_results
        print("Finished Analysis: Basic Stats")
        return self

    def calculate_time_savings(self) -> MBAnalysis:
        """Estimates agent time savings based on touch reduction."""
        if self.df_sum is None: return self
        print("--- Running Analysis: Time Savings Estimation ---")
        # PATCH: Use human_touch_count if available, otherwise fallback to attempts_by_row
        if 'human_touch_count' in self.df_sum.columns:
             metric_to_use = 'human_touch_count'
             print("Using 'human_touch_count' for time savings calculation.")
        elif 'attempts_by_row' in self.df_sum.columns:
             metric_to_use = 'attempts_by_row'
             print("Warning: 'human_touch_count' not found. Using 'attempts_by_row' as fallback for time savings.")
        else:
             print(f"Warning: Neither 'human_touch_count' nor 'attempts_by_row' found for time savings calculation.")
             self.results['time_savings'] = {'error': "Missing required columns"}
             return self

        # Ensure metric is numeric
        # Work on a copy to avoid modifying self.df_sum directly in this step
        df_sum_copy = self.df_sum.copy()
        df_sum_copy[metric_to_use] = pd.to_numeric(df_sum_copy[metric_to_use], errors='coerce')

        df = df_sum_copy.dropna(subset=[metric_to_use, 'mb_touched']) # Drop NA before splitting
        mb_df = df[df.mb_touched == 1]
        ctl_df = df[df.mb_touched == 0]

        if mb_df.empty or ctl_df.empty:
            print("Warning: Insufficient data (MB or Control group empty) for time savings calculation.")
            self.results['time_savings'] = {'error': "MB or Control group empty"}
            return self

        avg_mb = mb_df[metric_to_use].mean()
        avg_ctl = ctl_df[metric_to_use].mean()

        if pd.isna(avg_mb) or pd.isna(avg_ctl):
             print("Warning: Could not calculate average touches for MB or non-MB groups.")
             self.results['time_savings'] = {'error': "Could not calculate average touches"}
             return self

        touch_diff = avg_ctl - avg_mb
        hours_per_verification = touch_diff * cfg.time_per_touch_min / 60
        total_hours_saved = hours_per_verification * len(mb_df)

        # FTE calculation (requires history data for date range)
        fte = pd.NA
        weeks = pd.NA
        weekly_saved = pd.NA
        if self.df_hist is not None and 'historydatetime' in self.df_hist.columns:
            dates = pd.to_datetime(self.df_hist['historydatetime'], errors='coerce').dropna()
            if not dates.empty:
                min_date, max_date = dates.min(), dates.max()
                if pd.notna(min_date) and pd.notna(max_date) and max_date >= min_date:
                    days = (max_date - min_date).days
                    # Use max(1, days + 1) to handle single-day data and ensure at least 1 day duration
                    weeks = max(1, days + 1) / 7
                    if weeks > 0:
                        weekly_saved = total_hours_saved / weeks
                        fte = weekly_saved / 40 # Assuming 40-hour work week
                else:
                     print("Warning: Invalid date range found in historydatetime for FTE calculation.")
            else:
                 print("Warning: No valid dates found in historydatetime after dropping NaNs for FTE calculation.")
        else:
             print("Warning: History data or 'historydatetime' needed for accurate FTE calculation.")

        savings_dict = {
            'metric_used': metric_to_use,
            'avg_metric_mb': avg_mb,
            'avg_metric_ctl': avg_ctl,
            'metric_reduction': touch_diff,
            'hours_saved_per_verification': hours_per_verification,
            'mb_verification_count': len(mb_df),
            'total_hours_saved': total_hours_saved,
            'analysis_period_weeks': weeks,
            'weekly_hours_saved': weekly_saved,
            'fte_equivalent': fte,
        }
        self.results['time_savings'] = savings_dict
        print("Finished Analysis: Time Savings Estimation")
        return self

    def run_statistical_tests(self) -> MBAnalysis:
        """Performs key statistical tests (e.g., Welch t-test) on metrics."""
        if self.df_sum is None: return self
        print("--- Running Analysis: Statistical Tests ---")
        tests = {}
        metrics_to_test = {
            "tat_calendar_days": "TAT (Calendar Days)",
            "tat_business_days": "TAT (Business Days)",
            "attempts_by_row": "Attempts per Verification",
            "maxminusapplicant": "Non-Applicant Attempts",
            "email_yield": "Email Yield" # Requires calculation first
        }
        
        # Work on a copy to avoid modifying self.df_sum
        df_sum_copy = self.df_sum.copy()

        # Calculate email yield if possible
        if 'email_method_count' in df_sum_copy.columns and 'email_interaction_count' in df_sum_copy.columns:
            # Ensure source columns are numeric first
            df_sum_copy['email_method_count_num'] = pd.to_numeric(df_sum_copy['email_method_count'], errors='coerce').fillna(0)
            df_sum_copy['email_interaction_count_num'] = pd.to_numeric(df_sum_copy['email_interaction_count'], errors='coerce').fillna(0)
            
            # Calculate yield safely
            df_sum_copy['email_yield'] = np.where(
                df_sum_copy['email_method_count_num'] > 0,
                df_sum_copy['email_interaction_count_num'] / df_sum_copy['email_method_count_num'],
                np.nan # Use NaN if denominator is 0, t-test will handle it
            )
        else:
            print("Warning: Cannot calculate email yield for t-test - missing required columns.")
            if 'email_yield' in metrics_to_test: del metrics_to_test['email_yield']

        # Perform tests
        for metric, desc in metrics_to_test.items():
            if metric in df_sum_copy.columns:
                # Ensure metric is numeric
                df_sum_copy[metric] = pd.to_numeric(df_sum_copy[metric], errors='coerce')
                
                # Split data after ensuring numeric type
                mb_data = df_sum_copy[df_sum_copy.mb_touched == 1][metric]
                ctl_data = df_sum_copy[df_sum_copy.mb_touched == 0][metric]

                t_stat, p_value = welch_ttest(mb_data, ctl_data) # Uses cleaned data internally

                if t_stat is not None and p_value is not None:
                    tests[metric] = {
                        'description': desc,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < cfg.significance_level,
                        'mb_mean': mb_data.mean(), # Calculate mean on original split series
                        'ctl_mean': ctl_data.mean(),
                        'mb_n': len(mb_data.dropna()), # Count non-NA values
                        'ctl_n': len(ctl_data.dropna())
                    }
                else:
                     tests[metric] = {
                         'description': desc,
                         'error': 'Test could not be performed (insufficient data or zero variance)',
                         'mb_mean': mb_data.mean(), # Still report means if possible
                         'ctl_mean': ctl_data.mean(),
                         'mb_n': len(mb_data.dropna()),
                         'ctl_n': len(ctl_data.dropna())
                     }
            # else: # No need to warn here, handled above
                # print(f"Warning: Metric '{metric}' not found for statistical test.")

        self.results['statistical_tests'] = tests
        print("Finished Analysis: Statistical Tests")
        return self

    def calculate_autonomy(self) -> MBAnalysis:
        """Calculates autonomy metrics (human touches, rework, fallback)."""
        if self.df_hist is None or self.df_sum is None:
            print("Skipping autonomy analysis: History or Summary data missing.")
            return self
        # This adapts the v8 logic directly, could be further optimized
        print("--- Running Analysis: Autonomy Metrics ---")
        try:
            hist = self.df_hist.copy() # Work on a copy
            # Ensure required columns exist and have flags
            required_hist = ['searchid', 'historydatetime', 'is_mb_row']
            if not all(col in hist.columns for col in required_hist):
                # Attempt to recalculate is_mb_row if missing
                if 'is_mb_row' not in hist.columns and 'agentname' in hist.columns and 'username' in hist.columns:
                     print("Warning: 'is_mb_row' missing, recalculating for autonomy analysis.")
                     mb_pattern = cfg.mb_agent_identifier.lower()
                     hist['agentname'] = hist['agentname'].fillna('').astype(str)
                     hist['username'] = hist['username'].fillna('').astype(str)
                     hist['is_mb_row'] = (
                         hist['agentname'].str.lower().str.contains(mb_pattern, na=False) |
                         hist['username'].str.lower().str.contains(mb_pattern, na=False)
                     )
                else:
                    raise ValueError(f"Missing required columns in history for autonomy: {required_hist}")

            if not pd.api.types.is_datetime64_any_dtype(hist['historydatetime']):
                hist['historydatetime'] = pd.to_datetime(hist['historydatetime'], errors='coerce')

            hist['is_human_touch'] = ~hist['is_mb_row']
            # Drop rows with NaT historydatetime before sorting/grouping
            hist_sorted = hist.sort_values(['searchid', 'historydatetime']).dropna(subset=['historydatetime'])
            
            if hist_sorted.empty:
                 print("Warning: No valid history data after dropping NaT dates for autonomy analysis.")
                 self.results['autonomy_summary'] = {'error': 'No valid history data'}
                 self.results['autonomy_details'] = pd.DataFrame()
                 return self

            # Human touch count
            human_touches = hist_sorted[hist_sorted['is_human_touch']].groupby('searchid').size().rename('human_touch_count')

            # Rework (Human after MB)
            # Ensure shift happens correctly even with missing groups or single-row groups
            hist_sorted['prev_is_human'] = hist_sorted.groupby('searchid')['is_human_touch'].shift(1)
            rework_mask = hist_sorted['is_human_touch'] & (hist_sorted['prev_is_human'] == False)
            rework_search_ids = hist_sorted.loc[rework_mask, 'searchid'].unique()

            # Fallback (Last touch is human)
            last_touches = hist_sorted.groupby('searchid').tail(1)
            fallback_search_ids = last_touches.loc[last_touches['is_human_touch'], 'searchid'].unique()

            # Aggregate
            autonomy_df = pd.DataFrame({'searchid': self.df_sum['searchid'].unique()})
            autonomy_df = pd.merge(autonomy_df, human_touches, on='searchid', how='left')
            autonomy_df['human_touch_count'] = autonomy_df['human_touch_count'].fillna(0).astype('Int64')
            autonomy_df['fully_autonomous'] = (autonomy_df['human_touch_count'] == 0).astype(int)
            autonomy_df['has_rework'] = autonomy_df['searchid'].isin(rework_search_ids).astype(int)
            autonomy_df['had_fallback'] = autonomy_df['searchid'].isin(fallback_search_ids).astype(int)

            # Merge results back to self.df_sum
            self._merge_analysis_results('autonomy', autonomy_df)

            # Calculate summary metrics
            summary_metrics = {}
            mb_ids = self.df_sum.loc[self.df_sum['mb_touched'] == 1, 'searchid'].unique()
            # Ensure autonomy_df is used for calculations, not the potentially modified self.df_sum
            autonomy_mb_subset = autonomy_df[autonomy_df['searchid'].isin(mb_ids)]

            summary_metrics['avg_human_touches_overall'] = autonomy_df['human_touch_count'].mean()
            summary_metrics['autonomy_rate_overall'] = autonomy_df['fully_autonomous'].mean() * 100
            summary_metrics['rework_rate_mb_only'] = autonomy_mb_subset['has_rework'].mean() * 100 if not autonomy_mb_subset.empty else 0
            summary_metrics['fallback_rate_mb_only'] = autonomy_mb_subset['had_fallback'].mean() * 100 if not autonomy_mb_subset.empty else 0

            self.results['autonomy_summary'] = summary_metrics
            self.results['autonomy_details'] = autonomy_df # Store the detailed df as well

            print("Finished Analysis: Autonomy Metrics")
        except Exception as e:
            print(f"Error during autonomy calculation: {e}")
            self.results['autonomy_summary'] = {'error': str(e)}
            self.results['autonomy_details'] = pd.DataFrame() # Ensure key exists
        return self

    def analyze_contact_plan_impact(self) -> MBAnalysis:
        """Analyzes the impact of MB contact plans (coverage, delta)."""
        if self.df_sum is None:
            print("Skipping contact plan analysis: Summary data missing.")
            return self
        if 'contact_plan_provided' not in self.df_sum.columns:
            print("Skipping contact plan analysis: 'contact_plan_provided' flag missing.")
            # Attempt to recalculate flag if history is available
            if self.df_hist is not None and 'is_mb_contact_plan_note' in self.df_hist.columns:
                 print("Warning: 'contact_plan_provided' missing in summary, recalculating from history.")
                 plan_flag_agg = self.df_hist.groupby('searchid')['is_mb_contact_plan_note'].max().reset_index()
                 plan_flag_agg = plan_flag_agg.rename(columns={'is_mb_contact_plan_note': 'contact_plan_provided'})
                 plan_flag_agg['contact_plan_provided'] = plan_flag_agg['contact_plan_provided'].astype(int)
                 # Merge this recalculated flag
                 self.df_sum = self.df_sum.drop(columns=['contact_plan_provided'], errors='ignore')
                 self.df_sum = pd.merge(self.df_sum, plan_flag_agg, on='searchid', how='left')
                 self.df_sum['contact_plan_provided'] = self.df_sum['contact_plan_provided'].fillna(0).astype(int)
            else:
                 print("Error: Cannot proceed with contact plan analysis - flag missing and cannot recalculate.")
                 return self


        print("--- Running Analysis: Contact Plan Impact ---")
        try:
            s = self.df_sum.copy() # Work on a copy
            # Ensure flag is boolean for grouping logic clarity
            s['contact_plan_provided_bool'] = s['contact_plan_provided'].astype(bool)

            # Ensure metrics are numeric
            metrics_needed = ['is_completed', 'tat_calendar_days', 'attempts_by_row']
            for m in metrics_needed:
                if m in s.columns:
                     s[m] = pd.to_numeric(s[m], errors='coerce')
                else:
                     # If a metric is missing, we cannot calculate delta reliably
                     print(f"Warning: Missing required column '{m}' for contact plan delta analysis. Skipping delta calculation.")
                     self.results['contact_plan_delta'] = pd.DataFrame() # Store empty DF
                     self.results['contact_plan_verdict'] = {'error': f"Missing column: {m}"}
                     return self

            # Perform aggregation
            grp = s.groupby('contact_plan_provided_bool')
            delta_agg = grp.agg(
                search_count=('searchid', 'size'),
                completion_rate=('is_completed', 'mean'),
                avg_tat_days=('tat_calendar_days', 'mean'),
                avg_attempts_by_row=('attempts_by_row', 'mean')
            ).reset_index()
            # Rename boolean column back for consistency if needed, or keep as bool
            delta_agg = delta_agg.rename(columns={'contact_plan_provided_bool': 'contact_plan_provided'})

            # Calculate delta (plan - no_plan)
            row_with = delta_agg[delta_agg['contact_plan_provided'] == True]
            row_without = delta_agg[delta_agg['contact_plan_provided'] == False]

            delta_completion_rate = np.nan
            delta_tat_days = np.nan
            delta_attempts = np.nan

            if not row_with.empty and not row_without.empty:
                # Calculate only if means are not NaN
                if pd.notna(row_with['completion_rate'].iloc[0]) and pd.notna(row_without['completion_rate'].iloc[0]):
                    delta_completion_rate = row_with['completion_rate'].iloc[0] - row_without['completion_rate'].iloc[0]
                if pd.notna(row_with['avg_tat_days'].iloc[0]) and pd.notna(row_without['avg_tat_days'].iloc[0]):
                    delta_tat_days = row_with['avg_tat_days'].iloc[0] - row_without['avg_tat_days'].iloc[0]
                if pd.notna(row_with['avg_attempts_by_row'].iloc[0]) and pd.notna(row_without['avg_attempts_by_row'].iloc[0]):
                    delta_attempts = row_with['avg_attempts_by_row'].iloc[0] - row_without['avg_attempts_by_row'].iloc[0]

            # PATCH: Add delta columns to both rows, filling False row with NaN
            delta_agg['delta_completion_rate'] = np.nan
            delta_agg['delta_tat_days'] = np.nan
            delta_agg['delta_attempts'] = np.nan

            delta_agg.loc[delta_agg.contact_plan_provided == True, 'delta_completion_rate'] = delta_completion_rate
            delta_agg.loc[delta_agg.contact_plan_provided == True, 'delta_tat_days'] = delta_tat_days
            delta_agg.loc[delta_agg.contact_plan_provided == True, 'delta_attempts'] = delta_attempts

            self.results['contact_plan_delta'] = delta_agg

            # Calculate Contact Plan Verdict
            verdict = {'overall_good': False, 'reason': 'Prerequisites not met'}
            if not row_with.empty and not row_without.empty and pd.notna(delta_tat_days) and pd.notna(delta_completion_rate) and pd.notna(delta_attempts):
                 # Test TAT significance using the main summary df 's'
                 group_with_plan = s[s['contact_plan_provided_bool']]['tat_calendar_days']
                 group_without_plan = s[~s['contact_plan_provided_bool']]['tat_calendar_days']
                 _, p_val_tat = welch_ttest(group_with_plan, group_without_plan)

                 # Check conditions using config thresholds
                 # Note: tat reduction check uses negative delta
                 tat_ok = (delta_tat_days < -cfg.contact_plan_tat_reduction_days) and (p_val_tat is not None and p_val_tat < cfg.significance_level)
                 completion_ok = (delta_completion_rate * 100) > cfg.contact_plan_completion_increase_pp
                 attempts_ok = delta_attempts < 0 # Attempts should be lower

                 verdict = {
                     'tat_ok': tat_ok,
                     'completion_ok': completion_ok,
                     'attempts_ok': attempts_ok,
                     'overall_good': tat_ok and completion_ok and attempts_ok,
                     'delta_tat_days': delta_tat_days,
                     'tat_p_value': p_val_tat,
                     'delta_completion_pp': delta_completion_rate * 100,
                     'delta_attempts': delta_attempts,
                     'reason': 'OK'
                 }
            elif row_with.empty or row_without.empty:
                 verdict['reason'] = "Missing 'with plan' or 'without plan' group"
            else:
                 verdict['reason'] = "Delta calculation failed (NaN results)"

            self.results['contact_plan_verdict'] = verdict
            print("Finished Analysis: Contact Plan Impact")

        except Exception as e:
            print(f"Error during contact plan impact analysis: {e}")
            self.results['contact_plan_delta'] = {'error': str(e)}
            self.results['contact_plan_verdict'] = {'error': str(e)}
        return self

    def calculate_verdicts(self) -> MBAnalysis:
        """Calculates overall verdicts based on analysis results."""
        if self.df_sum is None: return self
        print("--- Running Analysis: Verdict Calculation ---")
        verdicts = {}

        # Email Performance Verdict (adapting v8 logic)
        email_verdict = {'verdict': False, 'reason': 'Prerequisites not met'}
        required_email = ['mb_touched', 'email_method_count', 'email_interaction_count']
        if all(col in self.df_sum.columns for col in required_email):
            try:
                # Ensure counts are numeric
                s = self.df_sum.copy()
                s['email_method_count'] = pd.to_numeric(s['email_method_count'], errors='coerce').fillna(0)
                s['email_interaction_count'] = pd.to_numeric(s['email_interaction_count'], errors='coerce').fillna(0)

                subset = s[s['email_method_count'] > 0].copy() # Only searches with outbound emails
                if not subset.empty:
                    # Calculate yield, handle division by zero by resulting in NaN
                    subset['email_yield'] = subset['email_interaction_count'].astype(float) / subset['email_method_count'].astype(float)
                    subset['email_yield'] = subset['email_yield'].replace([np.inf, -np.inf], np.nan) # Replace inf with NaN

                    mb_yield_data = subset[subset['mb_touched'] == 1]['email_yield']
                    ctl_yield_data = subset[subset['mb_touched'] == 0]['email_yield']

                    mb_mean = mb_yield_data.mean() # mean ignores NaN
                    ctl_mean = ctl_yield_data.mean()
                    diff_pp = (mb_mean - ctl_mean) * 100 if pd.notna(mb_mean) and pd.notna(ctl_mean) else np.nan

                    t_stat, p_val = welch_ttest(mb_yield_data, ctl_yield_data) # Handles NaN internally

                    verdict_bool = False
                    reason = 'OK'
                    if t_stat is None: # Test couldn't run
                        reason = f'Insufficient data (MB: {len(mb_yield_data.dropna())}, CTL: {len(ctl_yield_data.dropna())}) or zero variance'
                    elif pd.isna(diff_pp):
                         reason = 'Could not calculate yield difference (means were NaN)'
                    else:
                        # Check if p_val is not None before comparing
                        if p_val is not None:
                             verdict_bool = (diff_pp >= cfg.email_verdict_min_yield_increase_pp) and (p_val < cfg.significance_level)
                        else:
                             reason = 'T-test failed, cannot determine significance'


                    email_verdict = {
                        'verdict': verdict_bool, 'mb_yield': mb_mean, 'ctl_yield': ctl_mean,
                        'diff_pp': diff_pp, 'p_value': p_val, 'reason': reason
                    }
                else:
                     email_verdict['reason'] = 'No searches with email_method_count > 0'
            except Exception as e:
                 print(f"Error calculating email verdict: {e}")
                 email_verdict['reason'] = f'Error: {e}'
        else:
            missing_cols = [col for col in required_email if col not in self.df_sum.columns]
            email_verdict['reason'] = f'Missing columns: {missing_cols}'

        verdicts['email_performance'] = email_verdict

        # Add Contact Plan Verdict (already calculated)
        verdicts['contact_plan'] = self.results.get('contact_plan_verdict', {'error': 'Not calculated yet'})

        self.results['verdicts'] = verdicts
        print("Finished Analysis: Verdict Calculation")
        return self

    def calculate_agent_office_metrics(self) -> MBAnalysis:
         """Calculates metrics grouped by agent and office."""
         if self.df_sum is None: return self
         print("--- Running Analysis: Agent/Office Metrics ---")
         s = self.df_sum.copy() # Work on a copy
         results = {}

         # --- Agent Throughput (Weekly/Monthly/Daily) ---
         agent_col = 'agentname'
         if agent_col in s.columns and 'completiondate' in s.columns and 'is_completed' in s.columns:
             # Ensure types before filtering
             s[agent_col] = s[agent_col].fillna('Unknown Agent').astype(str)
             s['is_completed'] = pd.to_numeric(s['is_completed'], errors='coerce').fillna(0).astype(int)
             s['completiondate'] = pd.to_datetime(s['completiondate'], errors='coerce')

             completed = s[(s['is_completed'] == 1) & s['completiondate'].notna()].copy()
             if not completed.empty:
                 try:
                     # Weekly
                     completed['week'] = completed['completiondate'].dt.to_period('W-MON').astype(str)
                     weekly = completed.groupby([agent_col, 'week'], as_index=False).size().rename(columns={'size': 'verified_count'})
                     results['verified_by_agent_week'] = weekly
                 except Exception as e: print(f"Error calculating weekly agent throughput: {e}")
                 try:
                     # Monthly
                     completed['month'] = completed['completiondate'].dt.to_period('M').astype(str)
                     monthly = completed.groupby([agent_col, 'month'], as_index=False).size().rename(columns={'size': 'verified_count'})
                     results['verified_by_agent_month'] = monthly
                 except Exception as e: print(f"Error calculating monthly agent throughput: {e}")
                 try:
                     # Daily
                     completed['day'] = completed['completiondate'].dt.date
                     daily = completed.groupby([agent_col, 'day'], as_index=False).size().rename(columns={'size': 'verified_count'})
                     results['verified_by_agent_day'] = daily
                 except Exception as e: print(f"Error calculating daily agent throughput: {e}")
             else:
                 print("Info: No completed searches with dates for agent throughput.")
         else:
             print("Warning: Missing columns for agent throughput (agentname, completiondate, is_completed).")

         # --- Monthly Completion Breakdowns (for Excel) ---
         monthly_breakdowns = {}
         if 'is_completed' in s.columns and 'completiondate' in s.columns and pd.api.types.is_datetime64_any_dtype(s['completiondate']) and 'searchtype' in s.columns:
             # Re-filter completed on the potentially modified 's' DataFrame
             s['completiondate'] = pd.to_datetime(s['completiondate'], errors='coerce') # Ensure datetime again
             completed = s[(s['is_completed'] == 1) & s['completiondate'].notna()].copy()
             if not completed.empty:
                 try:
                     completed['month_period'] = completed['completiondate'].dt.to_period('M').astype(str)
                     completed['searchtype'] = completed['searchtype'].fillna('Unknown Type').astype(str)
                     # Overall
                     monthly_breakdowns['Overall'] = completed.groupby(['month_period', 'searchtype'])['searchid'].count().reset_index(name='completions')
                     # By Office
                     if 'office' in completed.columns:
                         completed['office'] = completed['office'].fillna('Unknown Office').astype(str)
                         monthly_breakdowns['By Office'] = completed.groupby(['month_period', 'office', 'searchtype'])['searchid'].count().reset_index(name='completions')
                     # By Agent
                     if 'agentname' in completed.columns:
                          completed['agentname'] = completed['agentname'].fillna('Unknown Agent').astype(str)
                          monthly_breakdowns['By Agent'] = completed.groupby(['month_period', 'agentname', 'searchtype'])['searchid'].count().reset_index(name='completions')
                 except Exception as e:
                      print(f"Error calculating monthly breakdowns: {e}")
             else:
                 print("Info: No completed searches for monthly breakdowns.")
         else:
             print("Warning: Missing columns for monthly breakdowns (is_completed, completiondate, searchtype).")
         results['monthly_completion_breakdowns'] = monthly_breakdowns

         # --- Agent Completion Uplift ---
         agent_uplift_df = pd.DataFrame() # Default empty
         if agent_col in s.columns and 'contact_plan_provided' in s.columns and 'is_completed' in s.columns:
            try:
                # Ensure types on 's' before grouping
                s[agent_col] = s[agent_col].fillna('Unknown Agent').astype(str)
                s['contact_plan_provided'] = s['contact_plan_provided'].astype(bool)
                s['is_completed'] = pd.to_numeric(s['is_completed'], errors='coerce')

                # Group and aggregate mean/count, handle potential missing groups with fill_value=0 for count
                g = s.groupby([agent_col, 'contact_plan_provided'])['is_completed'].agg(['mean', 'count'])
                g = g.unstack(level='contact_plan_provided', fill_value=0) # Fill missing groups

                # Flatten columns carefully, checking if levels exist
                flat_cols = {}
                for stat in ['mean', 'count']:
                     for status in [False, True]:
                          col_tuple = (stat, status)
                          new_col_name = f"{stat}_{status}"
                          if col_tuple in g.columns:
                               flat_cols[col_tuple] = new_col_name
                          else:
                               # If a column is missing (e.g., no agent ever had contact_plan=True), add it with NaN/0
                               print(f"Info: Missing column {col_tuple} in agent uplift aggregation. Adding with default values.")
                               g[col_tuple] = 0 if stat == 'count' else np.nan
                               flat_cols[col_tuple] = new_col_name

                g.columns = g.columns.map(lambda x: flat_cols.get(x, '_'.join(map(str, x)))) # Flatten multiindex


                # Filter agents: must have minimum number of searches in both groups
                g_filtered = g[(g['count_True'] >= cfg.agent_uplift_min_searches) & (g['count_False'] >= cfg.agent_uplift_min_searches)].copy()


                if not g_filtered.empty:
                    # Calculate delta only for filtered agents
                    g_filtered['delta_completion_pp'] = (g_filtered['mean_True'] - g_filtered['mean_False']) * 100

                    # Add significance test (simplified loop)
                    p_values = []
                    for agent in g_filtered.index:
                        agent_data = s[s[agent_col] == agent] # Filter original data for this agent
                        plan = agent_data[agent_data.contact_plan_provided]['is_completed'] # Use boolean index
                        nolp = agent_data[~agent_data.contact_plan_provided]['is_completed']
                        _, p = welch_ttest(plan, nolp) # welch_ttest handles NaNs and size checks
                        p_values.append(p if p is not None else np.nan)

                    g_filtered['p_value'] = p_values
                    g_filtered['significant'] = (g_filtered['p_value'] < cfg.significance_level) & g_filtered['p_value'].notna()
                    agent_uplift_df = g_filtered.reset_index()
                else:
                     print("Info: No agents met criteria for completion uplift analysis.")

            except Exception as e:
                print(f"Error calculating agent completion uplift: {e}")
                traceback.print_exc()
         else:
             print("Warning: Missing columns for agent completion uplift.")
         results['agent_completion_uplift'] = agent_uplift_df

         self.results['agent_office_metrics'] = results # Store all under one key
         print("Finished Analysis: Agent/Office Metrics")
         return self

    def calculate_queue_metrics(self) -> MBAnalysis:
        """Calculates queue-related metrics like age and depth."""
        if self.df_hist is None or self.df_sum is None:
            print("Skipping queue metrics: History or Summary data missing.")
            return self
        print("--- Running Analysis: Queue Metrics ---")
        results = {}
        h = self.df_hist.copy() # Work on copy
        s = self.df_sum.copy()

        # --- Open Queue Age ---
        open_queue_df = pd.DataFrame()
        required_hist_q = ['searchid', 'historydatetime', 'searchstatus', 'is_mb_row']
        if all(col in h.columns for col in required_hist_q):
            try:
                h['searchstatus'] = h['searchstatus'].fillna('').astype(str).str.upper()
                if not pd.api.types.is_datetime64_any_dtype(h['historydatetime']):
                    h['historydatetime'] = pd.to_datetime(h['historydatetime'], errors='coerce')

                # Get last record per search, dropping those with NaT dates first
                h_valid_dates = h.dropna(subset=['historydatetime'])
                last_records = h_valid_dates.loc[h_valid_dates.groupby('searchid')['historydatetime'].idxmax()]
                open_searches = last_records[last_records['searchstatus'] != cfg.completion_status].copy()

                if not open_searches.empty:
                    # Determine timezone from data if available, otherwise assume local/naive
                    source_tz = open_searches['historydatetime'].dt.tz
                    try:
                        # Attempt to get timezone-aware current time
                        today = pd.Timestamp.now(tz=source_tz).normalize()
                    except TypeError: # Handle case where source_tz might be None or invalid
                        today = pd.Timestamp.now().normalize() # Use naive local time
                        if source_tz is not None: # If data had tz, make today naive too
                             open_searches['historydatetime'] = open_searches['historydatetime'].dt.tz_localize(None)

                    # Ensure 'today' is compatible with historydatetime timezone for subtraction
                    if today.tz is not None and open_searches['historydatetime'].dt.tz is None:
                         today = today.tz_localize(None)
                    elif today.tz is None and open_searches['historydatetime'].dt.tz is not None:
                         # This case is tricky - convert data to naive or make today aware?
                         # Safest is often to work with naive dates for calculations like this
                         today = pd.Timestamp.now().normalize()
                         open_searches['historydatetime'] = open_searches['historydatetime'].dt.tz_localize(None)


                    open_searches['age_days'] = (today - open_searches['historydatetime'].dt.normalize()).dt.days.clip(lower=0)
                    bins = [-1, 2, 5, 10, 20, float('inf')]
                    labels = ['0-2 days', '3-5 days', '6-10 days', '11-20 days', '>20 days']
                    open_searches['age_bucket'] = pd.cut(open_searches['age_days'], bins=bins, labels=labels, right=True)
                    # Ensure is_mb_row is boolean/int for grouping
                    open_searches['is_mb_row'] = open_searches['is_mb_row'].astype(int)
                    open_queue_df = (
                        open_searches.groupby(['is_mb_row', 'age_bucket'], observed=False)
                                     .size().rename('count')
                                     .unstack(fill_value=0) # Pivot for easier plotting/viewing
                    )
                    open_queue_df.columns = ['Non-MB Count' if col == 0 else 'MB Count' for col in open_queue_df.columns]
                    open_queue_df = open_queue_df.reset_index()

                else:
                     print("Info: No open searches found for queue age analysis.")
            except Exception as e:
                 print(f"Error calculating open queue age: {e}")
                 traceback.print_exc()
        else:
             missing_cols_q = [col for col in required_hist_q if col not in h.columns]
             print(f"Warning: Missing columns for open queue age: {missing_cols_q}")
        results['open_queue_age_by_mb'] = open_queue_df

        # --- First Day SLA Rate ---
        sla_df = pd.DataFrame()
        if 'tat_calendar_days' in s.columns and 'mb_touched' in s.columns:
            try:
                s_copy = s[['mb_touched', 'tat_calendar_days']].copy()
                s_copy['tat_calendar_days'] = pd.to_numeric(s_copy['tat_calendar_days'], errors='coerce')
                # Handle NA before comparison
                s_copy['first_day_close'] = np.where(
                    s_copy['tat_calendar_days'].isna(),
                    0, # Treat NA TAT as not meeting SLA
                    (s_copy['tat_calendar_days'] <= 1).astype(int)
                )
                sla_df = (
                    s_copy.groupby('mb_touched')['first_day_close']
                          .agg(rate='mean', count='size') # Get rate and count
                          .reset_index()
                )
                # Add the flag back to the main summary df if needed downstream
                # self.df_sum['first_day_close'] = s_copy['first_day_close'] # Add this line if needed
            except Exception as e:
                 print(f"Error calculating first day SLA rate: {e}")
        else:
             print("Warning: Missing columns for first day SLA rate (tat_calendar_days, mb_touched).")
        results['first_day_sla_rate'] = sla_df

        # --- Weekly Queue Depth (> SLA Days) ---
        q_depth_df = pd.DataFrame()
        required_hist_d = ['searchid', 'historydatetime', 'searchstatus']
        if all(col in h.columns for col in required_hist_d):
            try:
                h_copy = h[required_hist_d].dropna(subset=['historydatetime']).copy()
                if not pd.api.types.is_datetime64_any_dtype(h_copy['historydatetime']):
                     h_copy['historydatetime'] = pd.to_datetime(h_copy['historydatetime'], errors='coerce')
                h_copy['searchstatus'] = h_copy['searchstatus'].fillna('').astype(str).str.upper()

                # Re-check for NaT dates after conversion
                h_copy = h_copy.dropna(subset=['historydatetime'])

                if not h_copy.empty:
                    last_records = h_copy.loc[h_copy.groupby('searchid')['historydatetime'].idxmax()]
                    open_searches = last_records[last_records['searchstatus'] != cfg.completion_status].copy()

                    if not open_searches.empty:
                        # Use business_days_inclusive helper
                        today = pd.Timestamp.now(tz=open_searches['historydatetime'].dt.tz) # Match source timezone
                        start_dates = open_searches['historydatetime']
                        # Create a Series of 'today' aligned with start_dates index
                        end_dates = pd.Series(today, index=start_dates.index)

                        open_searches['age_bus'] = business_days_inclusive(start_dates, end_dates).clip(lower=0)
                        # Assign week based on the last history date (start_dates here)
                        open_searches['week'] = start_dates.dt.normalize().dt.to_period('W-MON').astype(str)

                        # Aggregate weekly stats
                        weekly = (
                            open_searches.groupby('week')
                                         .agg(total_open=('searchid','nunique'),
                                              # Ensure age_bus is numeric before comparison
                                              open_gt_sla=('age_bus', lambda x: (pd.to_numeric(x, errors='coerce') > cfg.queue_depth_sla_days).sum()))
                                         .reset_index()
                        )
                        # Calculate percentage, handle division by zero
                        weekly['pct_open_gt_sla'] = np.where(
                            weekly['total_open'] > 0,
                            (weekly['open_gt_sla'] / weekly['total_open'] * 100).round(2),
                            0
                        )
                        q_depth_df = weekly
                    else:
                         print("Info: No open searches found for queue depth analysis.")
                else:
                     print("Warning: No history records remaining after dropping NaT dates for queue depth.")
            except Exception as e:
                 print(f"Error calculating weekly queue depth: {e}")
                 traceback.print_exc()
        else:
             missing_cols_d = [col for col in required_hist_d if col not in h.columns]
             print(f"Warning: Missing columns for weekly queue depth: {missing_cols_d}")
        results['weekly_queue_depth'] = q_depth_df


        self.results['queue_metrics'] = results # Store all under one key
        print("Finished Analysis: Queue Metrics")
        return self

    # --- NEW ANALYSIS METHODS START ---

    def calculate_cohens_d(self) -> MBAnalysis:
        """
        Computes Cohen's d for key metrics between MB-touched vs control.
        Stores results in self.results['effect_sizes'].
        """
        if self.df_sum is None:
            print("Skipping Cohen's d: Summary data missing.")
            return self
        print("--- Running Analysis: Cohen's d Effect Sizes ---")
        df = self.df_sum.copy()
        df['mb'] = df.mb_touched == 1 # Ensure mb_touched is boolean/int for comparison

        metrics = ['tat_calendar_days', 'tat_business_days', 'attempts_by_row']
        effects = {}
        for m in metrics:
            if m not in df.columns:
                print(f"Warning: Metric '{m}' not found for Cohen's d calculation.")
                effects[m] = None
                continue
            # Ensure metric is numeric and drop NAs
            a = df.loc[df.mb, m].dropna().astype(float)
            b = df.loc[~df.mb, m].dropna().astype(float)

            # pooled SD calculation
            n1, n2 = len(a), len(b)
            if n1 < 2 or n2 < 2:
                print(f"Info: Insufficient data for Cohen's d on '{m}' (MB: {n1}, CTL: {n2}). Need >= 2.")
                effects[m] = None
                continue

            s1, s2 = a.std(ddof=1), b.std(ddof=1) # Use sample standard deviation

            # Check for NaN or zero variances before calculating pooled SD
            if pd.isna(s1) or pd.isna(s2) or (np.isclose(s1, 0) and np.isclose(s2, 0)):
                 print(f"Info: Cannot calculate Cohen's d for '{m}' due to NaN or zero variance in one or both groups.")
                 effects[m] = None
                 continue

            # Handle case where one variance is zero but means are different
            if (np.isclose(s1, 0) or np.isclose(s2, 0)) and not np.isclose(a.mean(), b.mean()):
                 print(f"Info: Skipping Cohen's d for '{m}' due to zero variance in one group with different means.")
                 effects[m] = None
                 continue
            elif np.isclose(s1, 0) and np.isclose(s2, 0) and np.isclose(a.mean(), b.mean()):
                 # If both variances are zero and means are same, effect is 0
                 pooled_sd = 0
            else:
                 # Calculate pooled standard deviation
                 pooled_sd = (( (n1-1)*s1**2 + (n2-1)*s2**2 ) / (n1+n2-2))**0.5

            # Calculate Cohen's d
            d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0
            effects[m] = {'cohens_d': d, 'n_mb': n1, 'n_ctl': n2}

        self.results['effect_sizes'] = effects
        print("Finished Analysis: Cohen's d Effect Sizes")
        return self


    def run_anova_by_capability(self) -> MBAnalysis:
        """
        Performs one-way ANOVA on tat_calendar_days grouped by capability category:
           0 = No MB, 1 = Contact Only, 2 = Email Only, 3 = Both
        Stores results in self.results['anova'].
        """
        if self.df_sum is None:
            print("Skipping ANOVA: Summary data missing.")
            return self
        if not all(c in self.df_sum.columns for c in ['mb_contact_research', 'mb_email_handling', 'tat_calendar_days']):
             missing_cols = [c for c in ['mb_contact_research', 'mb_email_handling', 'tat_calendar_days'] if c not in self.df_sum.columns]
             print(f"Skipping ANOVA: Missing required columns: {missing_cols}.")
             self.results['anova'] = {'error': f'Missing columns: {missing_cols}'}
             return self

        print("--- Running Analysis: ANOVA by Capability ---")
        df = self.df_sum.copy()
        # Define category: Ensure flags are int (0/1)
        df['mb_contact_research'] = df['mb_contact_research'].fillna(0).astype(int)
        df['mb_email_handling'] = df['mb_email_handling'].fillna(0).astype(int)
        df['cap_cat'] = (
            df.mb_contact_research * 1
            + df.mb_email_handling * 2
        )
        groups = {
            0: "No MB",
            1: "Contact Only",
            2: "Email Only",
            3: "Both"
        }
        samples = []
        labels = []
        for code, label in groups.items():
            # Ensure TAT is numeric, drop NAs for this group
            arr = df.loc[df.cap_cat == code, 'tat_calendar_days'].dropna().astype(float)
            if len(arr) >= 2: # Need at least 2 data points per group for ANOVA
                samples.append(arr)
                labels.append(label)
            else:
                 print(f"Info: Skipping ANOVA group '{label}' (code {code}) due to insufficient data (< 2).")

        if len(samples) < 2:
            print("Error: Not enough capability groups with sufficient data (>=2) for ANOVA.")
            self.results['anova'] = {'error': 'Not enough groups with data'}
            return self

        try:
            F, p = stats.f_oneway(*samples)
            self.results['anova'] = {
                'F_statistic': F,
                'p_value': p,
                'groups': labels # List of group labels included in the test
            }
        except Exception as e:
             print(f"Error during ANOVA calculation: {e}")
             self.results['anova'] = {'error': str(e)}

        print("Finished Analysis: ANOVA by Capability")
        return self


    def calculate_contact_efficiency(self) -> MBAnalysis:
        """
        Detailed contact-efficiency: for each contact channel (email/phone/fax),
        compute for MB vs control:
          - average # contacts provided (total_contacts)
          - yield: interactions/attempts_by_row when method matches
        Stores results in self.results['contact_efficiency'].
        """
        if self.df_sum is None:
            print("Skipping Contact Efficiency: Summary data missing.")
            return self
        # Check for necessary columns
        required_cols = ['mb_touched', 'email_count', 'phone_count', 'fax_count',
                         'email_method_count', 'email_interaction_count']
        if not all(c in self.df_sum.columns for c in required_cols):
             missing_cols = [c for c in required_cols if c not in self.df_sum.columns]
             print(f"Skipping Contact Efficiency: Missing required columns: {missing_cols}.")
             self.results['contact_efficiency'] = {'error': f'Missing columns: {missing_cols}'}
             return self

        print("--- Running Analysis: Contact Efficiency ---")
        df = self.df_sum.copy()
        df['mb'] = df.mb_touched == 1

        channels = [
            ('email', 'email_count', 'email_method_count', 'email_interaction_count'),
            ('phone', 'phone_count', None, None), # No yield calculation for phone/fax currently
            ('fax',   'fax_count',   None, None)
        ]
        ce = {}
        for name, count_col, method_col, interact_col in channels:
            out = {}
            for mb_flag in [False, True]:
                sub = df[df.mb == mb_flag]
                # avg # contacts extracted (ensure numeric and fillna)
                out_key = 'mb' if mb_flag else 'ctl'
                avg_cnt = pd.to_numeric(sub[count_col], errors='coerce').fillna(0).mean()
                out[out_key] = {'avg_extracted': avg_cnt}
                # if email channel, compute yield = interactions/methods
                if method_col and interact_col:
                    # Ensure components are numeric and filled
                    denom = pd.to_numeric(sub[method_col], errors='coerce').fillna(0)
                    numer = pd.to_numeric(sub[interact_col], errors='coerce').fillna(0)
                    # avoid divide by zero using pd.NA for replacement
                    # Calculate yield safely, replacing inf/-inf with NaN after division
                    yield_series = (numer / denom.replace({0: pd.NA})).replace([np.inf, -np.inf], np.nan)

                    out[out_key].update({
                        'avg_methods': denom.mean(),
                        'avg_interactions': numer.mean(),
                        'avg_yield': yield_series.mean() # mean ignores NaN
                    })
            ce[name] = out

        self.results['contact_efficiency'] = ce
        print("Finished Analysis: Contact Efficiency")
        return self


    def summarize_by_capability_category(self) -> MBAnalysis:
        """
        Produces category-level summaries (No MB/Contact Only/Email Only/Both)
        for key metrics.
        Stores DataFrame in self.results['category_summary'].
        """
        if self.df_sum is None:
            print("Skipping Category Summary: Summary data missing.")
            return self
        # Check for necessary columns
        required_base = ['mb_contact_research', 'mb_email_handling']
        required_metrics = ['tat_calendar_days', 'attempts_by_row', 'total_contacts']
        required_cols = required_base + required_metrics
        if not all(c in self.df_sum.columns for c in required_cols):
             missing_cols = [c for c in required_cols if c not in self.df_sum.columns]
             print(f"Skipping Category Summary: Missing required columns: {missing_cols}.")
             self.results['category_summary'] = pd.DataFrame({'error': f'Missing columns: {missing_cols}'}, index=[0])
             return self

        print("--- Running Analysis: Summary by Capability Category ---")
        df = self.df_sum.copy()
        # Define category (Ensure flags are int 0/1)
        df['mb_contact_research'] = df['mb_contact_research'].fillna(0).astype(int)
        df['mb_email_handling'] = df['mb_email_handling'].fillna(0).astype(int)
        df['cap_cat'] = (
            df.mb_contact_research * 1
            + df.mb_email_handling * 2
        )
        cat_map = {0: "No MB", 1: "Contact Only", 2: "Email Only", 3: "Both"}
        df['cap_cat_label'] = df.cap_cat.map(cat_map)

        # Define aggregations, ensuring metrics are numeric first
        agg = {}
        for m in required_metrics:
             df[m] = pd.to_numeric(df[m], errors='coerce') # Convert here before aggregation
             agg[m] = ['count', 'mean', 'std'] # Define agg functions

        try:
            summary = (
                df.groupby('cap_cat_label', observed=False) # Use observed=False to include all categories
                  .agg(agg)
            )
            # flatten columns cleanly
            summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
            summary = summary.reset_index()
            self.results['category_summary'] = summary
        except Exception as e:
             print(f"Error during category summary aggregation: {e}")
             self.results['category_summary'] = pd.DataFrame({'error': str(e)}, index=[0])

        print("Finished Analysis: Summary by Capability Category")
        return self

    # --- NEW ANALYSIS METHODS END ---

    # --------------- 4. Reporting Step ------------------
    def report(self) -> MBAnalysis:
        """Generates and saves all outputs (CSVs, JSON, Excel, Plots, Summary)."""
        if self.df_sum is None:
            print("Skipping report generation: Summary data not available.")
            return self

        print("--- Step 4: Generating Reports and Outputs ---")
        output_dir = cfg.out_dir # Use configured output directory

        # --- Persist Core DataFrames ---
        persist(self.df_sum, "summary_data", "csv", output_dir=output_dir)
        if self.df_hist is not None:
             # Optionally save a sample of history or the filtered history
             # Consider saving the filtered history used for analysis if different from original
             persist(self.df_hist.head(1000), "history_data_sample", "csv", output_dir=output_dir)

        # --- Persist Analysis Results ---
        # Save the main results dictionary (metrics, verdicts, etc.) as JSON
        # Exclude large DataFrames from the main JSON to keep it readable
        results_to_persist = {k: v for k, v in self.results.items() if not isinstance(v, pd.DataFrame) or v.shape[0] < 1000} # Example threshold
        persist(results_to_persist, "analysis_metrics_verdicts", "json", output_dir=output_dir)

        # Save specific DataFrames from results if they exist and are not empty
        detailed_results_to_save = {
            'autonomy_details': 'autonomy_details',
            'contact_plan_delta': 'contact_plan_delta',
            'category_summary': 'category_summary', # Added new result
            # Agent/Office metrics are nested, handle separately or save main dict as JSON
            # 'agent_completion_uplift': 'agent_completion_uplift', # This might be inside agent_office_metrics
            # Queue metrics are nested
            # effect_sizes and contact_efficiency are dicts, saved in main json
            # anova is a dict, saved in main json
        }
        # Save agent uplift if it exists
        agent_office_mets = self.results.get('agent_office_metrics', {})
        if isinstance(agent_office_mets, dict):
             agent_uplift = agent_office_mets.get('agent_completion_uplift')
             if isinstance(agent_uplift, pd.DataFrame) and not agent_uplift.empty:
                  persist(agent_uplift, 'agent_completion_uplift', "csv", output_dir=output_dir)

        for key, stem in detailed_results_to_save.items():
             data = self.results.get(key)
             if isinstance(data, pd.DataFrame) and not data.empty:
                 # Check if the DataFrame contains an error message
                 if 'error' not in data.columns:
                      persist(data, stem, "csv", output_dir=output_dir) # Corrected indentation
                 else:
                      print(f"Info: Skipping save for '{stem}.csv' as it contains an error message.")
             elif isinstance(data, dict) and key not in ['basic_stats', 'queue_metrics', 'agent_office_metrics', 'verdicts', 'effect_sizes', 'anova', 'contact_efficiency']: # Avoid saving large nested dicts or dicts already in main json
                  # Persist smaller dicts as JSON if needed
                  persist(data, stem, "json", output_dir=output_dir)


        # --- Generate and Save Plots ---
        print("Generating plots...")
        self._define_plot_configs() # Define plot parameters
        for plot_cfg in self.plot_configs:
            data_key = plot_cfg.pop('data_key', None) # Get data source key (e.g., 'basic_stats')
            metric_df_key = plot_cfg.pop('metric_df_key', None) # Get specific df key within data_key (e.g., 'tat_calendar_days_by_mb_touch')
            df_direct = plot_cfg.pop('df', None) # Check for directly passed df
            data_source = None

            if df_direct is not None and isinstance(df_direct, pd.DataFrame):
                 data_source = df_direct
            elif data_key:
                 # Try getting data from self.results first
                 results_data = self.results.get(data_key)
                 if metric_df_key and isinstance(results_data, dict):
                      data_source = results_data.get(metric_df_key)
                 elif isinstance(results_data, pd.DataFrame): # If data_key directly points to a DF
                      data_source = results_data
                 elif data_key == 'df_summary' and self.df_sum is not None: # Check self.df_sum
                      data_source = self.df_sum
                 else:
                      print(f"Warning: Could not find data source for key '{data_key}' (metric key: '{metric_df_key}') for plot '{plot_cfg.get('title')}'.")
            # else: # No data_key or df provided

            if data_source is not None and isinstance(data_source, pd.DataFrame) and not data_source.empty:
                 # Ensure required columns exist in the DataFrame
                 required_cols = [plot_cfg.get('x'), plot_cfg.get('y'), plot_cfg.get('hue')]
                 required_cols = [col for col in required_cols if col is not None] # Filter out None hue
                 if all(col in data_source.columns for col in required_cols):
                     create_barplot(df=data_source, **plot_cfg)
                 else:
                      missing_plot_cols = [col for col in required_cols if col not in data_source.columns]
                      print(f"Warning: Skipping plot '{plot_cfg.get('title')}' - missing columns in data source: {missing_plot_cols}")
            elif df_direct is None: # Only warn if data wasn't passed directly
                 print(f"Warning: Skipping plot '{plot_cfg.get('title')}' due to missing, empty, or invalid data source (key: {data_key}, metric: {metric_df_key}).")


        # --- Generate and Save Excel Workbook ---
        print("Generating Excel workbook...")
        excel_sheets = {"Summary": self.df_sum.copy()} # Start with a copy of summary
        # Add agent throughput sheets if available
        agent_metrics = self.results.get('agent_office_metrics', {})
        if isinstance(agent_metrics, dict):
             # Add verified counts
             for key in ['verified_by_agent_week', 'verified_by_agent_month', 'verified_by_agent_day']:
                 df_sheet = agent_metrics.get(key)
                 if isinstance(df_sheet, pd.DataFrame) and not df_sheet.empty:
                     excel_sheets[key.replace('verified_by_', '').replace('_', ' ').title()] = df_sheet.copy() # Use descriptive sheet names
             # Add monthly breakdown sheets
             monthly_breakdowns = agent_metrics.get('monthly_completion_breakdowns', {})
             if isinstance(monthly_breakdowns, dict):
                  for sheet_name, df_sheet in monthly_breakdowns.items():
                       if isinstance(df_sheet, pd.DataFrame) and not df_sheet.empty:
                            excel_sheets[f"Monthly {sheet_name}"] = df_sheet.copy()
             # Add agent uplift
             agent_uplift = agent_metrics.get('agent_completion_uplift')
             if isinstance(agent_uplift, pd.DataFrame) and not agent_uplift.empty:
                  excel_sheets["Agent Uplift"] = agent_uplift.copy()

        # Add other key results DataFrames
        other_dfs = {
            "Contact Plan Delta": self.results.get('contact_plan_delta'),
            "Autonomy Details": self.results.get('autonomy_details'),
            "Category Summary": self.results.get('category_summary'), # Added new result
            "First Day SLA": self.results.get('queue_metrics', {}).get('first_day_sla_rate'),
            "Weekly Queue Depth": self.results.get('queue_metrics', {}).get('weekly_queue_depth'),
        }
        for name, df_sheet in other_dfs.items():
             if isinstance(df_sheet, pd.DataFrame) and not df_sheet.empty:
                  excel_sheets[name] = df_sheet.copy()

        persist(excel_sheets, "analysis_workbook", "xlsx", output_dir=output_dir)

        # --- Generate and Save Executive Summary ---
        print("Generating executive summary...")
        self._generate_executive_summary(output_dir)

        print(f"--- Reporting complete. Outputs saved in subdirectories under: {output_dir} ---")
        return self

    def _define_plot_configs(self):
        """Defines configurations for standard plots."""
        self.plot_configs = []
        basic_stats_results = self.results.get('basic_stats', {})

        # --- Plot 1: TAT Comparison ---
        for tat_type in ["calendar", "business"]:
            metric_key = f"tat_{tat_type}_days_by_mb_touch"
            stats_df = basic_stats_results.get(metric_key)
            if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
                self.plot_configs.append({
                    'data_key': 'basic_stats', # Source dict
                    'metric_df_key': metric_key, # Specific df key
                    'x': 'mb_touched', 'y': 'mean',
                    'title': f'Avg TAT ({tat_type.title()} Days) - MB vs Control',
                    'ylabel': 'Avg Days', 'xlabel': 'MB Touched (0=No, 1=Yes)',
                    'fname': f'tat_{tat_type}_comparison_bar', 'label_fmt': '{:.1f}'
                })

        # --- Plot 2: Attempts Comparison ---
        metric_key = "attempts_by_row_by_mb_touch"
        stats_df = basic_stats_results.get(metric_key)
        if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
             self.plot_configs.append({
                'data_key': 'basic_stats', 'metric_df_key': metric_key,
                'x': 'mb_touched', 'y': 'mean',
                'title': 'Avg Attempts per Verification - MB vs Control',
                'ylabel': 'Avg Attempts', 'xlabel': 'MB Touched (0=No, 1=Yes)',
                'fname': 'attempts_comparison_bar', 'label_fmt': '{:.1f}'
             })

        # --- Plot 3: Autonomy Metrics ---
        autonomy_summary = self.results.get('autonomy_summary')
        if isinstance(autonomy_summary, dict) and 'error' not in autonomy_summary:
             autonomy_plot_data = [
                 {'Metric': 'Autonomy Rate (%)', 'Value': autonomy_summary.get('autonomy_rate_overall')},
                 {'Metric': 'Rework Rate (MB, %)', 'Value': autonomy_summary.get('rework_rate_mb_only')},
                 {'Metric': 'Fallback Rate (MB, %)', 'Value': autonomy_summary.get('fallback_rate_mb_only')},
                 {'Metric': 'Avg Human Touches', 'Value': autonomy_summary.get('avg_human_touches_overall')}
             ]
             autonomy_plot_df = pd.DataFrame(autonomy_plot_data).dropna(subset=['Value'])
             if not autonomy_plot_df.empty:
                 self.plot_configs.append({
                     'df': autonomy_plot_df, # Pass DF directly
                     'x': 'Metric', 'y': 'Value',
                     'title': 'Autonomy & Rework Metrics', 'ylabel': 'Value / Rate (%)',
                     'fname': 'autonomy_metrics_bar', 'label_fmt': '{:.1f}'
                 })

        # --- Plot 4: Contact Plan Delta ---
        plan_delta = self.results.get('contact_plan_delta')
        if isinstance(plan_delta, pd.DataFrame) and not plan_delta.empty and 'contact_plan_provided' in plan_delta.columns:
             delta_data = plan_delta[plan_delta['contact_plan_provided'] == True].copy() # Get the 'True' row
             if not delta_data.empty and delta_data.notna().all(axis=None): # Check if delta values are valid
                  delta_plot_data = pd.DataFrame([
                      {'Metric': 'Δ TAT (Days)', 'Value': delta_data['delta_tat_days'].iloc[0]},
                      {'Metric': 'Δ Completion (%)', 'Value': delta_data['delta_completion_rate'].iloc[0] * 100},
                      {'Metric': 'Δ Attempts', 'Value': delta_data['delta_attempts'].iloc[0]}
                  ]).dropna(subset=['Value'])
                  if not delta_plot_data.empty:
                       self.plot_configs.append({
                          'df': delta_plot_data,
                          'x': 'Metric', 'y': 'Value',
                          'title': 'Impact of MB Contact Plan (Plan vs No Plan)', 'ylabel': 'Difference',
                          'fname': 'contact_plan_delta_bar', 'label_fmt': '{:+.1f}' # Show sign
                       })

        # --- Plot 5: Open Queue Age Distribution ---
        queue_age_data = self.results.get('queue_metrics', {}).get('open_queue_age_by_mb')
        # Expecting pivoted data: index=age_bucket, cols=['Non-MB Count', 'MB Count']
        if isinstance(queue_age_data, pd.DataFrame) and not queue_age_data.empty and 'age_bucket' in queue_age_data.columns:
             try:
                 # Melt the pivoted data for plotting with hue
                 queue_melt = pd.melt(queue_age_data,
                                      id_vars='age_bucket',
                                      value_vars=['Non-MB Count', 'MB Count'],
                                      var_name='MB Touched', value_name='Count')
                 # Map names for clarity in legend
                 queue_melt['MB Touched'] = queue_melt['MB Touched'].replace({'Non-MB Count': 'Non-MB', 'MB Count': 'MB'})

                 self.plot_configs.append({
                    'df': queue_melt,
                    'x': 'age_bucket', 'y': 'Count', 'hue': 'MB Touched',
                    'title': 'Open Queue Age Distribution by MB Touch', 'xlabel': 'Age Bucket',
                    'fname': 'open_queue_age_dist_bar', 'label_fmt': '{:.0f}'
                 })
             except Exception as e:
                  print(f"Error preparing queue age data for plotting: {e}")

        # Add more plot configurations here as needed...

    def _generate_executive_summary(self, output_dir: str) -> None:
        """Generates a markdown executive summary."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        summary_parts = [
            f"# Murphy Brown Impact Analysis: Executive Summary",
            f"\n**Generated on:** {timestamp}\n",
            "## Overview",
            "This report summarizes the impact of Murphy Brown (MB) on the verification process.",
        ]

        # Helper to format numbers or return 'N/A'
        def fmt(value, precision=1, sign=False, percent=False):
            if pd.isna(value): return "N/A"
            prefix = "+" if sign and value > 0 else ""
            suffix = "%" if percent else ""
            return f"{prefix}{value:.{precision}f}{suffix}"

        # Key Findings Section
        summary_parts.append("\n## Key Findings")

        # Autonomy
        autonomy = self.results.get('autonomy_summary', {})
        if isinstance(autonomy, dict) and 'error' not in autonomy:
            summary_parts.extend([
                "\n### Autonomy & Automation",
                f"- **End-to-End Autonomy Rate (Overall):** {fmt(autonomy.get('autonomy_rate_overall'), 1, percent=True)}",
                f"- **Rework Rate (MB Touched Only):** {fmt(autonomy.get('rework_rate_mb_only'), 1, percent=True)}",
                f"- **Avg. Human Touches (Overall):** {fmt(autonomy.get('avg_human_touches_overall'), 1)}",
            ])
        else:
             summary_parts.append("\n### Autonomy & Automation\n- Data unavailable or calculation failed.")


        # Efficiency (TAT, Attempts)
        summary_parts.append("\n### Efficiency Improvements")
        stats = self.results.get('statistical_tests', {})
        tat_test = stats.get('tat_calendar_days', {})
        att_test = stats.get('attempts_by_row', {})

        if isinstance(tat_test, dict) and 'error' not in tat_test:
            mb_mean = tat_test.get('mb_mean')
            ctl_mean = tat_test.get('ctl_mean')
            tat_diff = mb_mean - ctl_mean if pd.notna(mb_mean) and pd.notna(ctl_mean) else np.nan
            sig = tat_test.get('significant', False)
            summary_parts.append(f"- **Turnaround Time (Calendar):** {fmt(tat_diff, 1, sign=True)} days "
                                 f"(MB: {fmt(mb_mean, 1)} vs CTL: {fmt(ctl_mean, 1)}) "
                                 f"{'(Significant p<'+str(cfg.significance_level)+')' if sig else '(Not Significant)'}")
        else:
             summary_parts.append("- **Turnaround Time (Calendar):** Data unavailable or test failed.")

        if isinstance(att_test, dict) and 'error' not in att_test:
            mb_mean = att_test.get('mb_mean')
            ctl_mean = att_test.get('ctl_mean')
            att_diff = mb_mean - ctl_mean if pd.notna(mb_mean) and pd.notna(ctl_mean) else np.nan
            sig = att_test.get('significant', False)
            summary_parts.append(f"- **Attempts per Verification:** {fmt(att_diff, 1, sign=True)} attempts "
                                 f"(MB: {fmt(mb_mean, 1)} vs CTL: {fmt(ctl_mean, 1)}) "
                                 f"{'(Significant p<'+str(cfg.significance_level)+')' if sig else '(Not Significant)'}")
        else:
            summary_parts.append("- **Attempts per Verification:** Data unavailable or test failed.")

        # Time Savings
        savings = self.results.get('time_savings', {})
        if isinstance(savings, dict) and 'error' not in savings:
             summary_parts.extend([
                 "\n### Resource Impact",
                 f"- **Hours Saved Per Verification:** {fmt(savings.get('hours_saved_per_verification'), 2)} hours",
                 f"- **Total Hours Saved (Estimated):** {fmt(savings.get('total_hours_saved'), 1)} hours",
                 f"- **FTE Equivalent (Estimated):** {fmt(savings.get('fte_equivalent'), 2)} FTEs (over {fmt(savings.get('analysis_period_weeks'), 1)} weeks)",
             ])
        else:
             summary_parts.append("\n### Resource Impact\n- Data unavailable or calculation failed.")

        # Verdicts
        summary_parts.append("\n### Key Verdicts")
        verdicts = self.results.get('verdicts', {})
        email_v = verdicts.get('email_performance', {})
        contact_v = verdicts.get('contact_plan', {})

        if isinstance(email_v, dict) and 'error' not in email_v:
             p_val_str = fmt(email_v.get('p_value'), 3) if email_v.get('p_value') is not None else 'N/A'
             summary_parts.append(f"- **Email Performance:** {'IMPROVED' if email_v.get('verdict') else 'NO SIGNIFICANT IMPROVEMENT'} "
                                  f"(Yield Diff: {fmt(email_v.get('diff_pp'), 1, sign=True)}pp, p={p_val_str}) "
                                  f"Reason: {email_v.get('reason', 'N/A')}")
        else:
             summary_parts.append(f"- **Email Performance:** Verdict calculation failed. Reason: {email_v.get('reason', 'Unknown')}")

        if isinstance(contact_v, dict) and 'error' not in contact_v:
             p_val_str = fmt(contact_v.get('tat_p_value'), 3) if contact_v.get('tat_p_value') is not None else 'N/A'
             summary_parts.append(f"- **Contact Plan Impact:** {'POSITIVE' if contact_v.get('overall_good') else 'MIXED / NEGATIVE'} "
                                  f"(TAT OK: {contact_v.get('tat_ok', 'N/A')}, Comp OK: {contact_v.get('completion_ok', 'N/A')}, Att OK: {contact_v.get('attempts_ok', 'N/A')}) "
                                  f"Reason: {contact_v.get('reason', 'N/A')}")
        else:
             summary_parts.append(f"- **Contact Plan Impact:** Verdict calculation failed. Reason: {contact_v.get('reason', 'Unknown')}")

        # Conclusion
        summary_parts.extend([
            "\n## Conclusion",
            "Murphy Brown demonstrates measurable impact on verification processes. Key areas of improvement include [mention specific areas like TAT reduction, attempt reduction based on results].",
            "Contact plan usage shows [positive/mixed/negative] correlation with outcomes. Email handling shows [improvement/no improvement].",
            "\n## Next Steps",
            "1. Continue monitoring performance.",
            "2. Investigate areas where impact is mixed or negative.",
            "3. Explore further automation opportunities.",
        ])

        # --- Write to file ---
        report_content = "\n".join(summary_parts)
        # Use the persist helper's logic to determine the path
        try:
            script_name = os.path.splitext(os.path.basename(__file__))[0]
        except NameError:
            script_name = "refactored_analysis"
        output_subdir = os.path.join(output_dir, script_name)
        os.makedirs(output_subdir, exist_ok=True)
        final_report_path = os.path.join(output_subdir, f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md")

        try:
            with open(final_report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Saved Markdown Summary: {final_report_path}")
        except Exception as e:
            print(f"Error saving executive summary: {e}")


# ─────────────────────── 3. COMMAND LINE INTERFACE ─────────────────

def main():
    """Main execution function: parses args, runs pipeline."""
    p = argparse.ArgumentParser(description="Murphy Brown Impact Analysis (Refactored v9.1)")
    p.add_argument("--input_file", default=cfg.csv_default,
                   help=f"Path to input CSV file (default: {cfg.csv_default})")
    p.add_argument("--output_dir", default=cfg.out_dir,
                   help=f"Base directory for output files (default: {cfg.out_dir})")
    # Add arguments to override other config values if needed
    # p.add_argument("--mb_id", default=cfg.mb_agent_identifier)
    # p.add_argument("--time_per_touch", type=int, default=cfg.time_per_touch_min)

    args = p.parse_args()

    # --- Update config from args if necessary ---
    # Example: cfg.mb_agent_identifier = args.mb_id
    # Example: cfg.time_per_touch_min = args.time_per_touch
    cfg.out_dir = args.output_dir # Allow overriding output dir via CLI

    print(f"\n=== Starting Analysis Run (v9.1 - with v8 additions) ===") # Updated run message
    print(f"Input File: {args.input_file}")
    print(f"Output Directory: {cfg.out_dir}")
    start_time = datetime.now()

    try:
        # Instantiate and run the pipeline
        analysis = MBAnalysis(args.input_file)
        (analysis.ingest()
           .augment()
           .calculate_basic_stats()        # Descriptive stats by MB touch
           .calculate_autonomy()           # Autonomy, rework, fallback (merges human_touch_count)
           .calculate_time_savings()       # Uses human_touch_count if available
           .run_statistical_tests()        # T-tests for key metrics
           .analyze_contact_plan_impact()  # Delta analysis and verdict for contact plans
           .calculate_verdicts()           # Email performance verdict
           .calculate_agent_office_metrics() # Throughput, breakdowns, uplift
           .calculate_queue_metrics()      # SLA, queue depth/age
           # --- Inserted new analyses (Verified call order) ---
           .calculate_cohens_d()           # Effect sizes
           .run_anova_by_capability()      # ANOVA by capability group
           .calculate_contact_efficiency() # Detailed contact channel yield
           .summarize_by_capability_category() # Mean metrics by capability
           # --- End new analyses ---
           .report()                       # Generate all outputs
        )

        print("\n=== Analysis Complete ===")

    except Exception as e:
        print(f"\n--- FATAL ERROR DURING ANALYSIS ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        import traceback
        traceback.print_exc()
        print("--- Analysis Halted ---")

    finally:
        end_time = datetime.now()
        print(f"Total execution time: {end_time - start_time}")


if __name__ == "__main__":
    main()
