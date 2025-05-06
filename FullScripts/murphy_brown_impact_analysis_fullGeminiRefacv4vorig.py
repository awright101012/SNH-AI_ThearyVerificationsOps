#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
murphy_brown_impact_analysis_full.py

A comprehensive analysis script to quantify Murphy Brown's impact on the verification process,
focusing on both contact research and email handling capabilities.

Usage:
    python murphy_brown_impact_analysis_full.py [--input_file PATH_TO_CSV] [--output_dir PATH_TO_OUTPUT]
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

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

# --- Configuration & Constants ---
CSV_FILE_PATH_DEFAULT = r"C:\Users\awrig\Downloads\Fullo3Query.csv"
MB_AGENT_IDENTIFIER = "murphy.brown"  # Agent identifier (case-insensitive)
EXCLUDED_OFFICE = "SAT-HEX"  # Office to exclude for productivity

# Patterns to identify MB work in notes
MB_EMAIL_PATTERN = r"<div style=\"color:red\">Email received from"
MB_CONTACT_PATTERN = r"Contacts found from research \(ranked by most viable first\)"
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
CONTACT_PLAN_PHRASE = "contacts found" # Phrase indicating a contact plan note
CONTACT_KEYWORDS = ['phone:', 'email:', 'fax:'] # Keywords to count as contacts

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
    # 1️⃣  core timestamps
    # ---------------------------------------------------------------------------
    grp = df.groupby("searchid", as_index=False)

    core = (
        grp[ts_col].agg(first_attempt_time="min", last_attempt_time="max")
        .merge(
            # completiondate = *latest* hist-datetime where status == REVIEW
            df[df[status_col].astype(str).str.upper() == finished_status]
              .groupby("searchid", as_index=False)[ts_col]
              .max()
              .rename(columns={ts_col: "completiondate"}),
            how="left", on="searchid"
        )
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
            start_date = row.first_attempt_time.date()
            end_date = row.completiondate.date()
            # Ensure dates are valid before casting
            start_dt64 = np.datetime64(start_date, 'D')
            end_dt64 = np.datetime64(end_date, 'D') + np.timedelta64(1, 'D')
            return np.busday_count(
                start_dt64,
                end_dt64,
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

    try:
        # 1. Load ALL columns initially, let pandas infer basic types
        # Use error_bad_lines=False if rows might have incorrect number of fields
        # Use warn_bad_lines=True to see warnings about skipped rows
        df = pd.read_csv(
            file_path,
            low_memory=False,
            # on_bad_lines='warn' # Consider adding this to see warnings for parsing issues
            # engine='pyarrow' # Optional: uncomment if pandas >= 2.0 and pyarrow is installed
        )
        print(f"Initial load complete. Found {len(df)} rows and columns: {list(df.columns)}")

        # 2. Standardize column names to lowercase
        original_columns = list(df.columns)
        df.columns = df.columns.str.lower().str.strip() # Lowercase and strip whitespace
        standardized_columns = list(df.columns)
        if original_columns != standardized_columns:
            print(f"Standardized columns to: {standardized_columns}")
        else:
            print("Column names already seem standardized (lowercase, no leading/trailing spaces).")

        # 3. Define desired columns and types (keep original definitions)
        date_cols = [
            'historydatetime', 'completiondate', 'postdate', 'resultsentdate',
            'last_update', 'last_attempt', 'qadate', 'first_attempt'
        ]
        # Using nullable types where appropriate
        dtypes = {
            'searchid': 'Int64', # Use nullable integer
            'historyid': 'Int64', # Use nullable integer
            'resultid': 'Int16',
            'userid': 'string',
            'username': 'string',
            'agentname': 'string',
            'note': 'string',
            'searchstatus': 'string',
            'searchtype': 'string',
            'contactmethod': 'string'
        }
        # Define all columns potentially needed downstream
        all_desired_cols = list(set(date_cols + list(dtypes.keys()) + ['searchid', 'agentname', 'username', 'note', 'historydatetime', 'searchstatus', 'searchtype', 'resultid', 'historyid', 'completiondate', 'contactmethod']))


        # 4. Identify which desired columns ACTUALLY exist in the loaded DataFrame
        available_cols = [col for col in all_desired_cols if col in df.columns]
        missing_desired_cols = [col for col in all_desired_cols if col not in df.columns]

        if missing_desired_cols:
            print(f"Warning: The following desired columns were NOT found in the CSV: {missing_desired_cols}")

        print(f"Columns available for processing: {available_cols}")

        # Keep only the available desired columns
        df = df[available_cols].copy() # Use copy() to avoid SettingWithCopyWarning on subsequent ops

        # 5. Apply specific dtypes to existing columns
        print("Applying data types...")
        for col, dtype in dtypes.items():
            if col in df.columns:
                try:
                    # Handle potential large numbers or NaNs when converting to numeric/int
                    if dtype.startswith('Int'): # For nullable integers
                         # Convert to numeric first (handles non-numeric as NaN), then nullable Int
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                    elif pd.api.types.is_integer_dtype(dtype): # For standard numpy int (less robust to NAs)
                         # Attempt conversion, may fail if NaNs are present after coerce
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                    else: # For string, float, etc.
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column '{col}' to dtype {dtype}. Error: {e}. Keeping original type.")

        # 6. Parse date columns that exist
        print("Parsing date columns...")
        for col in date_cols:
             if col in df.columns:
                # Check if it's already datetime to avoid re-parsing
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        # if df[col].isnull().any():
                             # print(f"Note: Column '{col}' had values that could not be parsed to dates (set to NaT).")
                    except Exception as e:
                         print(f"Warning: Could not parse date column '{col}'. Error: {e}. Keeping original type.")
                else:
                    print(f"Column '{col}' already parsed as datetime.")


        print(f"Finished processing columns. Final DataFrame shape: {df.shape}. Columns: {list(df.columns)}")

        # 7. Validate essential columns are present *after* processing
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
        return 0.0 if mean_diff == 0 else np.inf * np.sign(mean_diff)
    
    # Handle potential division by zero if s_pooled is extremely small
    if np.isclose(s_pooled, 0):
        return 0.0 if np.isclose(mean_diff, 0) else np.inf * np.sign(mean_diff)
    
    return mean_diff / s_pooled


# --- Core Metric Calculation Functions ---

def calculate_attempt_count_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate attempt count metrics per search ID.
    
    Args:
        df: History DataFrame
    
    Returns:
        DataFrame with attempt metrics by searchid
    """
    print("Calculating attempt count metrics...")
    
    required_cols = ['searchid', 'historyid', 'resultid']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for attempt count metrics: {missing}")
    
    # Ensure resultid is numeric for comparison
    df['resultid'] = pd.to_numeric(df['resultid'], errors='coerce')
    
    grouped = df.groupby('searchid')
    attempts_by_row = grouped.size().rename('attempts_by_row')
    
    # Calculate distinct applicant contacts (resultid 16)
    applicant_contact_count = grouped.apply(
        lambda x: x.loc[x['resultid'] == 16, 'historyid'].nunique()
    ).rename('distinct_applicant_contact_count')
    
    # Combine metrics
    agg_metrics = pd.concat([attempts_by_row, applicant_contact_count], axis=1)
    agg_metrics['distinct_applicant_contact_count'] = agg_metrics['distinct_applicant_contact_count'].fillna(0).astype('Int64')
    
    # Calculate maxminusapplicant
    agg_metrics['maxminusapplicant'] = (
        agg_metrics['attempts_by_row'] - agg_metrics['distinct_applicant_contact_count']
    ).clip(lower=0).astype('Int64')  # Ensure non-negative and integer
    
    print("Attempt count metrics calculation complete.")
    return agg_metrics.reset_index()


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
    
    # Check notes for contact research pattern
    if 'note' in df_history.columns:
        # Find searches with MB contact research
        mask = df_history['note'].astype(str).str.contains(MB_CONTACT_PATTERN, na=False)
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
        agg_contacts = contact_details.groupby(df_history[mask]['searchid']).max()
        
        # Merge with main contact DataFrame
        contact_df = pd.merge(contact_df, agg_contacts, on='searchid', how='left')
        
        # Fill NaN values with 0
        for col in ['email_count', 'phone_count', 'fax_count', 'total_contacts']:
            if col in contact_df.columns:
                contact_df[col] = contact_df[col].fillna(0).astype('Int64') # Use nullable integer
            else:
                contact_df[col] = 0
    else:
        print("Warning: 'note' column not found in history data. Cannot analyze contact research.")
        # Initialize contact count columns with 0
        contact_df['email_count'] = 0
        contact_df['phone_count'] = 0
        contact_df['fax_count'] = 0
        contact_df['total_contacts'] = 0
    
    print(f"Found {contact_df['mb_contact_research'].sum()} searches with MB contact research.")
    return contact_df


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
        mask = df_history['note'].astype(str).str.contains(MB_EMAIL_PATTERN, na=False)
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
        'tat_calendar_days', 'tat_business_days', 'attempts_by_row',
        'distinct_applicant_contact_count', 'maxminusapplicant'
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
        groups = []
        for group in df_summary['capability_group'].unique():
            group_data = df_summary[df_summary['capability_group'] == group][metric].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
        
        # Only perform ANOVA if we have at least 2 groups with data
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                anova_results[metric] = {
                    'F_statistic': f_stat,
                    'p_value': p_val,
                    'significant': p_val < SIGNIFICANCE_LEVEL # Use defined constant
                }
            except Exception as e:
                print(f"Error performing ANOVA for {metric}: {e}")
                anova_results[metric] = {
                    'F_statistic': np.nan,
                    'p_value': np.nan,
                    'significant': np.nan # Indicate failure
                }
        else:
            print(f"Insufficient data for ANOVA on {metric} (Groups: {len(groups)}, Min Size: {min([len(g) for g in groups], default=0)})") # More informative message
            anova_results[metric] = {
                'F_statistic': np.nan,
                'p_value': np.nan,
                'significant': np.nan # Indicate insufficient data
            }
    
    # Create ANOVA summary DataFrame
    anova_df = pd.DataFrame([
        {
            'Metric': metric,
            'F_Statistic': results['F_statistic'],
            'P_Value': results['p_value'],
            'Significant': results['significant']
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
    df_history['is_human_touch'] = ~df_history['is_mb_row']

    # Clean up temporary columns
    df_history.drop(columns=['agentname_lower', 'username_lower'], inplace=True)

    # 2. Sort data for shift operations
    # Ensure historydatetime is datetime before sorting
    if not pd.api.types.is_datetime64_any_dtype(df_history['historydatetime']):
        df_history['historydatetime'] = pd.to_datetime(df_history['historydatetime'], errors='coerce')
    df_history_sorted = df_history.sort_values(['searchid', 'historydatetime']).copy()

    # 3. Calculate human touch count per searchid
    human_touches = (
        df_history_sorted[df_history_sorted['is_human_touch']]
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
        median_tat_business=('tat_business_days', 'median')
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
            median_tat_business=('tat_business_days', 'median')
        ).reset_index()
    else:
        capability_stats = None
    
    # Statistical test for TAT differences
    # ensure dtype float for t-test BEFORE creating subsets
    # Convert relevant columns to numeric, coercing errors
    completed_df['tat_calendar_days'] = pd.to_numeric(completed_df['tat_calendar_days'], errors='coerce')
    completed_df['tat_business_days'] = pd.to_numeric(completed_df['tat_business_days'], errors='coerce')

    mb_completed = completed_df[completed_df['mb_touched'] == 1]
    non_mb_completed = completed_df[completed_df['mb_touched'] == 0]
    
    stat_results = {}
    for metric in ['tat_calendar_days', 'tat_business_days']:
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
    required_cols = ['searchid', 'mb_touched', 'attempts_by_row', 'is_completed']
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
    
    # 1. Calculate outbound attempts per verification (attempts_by_row)
    attempts_stats = completed_df.groupby('mb_touched').agg(
        search_count=('searchid', 'count'),
        avg_attempts=('attempts_by_row', 'mean'),
        median_attempts=('attempts_by_row', 'median')
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
    
    # 3. Calculate yield metrics if email interaction data is available
    if 'email_interaction_count' in completed_df.columns and 'email_method_count' in completed_df.columns:
        # Calculate success rate for email channel (emails with responses)
        completed_df['email_yield'] = np.where(
            completed_df['email_method_count'] > 0,
            completed_df['email_interaction_count'] / completed_df['email_method_count'],
            0
        )
        
        yield_stats = completed_df.groupby('mb_touched').agg(
            avg_email_yield=('email_yield', 'mean'),
            email_attempts=('email_method_count', 'sum'),
            email_responses=('email_interaction_count', 'sum')
        ).reset_index()
        
        # Calculate overall channel yield
        for row in yield_stats.itertuples():
            if row.email_attempts > 0:
                mb_status = "MB" if row.mb_touched == 1 else "non-MB"
                overall_yield = row.email_responses / row.email_attempts * 100
                print(f"Email channel yield for {mb_status}: {overall_yield:.2f}% ({row.email_responses}/{row.email_attempts})")
    else:
        yield_stats = None
        print("Warning: Cannot calculate channel yield metrics - missing email interaction data")
    
    # 4. Statistical tests for attempts
    mb_completed = completed_df[completed_df['mb_touched'] == 1]
    non_mb_completed = completed_df[completed_df['mb_touched'] == 0]
    
    stat_results = {}
    for metric in ['attempts_by_row']:
        if metric not in completed_df.columns:
            continue
            
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
    
    return {
        'attempts_stats': attempts_stats,
        'contact_stats': contact_stats,
        'yield_stats': yield_stats,
        'statistical_tests': stat_results
    }


def estimate_agent_hours_saved(df_summary, autonomy_results, df_history, time_per_touch=5):
    """
    Estimate the number of agent hours saved by Murphy Brown.
    
    Uses df_summary (with human_touch_count) and df_history (for date range).
    
    Args:
        df_summary: Summary DataFrame (must include human_touch_count & mb_touched)
        autonomy_results: Output of calculate_mb_autonomy_metrics
        df_history: Raw history DataFrame (needed for date range)
        time_per_touch: Average time in minutes per human touch
    Returns:
        Dict with time savings & FTE equivalent
    """
    print("Estimating agent hours saved...")
    time_per_touch_hours = time_per_touch / 60

    if 'human_touch_count' not in df_summary.columns:
        print("Error: 'human_touch_count' not found in summary—skipping time savings")
        return None
    if 'mb_touched' not in df_summary.columns:
        print("Error: 'mb_touched' column required for time saving estimation")
        return None

    # Split MB vs non-MB
    mb_df     = df_summary[df_summary['mb_touched'] == 1]
    non_mb_df = df_summary[df_summary['mb_touched'] == 0]
    if mb_df.empty or non_mb_df.empty:
        print("Warning: Cannot calculate time savings—missing MB or non-MB rows")
        return None

    # Ensure count is float for mean calculation, handle potential NaNs introduced earlier
    av_mb  = mb_df['human_touch_count'].astype(float).mean()
    av_non = non_mb_df['human_touch_count'].astype(float).mean()

    # Check if means are NaN (can happen if all inputs were NaN or if groups were empty after filtering NAs)
    if pd.isna(av_mb) or pd.isna(av_non):
         print("Warning: Could not calculate average touches for MB or non-MB groups. Skipping time savings.")
         return None

    reduction = av_non - av_mb
    hrs_per_ver = reduction * time_per_touch_hours
    count_mb   = len(mb_df)
    total_hrs  = hrs_per_ver * count_mb

    print(f"Average human touches: {av_mb:.2f} (MB) vs {av_non:.2f} (non-MB)")
    print(f"Reduction per verification: {reduction:.2f}")
    print(f"Hours saved per verification: {hrs_per_ver:.2f}")
    print(f"Total hours saved across {count_mb}: {total_hrs:.2f}")

    # FTE calculation over the observed date range
    weeks = 1; weekly_saved = total_hrs; fte = np.nan # Initialize defaults
    if 'historydatetime' in df_history.columns and pd.api.types.is_datetime64_any_dtype(df_history['historydatetime']):
        dates = df_history['historydatetime'].dropna()
        if not dates.empty:
            # Ensure max >= min, otherwise days could be negative
            min_date, max_date = dates.min(), dates.max()
            if max_date >= min_date:
                days = (max_date - min_date).days
                # Use max(1, days) to prevent division by zero if min/max are same day
                weeks = max(1, days) / 7
                if weeks > 0:
                    weekly_saved = total_hrs / weeks
                    fte = weekly_saved / 40 # Assuming 40-hour work week
                else: # Should not happen with max(1, days) but as safety
                    weekly_saved = total_hrs # Assign total if weeks is 0 or less
                    fte = np.nan
            else: # Handle case where max_date < min_date (shouldn't happen with proper data)
                 print("Warning: Max date is earlier than min date in historydatetime. Cannot calculate FTE accurately.")
        else: # Handle case where dates is empty after dropna
            print("Warning: No valid dates found in historydatetime after dropping NaNs.")
    else:
        print("Warning: Cannot calculate FTE—invalid or missing historydatetime.")

    print(f"Equivalent FTE savings: {fte:.2f} (based on {weekly_saved:.2f} hrs/week over {weeks:.1f} weeks)")
    return {
        'avg_touches_mb':          av_mb,
        'avg_touches_non_mb':      av_non,
        'touch_reduction':         reduction,
        'hours_saved_per_verification': hrs_per_ver,
        'mb_verification_count':   count_mb,
        'total_hours_saved':       total_hrs,
        'weekly_hours_saved':      weekly_saved,
        'fte_equivalent':          fte
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
        'tat_calendar_days', 'tat_business_days', 'attempts_by_row',
        'distinct_applicant_contact_count', 'maxminusapplicant'
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
def contact_plan_delta(df_summary, contact_plan_flags):
    """
    Merge the plan flags into df_summary and return simple deltas:
        • completion-rate
        • avg TAT (calendar)
        • avg attempts
    """
    merged = df_summary.merge(
        contact_plan_flags[['searchid', 'contact_plan_provided']],
        on='searchid', how='left'
    )
    merged['contact_plan_provided'] = merged['contact_plan_provided'].fillna(False)

    grp = merged.groupby('contact_plan_provided')
    out = grp.agg(
        search_count           = ('searchid', 'size'),
        completion_rate        = ('is_completed', 'mean'),
        avg_tat_days           = ('tat_calendar_days', 'mean'),
        avg_attempts_by_row    = ('attempts_by_row', 'mean')
    ).reset_index()

    # Δ columns (plan – no-plan)
    if len(out) == 2:
        # Ensure boolean column exists before accessing .values
        if 'contact_plan_provided' in out.columns:
            row_with_mask = out['contact_plan_provided'] == True
            row_without_mask = out['contact_plan_provided'] == False
            
            # Check if both True and False groups exist
            if row_with_mask.any() and row_without_mask.any():
                row_with = out.loc[row_with_mask]
                row_without = out.loc[row_without_mask]
                
                # Safely access values only if rows exist
                if not row_with.empty and not row_without.empty:
                     out['delta_completion_rate'] = row_with['completion_rate'].values[0]  - row_without['completion_rate'].values[0]
                     out['delta_tat_days']        = row_with['avg_tat_days'].values[0]     - row_without['avg_tat_days'].values[0]
                     out['delta_attempts']        = row_with['avg_attempts_by_row'].values[0] - row_without['avg_attempts_by_row'].values[0]
                else:
                    print("Warning: Could not find both 'with plan' and 'without plan' groups for delta calculation.")
                    out['delta_completion_rate'] = np.nan
                    out['delta_tat_days']        = np.nan
                    out['delta_attempts']        = np.nan
            else:
                 print("Warning: Missing either 'with plan' or 'without plan' group for delta calculation.")
                 out['delta_completion_rate'] = np.nan
                 out['delta_tat_days']        = np.nan
                 out['delta_attempts']        = np.nan
        else:
            print("Warning: 'contact_plan_provided' column not found for delta calculation.")
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
    plt.style.use('seaborn-v0_8-darkgrid')
    
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


# --- Main Analysis Function ---

def perform_mb_impact_analysis(df_history):
    """
    Perform comprehensive analysis of Murphy Brown's impact using consolidated aggregations.

    Args:
        df_history: History DataFrame, pre-filtered for relevant search types.

    Returns:
        Dictionary with all analysis results
    """
    print("\n=== Starting Murphy Brown Impact Analysis ===")

    # --- ADDED: Check essential columns early ---
    required_cols = {
        'searchid', 'historydatetime', 'agentname', 'username', 'note',
        'searchstatus', 'resultid', 'contactmethod'
    }
    if not required_cols.issubset(df_history.columns):
        missing = required_cols.difference(df_history.columns)
        print(f"Error: Missing essential columns in df_history: {missing}. Aborting analysis.")
        return None
    # --- END CHECK ---

    # --- 1. Pre-calculations and Flagging on df_history ---
    print("Step 1: Pre-calculating flags on history data...")

    # Ensure necessary columns have correct types
    df_history['note'] = df_history['note'].fillna('').astype(str)
    df_history['agentname'] = df_history['agentname'].fillna('').astype(str)
    df_history['username'] = df_history['username'].fillna('').astype(str)
    df_history['resultid'] = pd.to_numeric(df_history['resultid'], errors='coerce').astype('Int16')
    df_history['contactmethod'] = df_history['contactmethod'].fillna('').astype(str).str.lower()
    df_history['searchstatus'] = df_history['searchstatus'].fillna('').astype(str).str.upper()
    # Ensure historydatetime is datetime for the TAT helper
    df_history['historydatetime'] = pd.to_datetime(df_history['historydatetime'], errors='coerce')

    # Cache MB agent pattern
    mb_pattern = MB_AGENT_IDENTIFIER.lower()

    # a) MB row flag
    df_history['is_mb_row'] = (
        df_history['agentname'].str.lower().str.contains(mb_pattern, na=False) |
        df_history['username'].str.lower().str.contains(mb_pattern, na=False)
    )

    # b) MB contact research flag (using regex)
    df_history['is_mb_contact_research_note'] = df_history['note'].str.contains(MB_CONTACT_PATTERN, na=False)

    # c) MB email handling flag (using regex)
    df_history['is_mb_email_handling_note'] = df_history['note'].str.contains(MB_EMAIL_PATTERN, na=False)

    # d) Applicant contact flag (Result ID 16)
    df_history['is_applicant_contact'] = (df_history['resultid'] == 16)

    # e) Email contact method flag
    df_history['is_email_contact_method'] = (df_history['contactmethod'] == 'email')

    # f) Completion status flag (used by TAT helper)
    df_history['is_completion_status'] = (df_history['searchstatus'] == 'REVIEW') # Kept for TAT helper

    # g) Extract contact details from MB contact research notes (Vectorized)
    print("Step 1b: Extracting contact details from notes (vectorized)...")
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
    df_history['email_interaction_instances'] = df_history['note'].str.count(MB_EMAIL_PATTERN)
        
    # --- 1c. Calculate TAT Block using Helper --- 
    print("Step 1c: Calculating completion and TAT metrics using helper...")
    tat_block = add_completion_and_tat(
        df_history,
        holidays=HOLIDAYS,
        status_col="searchstatus",
        finished_status="REVIEW" # <-- Use REVIEW as the completion status
    )

    # --- 2. Consolidated Aggregation by Search ID --- 
    print("Step 2: Performing consolidated aggregation by searchid...")
    
    # Define aggregation dictionary (REMOVED time/completion fields)
    agg_dict = {
        # Attempt Counts
        'attempts_by_row': ('historyid', 'size'), # Count rows per searchid
        'distinct_applicant_contact_count': ('is_applicant_contact', 'sum'), # Sum boolean flag
        
        # MB Touch & Capability Flags (take max, as True > False)
        'mb_touched': ('is_mb_row', 'max'),
        'mb_contact_research': ('is_mb_contact_research_note', 'max'),
        'mb_email_handling': ('is_mb_email_handling_note', 'max'),
        
        # Contact Counts (Sum counts extracted per note)
        'email_count': ('emails_in_note', 'sum'),
        'phone_count': ('phones_in_note', 'sum'),
        'fax_count': ('faxes_in_note', 'sum'),
        'total_contacts': ('total_contacts_in_note', 'sum'),
        
        # Email Metrics
        'email_interaction_count': ('email_interaction_instances', 'sum'), 
        'email_method_count': ('is_email_contact_method', 'sum'), # Count rows where method was email
    }

    # Remove aggregations if the source column doesn't exist in df_history
    agg_dict_filtered = {k: v for k, v in agg_dict.items() if v[0] in df_history.columns}

    # Perform the aggregation
    df_summary = df_history.groupby('searchid').agg(**agg_dict_filtered).reset_index()
    print(f"Aggregation complete. Summary table shape (before TAT merge): {df_summary.shape}")
   
    # --- Merge the TAT block ---
    df_summary = df_summary.merge(tat_block, on="searchid", how="left")
    print(f"Merged TAT block. Summary table shape (after TAT merge): {df_summary.shape}")

    # --- 3. Post-aggregation Calculations & Clean-up ---
    print("Step 3: Performing post-aggregation calculations...")

    # a) Calculate maxminusapplicant
    if 'attempts_by_row' in df_summary.columns and 'distinct_applicant_contact_count' in df_summary.columns:
        df_summary['maxminusapplicant'] = (
            df_summary['attempts_by_row'] - df_summary['distinct_applicant_contact_count']
        ).clip(lower=0).astype('Int64')
    else:
        df_summary['maxminusapplicant'] = pd.NA # Indicate it couldn't be calculated

    # b) REMOVED: Completion status and date calculation (handled by add_completion_and_tat)
    # c) REMOVED: TAT metric calculation (handled by add_completion_and_tat)

    # d) Convert boolean flags (Max gives True/False) to int (1/0)
    bool_flags = ['mb_touched', 'mb_contact_research', 'mb_email_handling']
    for flag in bool_flags:
        if flag in df_summary.columns:
            df_summary[flag] = df_summary[flag].astype(int)

    # e) Fill NA values in count columns resulting from aggregation (if any)
    count_cols = ['attempts_by_row', 'distinct_applicant_contact_count', 
                  'email_count', 'phone_count', 'fax_count', 'total_contacts',
                  'email_interaction_count', 'email_method_count']
    for col in count_cols:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].fillna(0).astype('Int64') # Use nullable integer

    # --- Step 4: Calculate Downstream Metrics using df_summary --- 
    print("Step 4: Calculating downstream analysis metrics...")

    # a) Calculate autonomy metrics (requires df_history with flags and the new df_summary)
    # Pass the original df_history which now contains is_mb_row and is_human_touch
    autonomy_results = calculate_mb_autonomy_metrics(df_history, df_summary)
    if autonomy_results and 'autonomy_df' in autonomy_results:
        # Merge autonomy metrics into summary
        autonomy_cols_to_merge = ['searchid', 'human_touch_count', 'fully_autonomous', 'has_rework', 'had_fallback']
        # Ensure the columns exist in the autonomy_df before trying to select them
        existing_autonomy_cols = [col for col in autonomy_cols_to_merge if col in autonomy_results['autonomy_df'].columns]
        if existing_autonomy_cols:
             df_summary = pd.merge(
                 df_summary,
                 autonomy_results['autonomy_df'][existing_autonomy_cols],
                 on='searchid', how='left'
             )
             # Fill NA for autonomy metrics if merge introduced them (shouldn't if keys align)
             for col in existing_autonomy_cols:
                 if col != 'searchid' and col in df_summary.columns and df_summary[col].isnull().any():
                     # Typically fill with 0 for counts/flags, maybe handle differently if needed
                     df_summary[col] = df_summary[col].fillna(0).astype('Int64') # Use nullable integer
        else:
            print("Warning: No autonomy columns found to merge.")
    else:
        print("Warning: Autonomy calculation failed or returned no results.")

    # b) Calculate time efficiency metrics
    time_efficiency_results = calculate_time_efficiency_metrics(df_summary)

    # c) Calculate contact efficiency metrics
    contact_efficiency_results = calculate_contact_efficiency_metrics(df_summary)

    # --- 🚀 NEW: Outbound Email/Fax Automation Rate (Phase 3A) ---
    print("Step X: Calculating outbound automation rate for email & fax…")
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
    # Success if resp_time ≤ end_time of 2nd attempt
    summary_out['auto_complete'] = (
        summary_out['resp_time'] <= summary_out['end_time']
    ).fillna(False)

    # 5. Compute rate (Removed redundant print)
    # automation_rate = summary_out['auto_complete'].mean()
    # print(f"Outbound automation rate (email / fax): {automation_rate:.1%}")

    # --- Minimal Key Metrics (Phase 3A) ---
    # summary_out: contains searchid, auto_complete (bool), resp_time, end_time
    # out:          the DataFrame of the first two attempts per search with contactmethod

    total = len(summary_out)
    automated = summary_out['auto_complete'].sum()

    # 1. Automation Rate
    automation_rate = automated / total if total > 0 else 0 # Handle division by zero

    # 2. Reassignment Rate
    reassignment_rate = 1 - automation_rate

    # 3. Time to First Response (hours)
    # Ensure 'out' is not empty before grouping
    if not out.empty:
        first_times = out.groupby('searchid')['historydatetime'].first().rename('first_time')
        resp_times  = summary_out.set_index('searchid')['resp_time']
        # Align indices before subtraction and handle cases where keys might not match
        resp_times, first_times = resp_times.align(first_times, join='inner')
        if not resp_times.empty and not first_times.empty:
            response_hours = (resp_times - first_times).dt.total_seconds() / 3600
            avg_response_time = response_hours.mean()
        else:
            print("Warning: Could not calculate response time (no matching records between responses and first attempts).")
            avg_response_time = np.nan
    else:
        print("Warning: 'out' DataFrame is empty. Cannot calculate response time.")
        avg_response_time = np.nan


    # 4. Channel Success Rate
    email_ids = []
    fax_ids = []
    if not out.empty and 'contactmethod' in out.columns:
         email_ids = out.loc[out['contactmethod']=='email', 'searchid'].unique()
         fax_ids   = out.loc[out['contactmethod']=='fax',   'searchid'].unique()

    # Calculate rates safely, handling cases where IDs might not exist in summary_out index
    email_success_rate = np.nan
    if len(email_ids) > 0:
        valid_email_ids = summary_out.set_index('searchid').index.intersection(email_ids)
        if len(valid_email_ids) > 0:
            email_success_rate = summary_out.set_index('searchid').loc[valid_email_ids,   'auto_complete'].mean()
        else:
             print("Warning: No email attempts found in the summary data for success rate calculation.")

    fax_success_rate = np.nan
    if len(fax_ids) > 0:
        valid_fax_ids = summary_out.set_index('searchid').index.intersection(fax_ids)
        if len(valid_fax_ids) > 0:
            fax_success_rate = summary_out.set_index('searchid').loc[valid_fax_ids,     'auto_complete'].mean()
        else:
             print("Warning: No fax attempts found in the summary data for success rate calculation.")


    # Print all four
    print(f"Automation Rate:         {automation_rate:.1%}")
    print(f"Reassignment Rate:       {reassignment_rate:.1%}")
    print(f"Avg Time to Response:    {avg_response_time:.1f} hours")
    print(f"Email Success Rate:      {email_success_rate:.1%}")
    print(f"Fax Success Rate:        {fax_success_rate:.1%}")
    # --- end of Phase 3A automation check ---

    # d) Estimate agent hours saved (requires df_history for date range)
    time_savings_results = estimate_agent_hours_saved(
        df_summary, # Now contains human_touch_count
        autonomy_results,
        df_history, # Pass original df_history for date range
        time_per_touch=5
    )

    # e) Analyze impact by MB capability category
    capability_impact_results = analyze_mb_capability_impact(df_summary, extended_metrics=True)

    # f) Calculate metrics by MB category
    category_results = calculate_mb_category_metrics(df_summary)

    # g) Analyze MB contact plan impact (uses df_history)
    contact_plan_results = analyze_mb_contact_plan_impact(df_history, MB_AGENT_IDENTIFIER)

    # --- 5. Compile Results & Save/Visualize --- 
    print("Step 5: Compiling results and generating outputs...")
    # Initialize all_results dictionary EARLIER
    all_results = {
        'df_summary': df_summary,
        'autonomy': autonomy_results,
        'time_efficiency': time_efficiency_results,
        'contact_efficiency': contact_efficiency_results,
        'time_savings': time_savings_results,
        'capability_impact': capability_impact_results,
        'category_metrics': category_results,
        # Add placeholders for potentially calculated metrics
        'contact_plan_delta': None,
        'mb_contact_plan_weekly': None,
        'mb_contact_plan_monthly': None,
        'attempt_bucket_stats': None,
    }


    # --- Δ metrics for searches with vs without MB contact-plan --------------------
    if contact_plan_results and 'plan_flags' in contact_plan_results:
        # Ensure df_summary has required columns before calling delta function
        required_summary_cols = ['searchid', 'is_completed', 'tat_calendar_days', 'attempts_by_row']
        if all(col in df_summary.columns for col in required_summary_cols):
             plan_delta = contact_plan_delta(df_summary, contact_plan_results['plan_flags'])
             all_results['contact_plan_delta'] = plan_delta # Now all_results exists
             save_results(plan_delta, 'mb_contact_plan_delta')
        else:
            missing_cols = [col for col in required_summary_cols if col not in df_summary.columns]
            print(f"Warning: Cannot calculate contact plan delta. Missing columns in df_summary: {missing_cols}")
            plan_delta = None # Keep plan_delta as None if calculation fails
    else:
        print("Warning: Contact plan results or flags not available for delta calculation.")
        plan_delta = None # Keep plan_delta as None if calculation fails
    # --- END ADDED ---

    # Assign contact plan results if they exist
    if contact_plan_results:
        all_results['mb_contact_plan_weekly'] = contact_plan_results.get('weekly_summary')
        all_results['mb_contact_plan_monthly'] = contact_plan_results.get('monthly_summary')

    # Save key dataframes
    save_results(df_summary, "mb_analysis_summary")
    if 'mb_contact_plan_weekly' in all_results and all_results['mb_contact_plan_weekly'] is not None:
        save_results(all_results['mb_contact_plan_weekly'], "mb_contact_plan_weekly")
    if 'mb_contact_plan_monthly' in all_results and all_results['mb_contact_plan_monthly'] is not None:
        save_results(all_results['mb_contact_plan_monthly'], "mb_contact_plan_monthly")
    # --- ADDED: Check 'attempt_bucket_stats' existence before saving ---
    # Save attempt bucket stats if available and not empty
    if 'attempt_bucket_stats' in all_results and all_results['attempt_bucket_stats'] is not None and not all_results['attempt_bucket_stats'].empty:
        save_results(all_results['attempt_bucket_stats'], "mb_attempt_bucket_stats")
    # --- END ADDED ---

    # Create Detailed Sample Output (using df_history and the final df_summary)
    print("\n--- Creating Detailed Sample for Verification (Last 30 Days) ---")
    if 'historydatetime' in df_history.columns and pd.api.types.is_datetime64_any_dtype(df_history['historydatetime']):
        max_hist_date = df_history['historydatetime'].max()
        if pd.notna(max_hist_date):
            start_date_30_days = max_hist_date - pd.Timedelta(days=30)
            print(f"Filtering history for detailed sample (dates >= {start_date_30_days.date()})")
            df_history_last_30_days = df_history[
                (df_history['historydatetime'] >= start_date_30_days) & df_history['historydatetime'].notna()
            ].copy()

            if not df_history_last_30_days.empty:
                # Merge with the final summary dataframe
                summary_cols_to_merge = [col for col in df_summary.columns if col != 'searchid']
                df_verification_sample = pd.merge(
                    df_history_last_30_days,
                    df_summary[['searchid'] + summary_cols_to_merge],
                    on='searchid',
                    how='left',
                    suffixes=('_hist', '') # Suffix original history cols if they overlap
                )
                print(f"Created detailed verification sample with {len(df_verification_sample)} rows.")
                save_results(df_verification_sample, "detailed_sample_last_30d")
            else:
                print("Warning: No history records found within the last 30 days for detailed sample.")
        else:
            print("Warning: Could not determine max history date for detailed sample.")
    else:
        print("Warning: 'historydatetime' column missing or invalid in df_history. Cannot create 30-day sample.")

    # Create visualizations
    dashboard_dir = create_mb_impact_dashboard(all_results)
    all_results['dashboard_dir'] = dashboard_dir

    print("\n=== Murphy Brown Impact Analysis Complete ===")
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
    print("\n--- Generating Executive Summary ---")
    
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
    
    - **End-to-End Autonomy Rate:** {autonomy_metrics.get('autonomy_rate', 'N/A'):.1f}%
    - **Rework Rate:** {autonomy_metrics.get('rework_rate', 'N/A'):.1f}%
    - **Avg. Human Touches:** {autonomy_metrics.get('avg_human_touches', 'N/A'):.1f}
    
    ### Efficiency Improvements
    
    """
    
    # Add TAT comparison if available
    if time_efficiency and 'tat_stats' in time_efficiency:
        tat_stats = time_efficiency['tat_stats']
        if len(tat_stats) >= 2:
            mb_row = tat_stats[tat_stats['mb_touched'] == 1].iloc[0]
            non_mb_row = tat_stats[tat_stats['mb_touched'] == 0].iloc[0]
            tat_diff = mb_row['avg_tat_days'] - non_mb_row['avg_tat_days']
            
            report_content += f"""
    - **Turnaround Time Reduction:** {abs(tat_diff):.1f} days ({tat_diff/non_mb_row['avg_tat_days']*100:.1f}%)
    - **MB Avg. TAT:** {mb_row['avg_tat_days']:.1f} days vs. **Non-MB:** {non_mb_row['avg_tat_days']:.1f} days
    """
    
    # Add attempt reduction if available
    if contact_efficiency and 'attempts_stats' in contact_efficiency:
        attempt_stats = contact_efficiency['attempts_stats']
        if len(attempt_stats) >= 2:
            mb_row = attempt_stats[attempt_stats['mb_touched'] == 1].iloc[0]
            non_mb_row = attempt_stats[attempt_stats['mb_touched'] == 0].iloc[0]
            attempt_diff = mb_row['avg_attempts'] - non_mb_row['avg_attempts']
            
            report_content += f"""
    - **Attempt Reduction:** {abs(attempt_diff):.1f} attempts ({attempt_diff/non_mb_row['avg_attempts']*100:.1f}%)
    - **MB Avg. Attempts:** {mb_row['avg_attempts']:.1f} vs. **Non-MB:** {non_mb_row['avg_attempts']:.1f}
    """
    
    # Add time savings if available
    if time_savings:
        report_content += f"""
    ### Resource Impact
    
    - **Hours Saved Per Verification:** {time_savings.get('hours_saved_per_verification', 'N/A'):.2f} hours
    - **Total Hours Saved:** {time_savings.get('total_hours_saved', 'N/A'):.1f} hours
    - **FTE Equivalent:** {time_savings.get('fte_equivalent', 'N/A'):.2f} FTEs
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
    
    print(f"Executive summary saved to: {report_path}")
    return report_path


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
    
    print(f"\n=== Murphy Brown Impact Analysis Script ===")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir or 'Output/murphy_brown_analysis'}")
    
    start_time = datetime.now()
    
    try:
        # 1. Load data
        df_history = load_history_from_csv(args.input_file)

        # --- ADDED: Filter for 'empv' and 'eduv' search types (Moved Up) ---
        original_count = len(df_history)
        if 'searchtype' in df_history.columns:
            # Use .str.lower() for case-insensitivity and handle potential NAs safely
            # Filter for 'empv' OR 'eduv'
            target_types = ['empv', 'eduv'] # Updated to include both
            search_type_filter = df_history['searchtype'].astype(str).str.lower().isin(target_types)
            df_history = df_history.loc[search_type_filter].copy() # Use .loc and copy() to avoid SettingWithCopyWarning
            print(f"Filtered for {target_types} search types. Kept {len(df_history)} rows out of {original_count}.")
            if df_history.empty:
                print(f"No {target_types} records found after filtering. Exiting.")
                return # Exit if no matching records are left
        else:
            print("Warning: 'searchtype' column not found. Cannot filter by search type. Proceeding with all data.")
        # --- END ADDED FILTER ---

        # 2. Run comprehensive analysis
        # Ensure df_history is not empty before proceeding
        if not df_history.empty:
            analysis_results = perform_mb_impact_analysis(df_history)

            # 3. Generate executive summary
            if analysis_results:
                executive_summary = generate_executive_summary(analysis_results, args.output_dir)
                print(f"\nAnalysis Complete! Summary report available at: {executive_summary}")

                if 'dashboard_dir' in analysis_results:
                    print(f"Interactive dashboard available at: {analysis_results['dashboard_dir']}/dashboard.html")
        else:
            print("Skipping analysis as the filtered DataFrame is empty.")

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
    
    end_time = datetime.now()
    print(f"\nTotal execution time: {end_time - start_time}")


if __name__ == "__main__":
    main()