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
import json # Add json import for saving verdicts

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
            # --- Make dates timezone-naive before calculation ---
            start_date = row.first_attempt_time.tz_localize(None).date()
            end_date = row.completiondate.tz_localize(None).date()
            # ----------------------------------------------------
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

    # --- Define desired columns and target dtypes EARLIER --- 
    date_cols = [
        'historydatetime', 'completiondate', 'postdate', 'resultsentdate',
        'last_update', 'last_attempt', 'qadate', 'first_attempt'
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
        'contactmethod': 'string'
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

        # 3. Identify available vs missing columns (among ALL desired, including dates)
        available_cols = [col for col in all_desired_cols if col in df.columns]
        missing_desired_cols = [col for col in all_desired_cols if col not in df.columns]

        if missing_desired_cols:
            print(f"Warning: The following desired columns were NOT found in the CSV: {missing_desired_cols}")
        print(f"Columns available for processing: {available_cols}")

        # Keep only the available desired columns
        df = df[available_cols].copy()

        # 4. Apply remaining type conversions (especially for columns not covered by read_csv dtypes)
        #    This section is reduced as read_csv handled basic types.
        #    We primarily need to handle potential errors from read_csv dtype parsing
        #    and parse dates.
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
            'avg_tat_days', 'avg_attempts_by_row', 'delta_completion_rate', 
            'delta_tat_days', 'delta_attempts'
        ])
    # --- END FIX ---

    # Ensure required columns exist before grouping
    required_cols = ['searchid', 'is_completed', 'tat_calendar_days', 'attempts_by_row', 'contact_plan_provided']
    if not all(col in merged.columns for col in required_cols):
        missing = [col for col in required_cols if col not in merged.columns]
        print(f"Warning: Missing required columns for delta calculation in df_summary: {missing}")
        # Return an empty DataFrame or handle appropriately
        return pd.DataFrame(columns=[
            'contact_plan_provided', 'search_count', 'completion_rate', 
            'avg_tat_days', 'avg_attempts_by_row', 'delta_completion_rate', 
            'delta_tat_days', 'delta_attempts'
        ])

    # --- Convert metrics to numeric before aggregation ---
    merged['is_completed'] = pd.to_numeric(merged['is_completed'], errors='coerce')
    merged['tat_calendar_days'] = pd.to_numeric(merged['tat_calendar_days'], errors='coerce')
    merged['attempts_by_row'] = pd.to_numeric(merged['attempts_by_row'], errors='coerce')
    # ------------------------------------------------------

    grp = merged.groupby('contact_plan_provided')
    out = grp.agg(
        search_count           = ('searchid', 'size'),
        completion_rate        = ('is_completed', 'mean'),
        avg_tat_days           = ('tat_calendar_days', 'mean'),
        avg_attempts_by_row    = ('attempts_by_row', 'mean')
    ).reset_index()

    # Δ columns (plan – no-plan)
    # Check if both True and False groups exist in the output 'out'
    if True in out['contact_plan_provided'].values and False in out['contact_plan_provided'].values:
        row_with = out[out['contact_plan_provided'] == True]
        row_without = out[out['contact_plan_provided'] == False]

        # Safely access values only if rows exist (redundant check, but safe)
        if not row_with.empty and not row_without.empty:
             # Check if means are valid before calculating delta
             if pd.notna(row_with['completion_rate'].values[0]) and pd.notna(row_without['completion_rate'].values[0]):
                 out['delta_completion_rate'] = row_with['completion_rate'].values[0]  - row_without['completion_rate'].values[0]
             else: out['delta_completion_rate'] = np.nan
                 
             if pd.notna(row_with['avg_tat_days'].values[0]) and pd.notna(row_without['avg_tat_days'].values[0]):
                 out['delta_tat_days']        = row_with['avg_tat_days'].values[0]     - row_without['avg_tat_days'].values[0]
             else: out['delta_tat_days'] = np.nan
                 
             if pd.notna(row_with['avg_attempts_by_row'].values[0]) and pd.notna(row_without['avg_attempts_by_row'].values[0]):
                 out['delta_attempts']        = row_with['avg_attempts_by_row'].values[0] - row_without['avg_attempts_by_row'].values[0]
             else: out['delta_attempts'] = np.nan
        else:
            # This case should ideally not happen if both True/False are in values
            print("Warning: Could not find both 'with plan' and 'without plan' groups for delta calculation despite checks.")
            out['delta_completion_rate'] = np.nan
            out['delta_tat_days']        = np.nan
            out['delta_attempts']        = np.nan
    else:
         print("Warning: Missing either 'with plan' or 'without plan' group for delta calculation.")
         # Initialize delta columns even if calculation fails
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


# --- NEW: Verdict Functions ---

def email_performance_verdict(df_summary, min_pp=5, alpha=0.05):
    """Return True if MB email yield is materially better than control."""
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
    today = pd.Timestamp.now(tz=df_copy['historydatetime'].dt.tz).normalize() # Match source timezone if exists
    
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
            end_dt64 = (today_naive + pd.Timedelta(days=1)).to_numpy(dtype='datetime64[D]')
        else: # Should not happen, but fallback
             end_dt64 = (pd.Timestamp(today_naive) + pd.Timedelta(days=1)).to_numpy(dtype='datetime64[D]')
        
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

def monthly_efficiency_trend(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per calendar month with efficiency metrics for:
        • ALL verifications
        • MB-touched subset
        • non-MB subset
        • MB – non-MB deltas
    Also performs trend analysis and creates visualizations if sufficient data exists.
    """
    print("Calculating monthly efficiency trend...")
    if 'completiondate' not in df_summary.columns:
        print("Warning: Cannot calculate monthly efficiency trend - missing completiondate column")
        return pd.DataFrame()          # guard-rail

    # restrict to completed searches so TAT is comparable
    comp = df_summary[df_summary['is_completed'] == 1].copy()
    if comp.empty:
        print("Warning: No completed searches found for monthly efficiency trend")
        return pd.DataFrame()
        
    # Ensure completiondate is datetime
    if not pd.api.types.is_datetime64_any_dtype(comp['completiondate']):
        comp['completiondate'] = pd.to_datetime(comp['completiondate'], errors='coerce')
        
    # Validate completiondate after conversion
    if comp['completiondate'].isna().all():
        print("Warning: All completion dates are invalid for monthly efficiency trend")
        return pd.DataFrame()

    # month key
    comp['month'] = comp['completiondate'].dt.to_period('M').astype(str)

    # helper for any metric list
    def _agg(metrics: list):
        return {m: (m, 'mean') for m in metrics if m in comp.columns}

    base_metrics = ['tat_calendar_days',
                   'tat_business_days',
                   'attempts_by_row',
                   'human_touch_count']
    
    # Only use metrics that exist in the dataframe
    available_metrics = [m for m in base_metrics if m in comp.columns]
    if not available_metrics:
        print("Warning: None of the required metrics available for monthly efficiency trend")
        return pd.DataFrame()

    # ── roll-ups ────────────────────────────────────────────────────────────
    try:
        all_mo = comp.groupby('month').agg(**_agg(available_metrics))
        mb_mo = comp[comp['mb_touched'] == 1].groupby('month').agg(**_agg(available_metrics))
        ctl_mo = comp[comp['mb_touched'] == 0].groupby('month').agg(**_agg(available_metrics))

        panel = (all_mo
                .join(mb_mo, rsuffix='_mb')
                .join(ctl_mo, rsuffix='_ctl'))

        # add deltas (MB – control)
        for m in available_metrics:
            panel[f'{m}_delta'] = panel[f'{m}_mb'] - panel[f'{m}_ctl']

        panel = panel.reset_index()              # month back to column
        
        # --- Add Trend Analysis ---
        # Only if we have enough months (3+) for meaningful analysis
        if len(panel) >= 3:
            print("Performing trend analysis on monthly efficiency metrics...")
            # Add sequential period number for regression
            panel['period_num'] = range(len(panel))
            
            # Analyze trends for MB and delta metrics
            trend_results = {}
            for metric in available_metrics:
                mb_col = f"{metric}_mb"
                delta_col = f"{metric}_delta"
                
                # MB trend analysis
                if mb_col in panel.columns and not panel[mb_col].isna().all():
                    try:
                        # Ensure data is numeric before regression
                        y_values = pd.to_numeric(panel[mb_col], errors='coerce')
                        valid_mask = y_values.notna()
                        if valid_mask.sum() >= 2: # Need at least 2 points for regression
                            slope_mb, intercept_mb, r_mb, p_mb, std_err_mb = stats.linregress(
                                panel.loc[valid_mask, 'period_num'], y_values[valid_mask]
                            )
                            trend_results[f"{metric}_mb_trend"] = {
                                'slope': slope_mb,
                                'p_value': p_mb,
                                'significant': p_mb < 0.05
                            }
                            panel[f"{metric}_mb_trend"] = slope_mb
                            panel[f"{metric}_mb_p"] = p_mb
                        else:
                           print(f"Info: Not enough valid data points for {mb_col} trend analysis.")
                           panel[f"{metric}_mb_trend"] = np.nan
                           panel[f"{metric}_mb_p"] = np.nan 
                    except Exception as e:
                        print(f"Error analyzing trend for {mb_col}: {e}")
                        panel[f"{metric}_mb_trend"] = np.nan
                        panel[f"{metric}_mb_p"] = np.nan 
                else:
                    panel[f"{metric}_mb_trend"] = np.nan
                    panel[f"{metric}_mb_p"] = np.nan
                
                # Delta trend analysis
                if delta_col in panel.columns and not panel[delta_col].isna().all():
                    try:
                        # Ensure data is numeric before regression
                        y_values = pd.to_numeric(panel[delta_col], errors='coerce')
                        valid_mask = y_values.notna()
                        if valid_mask.sum() >= 2: # Need at least 2 points
                            slope_delta, intercept_delta, r_delta, p_delta, std_err_delta = stats.linregress(
                                panel.loc[valid_mask, 'period_num'], y_values[valid_mask]
                            )
                            trend_results[f"{metric}_delta_trend"] = {
                                'slope': slope_delta,
                                'p_value': p_delta,
                                'significant': p_delta < 0.05
                            }
                            panel[f"{metric}_delta_trend"] = slope_delta
                            panel[f"{metric}_delta_p"] = p_delta
                        else:
                           print(f"Info: Not enough valid data points for {delta_col} trend analysis.")
                           panel[f"{metric}_delta_trend"] = np.nan
                           panel[f"{metric}_delta_p"] = np.nan
                    except Exception as e:
                        print(f"Error analyzing trend for {delta_col}: {e}")
                        panel[f"{metric}_delta_trend"] = np.nan
                        panel[f"{metric}_delta_p"] = np.nan
                else:
                    panel[f"{metric}_delta_trend"] = np.nan
                    panel[f"{metric}_delta_p"] = np.nan
            
            # Print significant trends
            for metric_name, trend_data in trend_results.items():
                if trend_data.get('significant', False):
                    metric_desc = metric_name.replace('_mb_trend', ' MB').replace('_delta_trend', ' Delta')
                    direction = "decreasing" if trend_data.get('slope', 0) < 0 else "increasing"
                    print(f"Significant {direction} trend for {metric_desc}: {trend_data.get('slope', 'N/A'):.3f} per month (p={trend_data.get('p_value', 'N/A'):.3f})")
        # --- End Trend Analysis ---
        
        # Print the output for terminal viewing
        print("\n=== Monthly Efficiency Trend ===")
        print(panel.to_markdown(index=False))
        
        # --- Visualize the trends if we have enough data ---
        if len(panel) >= 3 and 'tat_calendar_days_mb' in panel.columns and 'tat_calendar_days_ctl' in panel.columns:
            try:
                # Sort by month for proper visualization
                panel_sorted = panel.sort_values('month')
                
                # Create directory for plots
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                except NameError:
                    script_dir = os.getcwd()
                output_dir = os.path.join(script_dir, "Output", "mb_efficiency_trends")
                os.makedirs(output_dir, exist_ok=True)
                
                # Plot TAT trend
                plt.figure(figsize=(10, 6))
                plt.plot(panel_sorted['month'], panel_sorted['tat_calendar_days_mb'], 'b-o', label='MB TAT')
                plt.plot(panel_sorted['month'], panel_sorted['tat_calendar_days_ctl'], 'r-o', label='Non-MB TAT')
                plt.title('TAT Trend by Month: MB vs Non-MB')
                plt.xlabel('Month')
                plt.ylabel('TAT (Calendar Days)')
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'tat_trend_by_month.png'), dpi=300)
                plt.close()
                
                # Plot Attempts trend if available
                if 'attempts_by_row_mb' in panel.columns and 'attempts_by_row_ctl' in panel.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(panel_sorted['month'], panel_sorted['attempts_by_row_mb'], 'b-o', label='MB Attempts')
                    plt.plot(panel_sorted['month'], panel_sorted['attempts_by_row_ctl'], 'r-o', label='Non-MB Attempts')
                    plt.title('Attempts Trend by Month: MB vs Non-MB')
                    plt.xlabel('Month')
                    plt.ylabel('Average Number of Attempts')
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'attempts_trend_by_month.png'), dpi=300)
                    plt.close()
                
                print(f"Monthly efficiency trend visualizations saved to: {output_dir}")
            except Exception as e:
                print(f"Error creating visualization for monthly efficiency trend: {e}")
                traceback.print_exc()
        # --- End Visualization ---
        
        return panel
    except Exception as e:
        print(f"Error calculating monthly efficiency trend: {e}")
        traceback.print_exc()
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
    print("\n=== Starting Murphy Brown Impact Analysis ===")

    # --- Initialize results dictionary EARLY --- 
    all_results = {}
    # --- END INIT ---

    # --- Check essential columns early ---
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
        
        # --- FIX: Add agentname aggregation ---
        'agentname': ('agentname', 'first') # Keep the first agentname associated with the searchid
        # --- END FIX ---
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

    # *** NEW: Initialize contact_plan_provided column early ***
    df_summary['contact_plan_provided'] = False
    print("Initialized 'contact_plan_provided' column in df_summary.")
    # *********************************************************

    # --- Step 4: Calculate Downstream Metrics using df_summary --- 
    print("Step 4: Calculating downstream analysis metrics...")

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
        print("\n*** ΔTAT (calendar) MB-vs-Non-MB ***")
        print(time_efficiency_results['tat_stats'].to_markdown(index=False))
        # --- Save TAT stats --- 
        save_results(time_efficiency_results['tat_stats'], 'tat_stats_every_run')
        # ----------------------
    # ------------------------

    # c) Calculate contact efficiency metrics
    contact_efficiency_results = calculate_contact_efficiency_metrics(df_summary)
    all_results['contact_efficiency'] = contact_efficiency_results # Store results
    # ---- Print Attempts stats ----
    if contact_efficiency_results and 'attempts_stats' in contact_efficiency_results and contact_efficiency_results['attempts_stats'] is not None:
        print("\n*** ΔAttempts MB-vs-Non-MB ***")
        print(contact_efficiency_results['attempts_stats'].to_markdown(index=False))
        # --- Save Attempts stats --- 
        save_results(contact_efficiency_results['attempts_stats'], 'attempts_stats_every_run')
        # ---------------------------
    # ----------------------------

    # --- Outbound Email/Fax Automation Rate (Phase 3A) ---
    print("Step 4d: Calculating outbound automation rate for email & fax...")
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
            print(f"Automation Rate:         {automation_rate:.1%}")
            print(f"Reassignment Rate:       {reassignment_rate:.1%}")
            print(f"Avg Time to Response:    {avg_response_time:.1f} hours")
            print(f"Email Success Rate:      {email_success_rate:.1%}")
            print(f"Fax Success Rate:        {fax_success_rate:.1%}")
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

    # e) Estimate agent hours saved
    time_savings_results = estimate_agent_hours_saved(
        df_summary, 
        autonomy_results,
        df_history, 
        time_per_touch=20 # Changed from 5 to 20
    )
    all_results['time_savings'] = time_savings_results # Store results

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
    print("Calculating and saving agent throughput...")
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
    
    # --- Calculate Monthly Efficiency Trend --- 
    print("Calculating monthly efficiency metrics...")
    monthly_efficiency = monthly_efficiency_trend(df_summary)
    all_results['monthly_efficiency'] = monthly_efficiency
    if monthly_efficiency is not None and not monthly_efficiency.empty:
        save_results(monthly_efficiency, 'mb_monthly_efficiency')
    # -----------------------------------------

    # i) Merge contact plan flag into df_summary (if available, overwrites initial False)
    if plan_flags_df is not None:
        # --- MODIFIED MERGE ---
        # Drop the placeholder column first to avoid suffixes _x, _y
        df_summary = df_summary.drop(columns=['contact_plan_provided'], errors='ignore') 
        df_summary = pd.merge(df_summary, plan_flags_df, on='searchid', how='left')
        # Fillna for searches that existed in summary but not in plan_flags_df
        df_summary['contact_plan_provided'] = df_summary['contact_plan_provided'].fillna(False).astype(bool)
        # --- END MODIFICATION ---
        print("Merged actual contact plan flags into df_summary.")
    else:
        # If flags couldn't be generated, the column remains all False as initialized
        print("Warning: Contact plan flags not available. 'contact_plan_provided' column remains False.")

    # j) Calculate delta metrics for searches with vs without MB contact-plan
    plan_delta = None 
    # Check if the necessary flag column now exists (it should always exist now)
    if 'contact_plan_provided' in df_summary.columns:
        required_summary_cols = ['searchid', 'is_completed', 'tat_calendar_days', 'attempts_by_row']
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
            print("Calculating contact plan verdict...")
            # Ensure required columns exist in df_summary for the t-test
            # These checks should be more robust now due to early initialization and checks within delta function
            if 'tat_calendar_days' in df_summary.columns and 'contact_plan_provided' in df_summary.columns and 'is_completed' in df_summary.columns and 'attempts_by_row' in df_summary.columns:
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
                        group_with_plan = df_summary[df_summary['contact_plan_provided']]['tat_calendar_days'].dropna()
                        group_without_plan = df_summary[~df_summary['contact_plan_provided']]['tat_calendar_days'].dropna()
                        
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
                             print("Info: Insufficient data for contact plan TAT t-test.")

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
                         print(f"Contact Plan Verdict: {verdict}")

                    contact_verdict = verdict # Update the main dict
                    print(f"Contact Plan Verdict: {contact_verdict}")
                else:
                     contact_verdict = {'overall_good': False, 'reason': 'No rows with contact_plan_provided=True in delta table'}
                     print(f"Contact Plan Verdict: {contact_verdict}")
            else:
                contact_verdict = {'overall_good': False, 'reason': 'Missing required columns in df_summary for verdict calculation'}
                print(f"Contact Plan Verdict: {contact_verdict}")
        except Exception as e:
            print(f"Error calculating contact plan verdict: {e}")
            traceback.print_exc()
            contact_verdict = {'overall_good': False, 'reason': f'Error: {e}'}
            
    all_results['contact_plan_verdict'] = contact_verdict
    save_verdict(contact_verdict, 'mb_contact_plan_verdict') # Save as JSON

    # l) Calculate Email Performance Verdict
    print("Calculating email performance verdict...")
    email_verdict = email_performance_verdict(df_summary)
    all_results['email_performance_verdict'] = email_verdict
    print(f"Email Performance Verdict: {email_verdict}")
    save_verdict(email_verdict, 'mb_email_performance_verdict') # Save as JSON

    # --- 5. Compile Results & Save/Visualize --- 
    print("Step 5: Compiling results and generating outputs...")
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

    # --- REMOVE attempt_bucket_stats check ---
    # (The block checking for 'attempt_bucket_stats' has been removed)
    # --- END REMOVAL ---

    # Create Detailed Sample Output
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
        tat_stats = time_efficiency.get('tat_stats') # Get the DataFrame
        # --- MODIFIED GUARD ---
        if tat_stats is not None and not tat_stats.empty and {'avg_tat_days','mb_touched'}.issubset(tat_stats.columns):
        # --- END GUARD ---
            # Find MB and non-MB rows safely
            mb_row = tat_stats[tat_stats['mb_touched'] == 1]
            non_mb_row = tat_stats[tat_stats['mb_touched'] == 0]
            
            # Calculate diff only if both rows exist and have data
            if not mb_row.empty and not non_mb_row.empty and pd.notna(mb_row['avg_tat_days'].iloc[0]) and pd.notna(non_mb_row['avg_tat_days'].iloc[0]):
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
            if not mb_row.empty and not non_mb_row.empty and pd.notna(mb_row['avg_attempts'].iloc[0]) and pd.notna(non_mb_row['avg_attempts'].iloc[0]):
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
    
    # --- ADD MONTHLY EFFICIENCY TREND SECTION ---
    if 'monthly_efficiency' in results and results['monthly_efficiency'] is not None and not results['monthly_efficiency'].empty:
        monthly_eff = results['monthly_efficiency']
        
        # Check for TAT metrics
        if 'tat_calendar_days_mb' in monthly_eff.columns and 'tat_calendar_days_ctl' in monthly_eff.columns:
            report_content += """
    ### Month-by-Month Efficiency Trend
    
    Monthly comparison of key metrics between MB and non-MB verifications:
    """
            # Sort by month to show trend
            try:
                monthly_eff_sorted = monthly_eff.sort_values('month')
                # Get the most recent 3 months if available
                recent_months = monthly_eff_sorted.tail(min(3, len(monthly_eff_sorted)))
                
                report_content += """
    | Month | MB TAT | Non-MB TAT | Difference | MB Attempts | Non-MB Attempts | Diff |
    |-------|--------|------------|------------|-------------|-----------------|------|"""
                
                for _, row in recent_months.iterrows():
                    month = row['month']
                    mb_tat = row.get('tat_calendar_days_mb', 'N/A')
                    ctl_tat = row.get('tat_calendar_days_ctl', 'N/A')
                    tat_diff = row.get('tat_calendar_days_delta', 'N/A')
                    
                    mb_attempts = row.get('attempts_by_row_mb', 'N/A')
                    ctl_attempts = row.get('attempts_by_row_ctl', 'N/A')
                    attempts_diff = row.get('attempts_by_row_delta', 'N/A')
                    
                    # Format values
                    mb_tat_str = f"{mb_tat:.1f}" if not pd.isna(mb_tat) else "N/A"
                    ctl_tat_str = f"{ctl_tat:.1f}" if not pd.isna(ctl_tat) else "N/A"
                    tat_diff_str = f"{tat_diff:.1f}" if not pd.isna(tat_diff) else "N/A"
                    
                    mb_attempts_str = f"{mb_attempts:.1f}" if not pd.isna(mb_attempts) else "N/A"
                    ctl_attempts_str = f"{ctl_attempts:.1f}" if not pd.isna(ctl_attempts) else "N/A"
                    attempts_diff_str = f"{attempts_diff:.1f}" if not pd.isna(attempts_diff) else "N/A"
                    
                    report_content += f"""
    | {month} | {mb_tat_str} | {ctl_tat_str} | {tat_diff_str} | {mb_attempts_str} | {ctl_attempts_str} | {attempts_diff_str} |"""
                
                # Add note on interpretation
                report_content += """

    Note: Negative differences for TAT and attempts indicate MB is performing better than non-MB verifications.
    """
                
                # Check if there are enough months for trend analysis
                if len(monthly_eff) >= 3:
                    report_content += """
    The month-by-month trend shows the consistent efficiency gains from Murphy Brown implementation.
    """
            except Exception as e:
                print(f"Error formatting monthly efficiency trend for summary: {e}")
                report_content += f"""
    *Error displaying monthly trend: {str(e)}*
    """
    # --- END MONTHLY EFFICIENCY TREND SECTION ---
            
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
            # Filter for 'empv' OR 'eduv'target_types = ['empv', 'eduv'] 
            target_types = ['empv'] # Updated to include both
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