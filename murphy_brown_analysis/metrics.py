#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py - Core metric calculation functions for Murphy Brown impact analysis

Contains all functions for calculating basic metrics from history data:
- TAT (turnaround time) calculations
- Attempt counts
- MB capability detection (contact research, email handling)
- Autonomy metrics
- Contact plan detection
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import re
from datetime import datetime
import os, re, warnings

# --- Configuration & Constants ---
MB_AGENT_IDENTIFIER = "murphy.brown"  # Agent identifier (case-insensitive)

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
CONTACT_PLAN_PHRASE = "contacts found"  # Phrase indicating a contact plan note
CONTACT_KEYWORDS = ['phone:', 'email:', 'fax:']  # Keywords to count as contacts


def standardize_column_names(df):
    """Ensure consistent column name access by standardizing case and handling variations."""
    # Convert to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    
    # Map common variations to standard names
    column_map = {
        'agent': 'agentname',
        'agent_name': 'agentname',
        'agnt': 'agentname',
        'agents': 'agentname',
        'datetime': 'historydatetime',
        'history_datetime': 'historydatetime',
        'histdatetime': 'historydatetime',
        'history_dt': 'historydatetime',
        'user': 'userid',
        'user_name': 'userid',
        'username': 'userid'
    }
    
    # Apply mapping for columns that exist
    renamed = False
    for old_col, new_col in column_map.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
            renamed = True
            
    # If two original columns now share the same new name (e.g., "userid"),
    # keep the first and drop the duplicates so downstream .str operations
    # receive a Series, not a DataFrame.
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique()
        print(f"Removing duplicated columns created during rename: {list(dupes)}")
        df = df.loc[:, ~df.columns.duplicated()]

    if renamed:
        print("Column names standardized for consistent access")
        
    return df


def prepare_history_df(df):
    """
    Prepare the history DataFrame for analysis by ensuring correct data types
    and adding basic flag columns.
    
    Args:
        df: History DataFrame
        
    Returns:
        Prepared DataFrame with additional flag columns
    """
    print("Preparing history DataFrame for analysis...")
    
    df = df.copy()
    
    # Add column standardization first
    df = standardize_column_names(df)
    
    # Ensure basic columns have correct types
    df['note'] = df['note'].fillna('').astype(str)
    df['agentname'] = df['agentname'].fillna('').astype(str)
    df['resultid'] = pd.to_numeric(df['resultid'], errors='coerce').astype('Int16')
    df['contactmethod'] = df['contactmethod'].fillna('').astype(str).str.lower()
    df['searchstatus'] = df['searchstatus'].fillna('').astype(str).str.upper()
    
    # Ensure historydatetime is properly parsed
    if 'historydatetime' in df.columns:
        try:
            # Convert to string first to handle potential mixed types
            df['historydatetime'] = df['historydatetime'].astype(str)

            # Try multiple formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%Y-%m-%d']:
                try:
                    df['historydatetime'] = pd.to_datetime(df['historydatetime'], format=fmt, errors='coerce')
                    if not df['historydatetime'].isna().all():
                        print(f"Successfully parsed historydatetime using format: {fmt}")
                        break
                except:
                    continue
                    
            # Fallback to no format specified if still all NaNs
            if df['historydatetime'].isna().all():
                print("Attempting generic datetime parsing for historydatetime...")
                df['historydatetime'] = pd.to_datetime(df['historydatetime'], errors='coerce')
                print("Used generic datetime parsing for historydatetime")
                
            # Final check if parsing worked
            if pd.api.types.is_datetime64_any_dtype(df['historydatetime']):
                valid_dates = df['historydatetime'].notna().sum()
                print(f"Found {valid_dates} valid historydatetime values out of {len(df)}")
            else:
                print("Warning: historydatetime could not be converted to datetime objects.")

        except Exception as e:
            print(f"Error parsing historydatetime: {e}")
    elif 'historydatetime' not in df.columns:
        print("Warning: 'historydatetime' column not found.")

    # Add basic flags
    mb_pattern = MB_AGENT_IDENTIFIER.lower()
    
    # MB row flag
    df['userid_lower'] = df['userid'].fillna('').astype(str).str.lower()
    df['is_mb_row'] = (
        df['agentname'].str.lower().str.contains(mb_pattern, na=False) |
        df['userid_lower'].eq(mb_pattern)
    )
    
    # MB contact research flag
    df['is_mb_contact_research_note'] = df['note'].str.contains(MB_CONTACT_PATTERN, na=False)
    
    # MB email handling flag
    df['is_mb_email_handling_note'] = df['note'].str.contains(MB_EMAIL_PATTERN, na=False)
    
    # Applicant contact flag (Result ID 16)
    df['is_applicant_contact'] = (df['resultid'] == 16)
    
    # Email contact method flag
    df['is_email_contact_method'] = (df['contactmethod'] == 'email')
    
    # Completion status flag
    df['is_completion_status'] = (df['searchstatus'] == 'REVIEW')
    
    print("History DataFrame preparation complete.")
    return df


def calculate_tat(df_hist, holidays=HOLIDAYS, status_col="searchstatus", finished_status="REVIEW"):
    """
    Calculate turnaround time metrics for each search ID.
    
    Args:
        df_hist: History DataFrame
        holidays: Array of holiday dates to exclude in business day calculations
        status_col: Column name containing status
        finished_status: Status value indicating search completion
        
    Returns:
        DataFrame with TAT metrics by searchid
    """
    print("Calculating TAT metrics...")
    
    required = {"searchid", status_col, "historydatetime"}
    missing = required.difference(df_hist.columns)
    if missing:
        raise ValueError(f"History data is missing required columns: {missing}")
    
    # Ensure timestamp column is datetime
    df = df_hist.copy()
    df["historydatetime"] = pd.to_datetime(df["historydatetime"], errors="coerce")
    
    # Group by searchid
    grp = df.groupby("searchid", as_index=False)
    
    # Calculate core timestamps
    core = (
        grp["historydatetime"].agg(first_attempt_time="min", last_attempt_time="max")
        .merge(
            # completiondate = latest hist-datetime where status == finished_status
            df[df[status_col].astype(str).str.upper() == finished_status]
              .groupby("searchid", as_index=False)["historydatetime"]
              .max()
              .rename(columns={"historydatetime": "completiondate"}),
            how="left", on="searchid"
        )
    )
    
    # Fallback: if nothing marked as finished_status, use last attempt
    mask_na = core["completiondate"].isna()
    core.loc[mask_na, "completiondate"] = core.loc[mask_na, "last_attempt_time"]
    
    # Completion flag
    core["is_completed"] = core["completiondate"].notna().astype(int)
    
    # Calendar TAT (inclusive)
    core["tat_calendar_days"] = np.where(
        core["is_completed"] == 1,
        (core["completiondate"].dt.normalize() 
         - core["first_attempt_time"].dt.normalize()).dt.days + 1,
        np.nan
    )
    core.loc[core["tat_calendar_days"] < 0, "tat_calendar_days"] = np.nan
    core["tat_calendar_days"] = core["tat_calendar_days"].astype('Float64')
    
    # Business-day TAT (inclusive)
    def _busdays(row):
        if row.is_completed != 1 or pd.isna(row.first_attempt_time) or pd.isna(row.completiondate):
            return np.nan
        
        try:
            # Make dates timezone-naive if they are timezone-aware
            start_time = row.first_attempt_time
            end_time = row.completiondate

            if start_time.tz is not None:
                start_time = start_time.tz_localize(None)
            if end_time.tz is not None:
                end_time = end_time.tz_localize(None)

            start_date = start_time.date()
            end_date = end_time.date()
            
            # Cast to np.datetime64 and add day for inclusivity
            start_dt64 = np.datetime64(start_date, 'D')
            end_dt64 = np.datetime64(end_date, 'D') + np.timedelta64(1, 'D')
            
            return np.busday_count(start_dt64, end_dt64, holidays=holidays)
        except Exception:
            return np.nan

    core["tat_business_days"] = core.apply(_busdays, axis=1).astype("Int64")
    
    print("TAT calculation complete.")
    return core[
        ["searchid", "first_attempt_time", "completiondate", "is_completed",
         "tat_calendar_days", "tat_business_days"]
    ]


def calculate_attempt_counts(df):
    """
    Calculate attempt count metrics per search ID.
    
    Args:
        df: History DataFrame
    
    Returns:
        DataFrame with attempt metrics by searchid
    """
    print("Calculating attempt count metrics...")
    
    required_cols = ['searchid', 'historyid', 'resultid', 'is_applicant_contact']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for attempt count metrics: {missing}")
    
    # Ensure required columns have correct types
    df = df.copy()
    df['resultid'] = pd.to_numeric(df['resultid'], errors='coerce')
    
    # Group by searchid
    agg_dict = {
        'attempts_by_row': ('historyid', 'size'),
        'distinct_applicant_contact_count': ('is_applicant_contact', 'sum')
    }
    
    attempts_df = df.groupby('searchid').agg(**agg_dict).reset_index()
    
    # Calculate maxminusapplicant
    attempts_df['maxminusapplicant'] = (
        attempts_df['attempts_by_row'] - attempts_df['distinct_applicant_contact_count']
    ).clip(lower=0).astype('Int64')
    
    print("Attempt count metrics calculation complete.")
    return attempts_df


def extract_contacts_from_notes(df):
    """
    Extract contact details from MB contact research notes.
    
    Args:
        df: History DataFrame with is_mb_contact_research_note flag
        
    Returns:
        DataFrame with contact counts by searchid
    """
    print("Extracting contact details from notes...")
    
    if 'is_mb_contact_research_note' not in df.columns or 'note' not in df.columns:
        print("Missing required columns for contact extraction.")
        return pd.DataFrame(columns=['searchid', 'email_count', 'phone_count', 'fax_count', 'total_contacts'])
    
    # Filter to MB contact research notes
    mask_contact_notes = df['is_mb_contact_research_note']
    if not mask_contact_notes.any():
        print("No MB contact research notes found.")
        return pd.DataFrame(columns=['searchid', 'email_count', 'phone_count', 'fax_count', 'total_contacts'])
    
    # Use extractall for all contact types in one pass
    contact_extract_pattern = r'(?i)(?:email:\s*([^\s@]+@[^\s@]+))|(?:phone:\s*([+\d\-\(\)]+))|(?:fax:\s*([+\d\-\(\)]+))'
    contacts_extracted = df.loc[mask_contact_notes, ['searchid', 'note']].copy()
    contacts_extracted = contacts_extracted.set_index('searchid')
    
    # Extract all matches and get counts
    all_contacts = contacts_extracted['note'].str.extractall(contact_extract_pattern)
    if all_contacts.empty:
        print("No contacts found in notes.")
        return pd.DataFrame(columns=['searchid', 'email_count', 'phone_count', 'fax_count', 'total_contacts'])
    
    # Rename columns for clarity
    all_contacts.columns = ['email_match', 'phone_match', 'fax_match']
    
    # Count non-NA matches per searchid
    contact_counts = all_contacts.notna().groupby(level=0).sum()
    contact_counts = contact_counts.rename(columns={
        'email_match': 'email_count',
        'phone_match': 'phone_count',
        'fax_match': 'fax_count'
    })
    
    # Add total column
    contact_counts['total_contacts'] = contact_counts.sum(axis=1)
    
    # Reset index to get searchid as column
    contact_counts = contact_counts.reset_index()
    
    # Fill NAs with 0 and convert to integers
    contact_count_cols = ['email_count', 'phone_count', 'fax_count', 'total_contacts']
    contact_counts[contact_count_cols] = contact_counts[contact_count_cols].fillna(0).astype('Int64')
    
    print("Contact extraction complete.")
    return contact_counts


def calculate_mb_capability_flags(df):
    """
    Calculate MB capability flags per search ID.
    
    Args:
        df: History DataFrame with MB flags
        
    Returns:
        DataFrame with MB capability flags by searchid
    """
    print("Calculating MB capability flags...")
    
    required_cols = ['searchid', 'is_mb_row', 'is_mb_contact_research_note', 'is_mb_email_handling_note']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for MB capability flags: {missing}")
    
    # Group by searchid and take max of each flag (True > False)
    mb_flags = df.groupby('searchid').agg(
        mb_touched=('is_mb_row', 'max'),
        mb_contact_research=('is_mb_contact_research_note', 'max'),
        mb_email_handling=('is_mb_email_handling_note', 'max')
    ).reset_index()
    
    # Convert boolean flags to int (1/0)
    flag_cols = ['mb_touched', 'mb_contact_research', 'mb_email_handling']
    for col in flag_cols:
        mb_flags[col] = mb_flags[col].astype(int)
    
    print("MB capability flags calculation complete.")
    return mb_flags


def calculate_email_metrics(df):
    """
    Calculate email interaction metrics per search ID.
    
    Args:
        df: History DataFrame with email flags
        
    Returns:
        DataFrame with email metrics by searchid
    """
    print("Calculating email metrics...")
    
    required_cols = ['searchid', 'is_email_contact_method', 'note']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for email metrics: {missing}")
    
    # Count email interactions in notes
    df = df.copy()
    df['email_interaction_instances'] = df['note'].str.count(MB_EMAIL_PATTERN)
    
    # Group by searchid
    email_metrics = df.groupby('searchid').agg(
        email_interaction_count=('email_interaction_instances', 'sum'),
        email_method_count=('is_email_contact_method', 'sum')
    ).reset_index()
    
    # Convert to nullable integers
    count_cols = ['email_interaction_count', 'email_method_count']
    email_metrics[count_cols] = email_metrics[count_cols].fillna(0).astype('Int64')
    
    print("Email metrics calculation complete.")
    return email_metrics


def calculate_autonomy_metrics(df):
    """
    Calculate autonomy-related metrics for Murphy Brown.
    
    Args:
        df: History DataFrame with is_mb_row flag
        
    Returns:
        DataFrame with autonomy metrics by searchid
    """
    print("Calculating MB autonomy metrics...")
    
    required_cols = ['searchid', 'historydatetime', 'is_mb_row']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for autonomy metrics: {missing}")
    
    # Ensure historydatetime is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['historydatetime']):
        df['historydatetime'] = pd.to_datetime(df['historydatetime'], errors='coerce')
    
    # Human touch is the inverse of MB row
    df['userid_lower'] = df['userid'].fillna('').astype(str).str.lower()
    mb_pattern = MB_AGENT_IDENTIFIER.lower()
    df['is_mb_row'] = (
        df['agentname'].str.lower().str.contains(mb_pattern, na=False) |
        df['userid_lower'].eq(mb_pattern)
    )
    df['is_human_touch'] = ~df['is_mb_row']
    
    # Sort by searchid and timestamp
    df_sorted = df.sort_values(['searchid', 'historydatetime']).copy()
    
    # Calculate human touch count per searchid
    human_touches = (
        df_sorted[df_sorted['is_human_touch']]
        .groupby('searchid')
        .size()
        .rename('human_touch_count')
    )
    
    # Identify Rework: Human touch immediately after an MB touch
    df_sorted['prev_is_human'] = df_sorted.groupby('searchid')['is_human_touch'].shift(1)
    rework_mask = df_sorted['is_human_touch'] & (df_sorted['prev_is_human'] == False)
    rework_search_ids = df_sorted.loc[rework_mask, 'searchid'].unique()
    
    # Identify Fallback: Last touch in a search is human
    last_touches = df_sorted.groupby('searchid').tail(1)
    fallback_search_ids = last_touches.loc[last_touches['is_human_touch'], 'searchid'].unique()
    
    # Create autonomy DataFrame
    all_search_ids = df['searchid'].unique()
    autonomy_df = pd.DataFrame({'searchid': all_search_ids})
    
    # Add human touch counts
    autonomy_df = pd.merge(autonomy_df, human_touches.reset_index(), on='searchid', how='left')
    autonomy_df['human_touch_count'] = autonomy_df['human_touch_count'].fillna(0).astype('Int64')
    
    # Add autonomy flags
    autonomy_df['fully_autonomous'] = (autonomy_df['human_touch_count'] == 0).astype(int)
    autonomy_df['has_rework'] = autonomy_df['searchid'].isin(rework_search_ids).astype(int)
    autonomy_df['had_fallback'] = autonomy_df['searchid'].isin(fallback_search_ids).astype(int)
    
    print("Autonomy metrics calculation complete.")
    return autonomy_df


def detect_contact_plans(df):
    """
    Identify searches where MB provided a contact plan and count contacts.
    
    Args:
        df: History DataFrame
        
    Returns:
        DataFrame with contact plan flags and counts by searchid
    """
    print("Detecting MB contact plans...")
    
    required_cols = ['searchid', 'userid', 'note', 'historydatetime']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for contact plan detection: {missing}")
    
    # Prepare data
    df = df.copy()
    df['userid_lower'] = df['userid'].fillna('').astype(str).str.lower()
    df['note_clean'] = df['note'].fillna('').astype(str).str.lower()
    
    # Ensure historydatetime is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['historydatetime']):
        df['historydatetime'] = pd.to_datetime(df['historydatetime'], errors='coerce')
    
    # Identify MB contact plan rows
    mb_pattern = MB_AGENT_IDENTIFIER.lower()
    is_mb_user = (df['userid_lower'] == mb_pattern)
    has_contact_phrase = df['note_clean'].str.contains(CONTACT_PLAN_PHRASE, case=False, na=False)
    df['is_mb_contact_plan_row'] = is_mb_user & has_contact_phrase
    
    # Count contacts within plan notes
    def count_contacts_in_note(note_text):
        if pd.isna(note_text) or not note_text:
            return 0
        count = 0
        for pattern in CONTACT_TYPE_PATTERNS.values():
            count += len(re.findall(pattern, note_text, re.IGNORECASE))
        return count
    
    # Initialize contacts column
    df['contacts_in_plan'] = 0
    
    # Calculate only for rows identified as MB contact plans
    plan_rows_mask = df['is_mb_contact_plan_row']
    df.loc[plan_rows_mask, 'contacts_in_plan'] = df.loc[plan_rows_mask, 'note_clean'].apply(count_contacts_in_note)
    
    # Aggregate by searchid
    search_agg = df.groupby('searchid', as_index=False).agg(
        contact_plan_provided=('is_mb_contact_plan_row', 'max'),
        total_contacts_provided=('contacts_in_plan', 'sum')
    )
    
    # Ensure contacts are 0 if no plan was provided
    search_agg.loc[~search_agg['contact_plan_provided'], 'total_contacts_provided'] = 0
    
    # Convert boolean flag to int
    search_agg['contact_plan_provided'] = search_agg['contact_plan_provided'].astype(int)
    
    print("Contact plan detection complete.")
    return search_agg


def calculate_phase3a_metrics(df):
    """
    Calculate Phase 3A outbound automation metrics.
    
    Args:
        df: History DataFrame with proper flags
        
    Returns:
        Dictionary with Phase 3A metrics and DataFrame with search-level details
    """
    print("Calculating Phase 3A outbound automation metrics...")
    
    required_cols = ['searchid', 'historydatetime', 'is_mb_row', 'contactmethod', 'searchstatus']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for Phase 3A metrics: {missing}")
    
    # Initialize results
    phase_3a_metrics = {}
    
    try:
        # Filter MB outbound attempts over email or fax
        mask_out = (
            df['is_mb_row'] &
            df['contactmethod'].isin(['email', 'fax'])
        )
        out = df[mask_out].sort_values(['searchid', 'historydatetime']).copy()
        
        if out.empty:
            print("No MB email/fax attempts found. Skipping Phase 3A calculation.")
            return {
                'metrics': {
                    'automation_rate': 0,
                    'reassignment_rate': 0,
                    'avg_response_time_hours': np.nan,
                    'email_success_rate': np.nan,
                    'fax_success_rate': np.nan
                },
                'summary_df': pd.DataFrame(columns=[
                    'searchid', 'attempts', 'end_time', 'resp_time', 'auto_complete'
                ])
            }
        
        # Only first 48 hours per search, then take up to 2 attempts
        first_ts = out.groupby('searchid')['historydatetime'].first().rename('t0')
        out = out.join(first_ts, on='searchid')
        out = out.loc[out['historydatetime'] <= out['t0'] + pd.Timedelta(hours=48)]
        out = out.groupby('searchid').head(2)
        
        # Pull first "REVIEW" (success) time per search
        resp = (
            df[df['searchstatus'] == 'REVIEW']
            .groupby('searchid', as_index=False)['historydatetime']
            .min()
            .rename(columns={'historydatetime': 'resp_time'})
        )
        
        # Build summary and flag auto-completes
        summary_out = (
            out.groupby('searchid')
               .agg(attempts=('historydatetime', 'count'),
                    end_time=('historydatetime', 'max'))
               .reset_index()
               .merge(resp, on='searchid', how='left')
        )
        
        # Success if resp_time <= end_time of 2nd attempt
        summary_out['auto_complete'] = (
            summary_out['resp_time'] <= summary_out['end_time']
        ).fillna(False)
        
        # Calculate key metrics
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
            resp_times = summary_out.set_index('searchid')['resp_time']
            resp_times, first_times = resp_times.align(first_times, join='inner')
            
            if not resp_times.empty and not first_times.empty:
                response_hours = (resp_times - first_times).dt.total_seconds() / 3600
                avg_response_time = response_hours.mean()
            else:
                avg_response_time = np.nan
        else:
            avg_response_time = np.nan
            
        phase_3a_metrics['avg_response_time_hours'] = avg_response_time
        
        # 4. Channel Success Rate
        email_ids = out.loc[out['contactmethod'] == 'email', 'searchid'].unique()
        fax_ids = out.loc[out['contactmethod'] == 'fax', 'searchid'].unique()
        
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
        
        print("Phase 3A metrics calculation complete.")
        
        # Ensure metrics dictionary has all expected keys
        default_metrics = {
            'automation_rate': 0.0,
            'reassignment_rate': 0.0,
            'avg_response_time_hours': 0.0,
            'email_success_rate': 0.0,
            'fax_success_rate': 0.0
        }

        for key, default_value in default_metrics.items():
            if key not in phase_3a_metrics:
                phase_3a_metrics[key] = default_value
        
        return {
            'metrics': phase_3a_metrics,
            'summary_df': summary_out
        }
        
    except Exception as e:
        print(f"Error during Phase 3A calculation: {e}")
        return {
            'metrics': {
                'automation_rate': 0,
                'reassignment_rate': 0,
                'avg_response_time_hours': np.nan,
                'email_success_rate': np.nan,
                'fax_success_rate': np.nan
            },
            'summary_df': pd.DataFrame(columns=[
                'searchid', 'attempts', 'end_time', 'resp_time', 'auto_complete'
            ])
        }


def queue_depth_weekly(df_hist: pd.DataFrame, sla_days: int = 5) -> pd.DataFrame:
    """
    Weekly % of OPEN searches (> `sla_days` business-days old).

    A search is considered **open** if its *last* history row has
    `searchstatus != 'REVIEW'`.
    """
    need = {'searchid', 'historydatetime', 'searchstatus'}
    if not need.issubset(df_hist.columns):
        warnings.warn("Missing required columns for queue_depth_weekly. Returning empty DataFrame.")
        return pd.DataFrame()

    tmp = df_hist[['searchid', 'historydatetime', 'searchstatus']].copy()
    tmp['historydatetime'] = pd.to_datetime(tmp['historydatetime'], errors='coerce')

    # Handle potential NaT values after coercion
    if tmp['historydatetime'].isnull().any():
        warnings.warn("Some 'historydatetime' values could not be parsed and were set to NaT.")
        tmp.dropna(subset=['historydatetime'], inplace=True)
        if tmp.empty:
            return pd.DataFrame()

    tmp['searchstatus'] = tmp['searchstatus'].fillna('').astype(str).str.upper() # Ensure fillna and str conversion

    # pick last row per search
    # Use .idx max() which handles NaT properly if any remained (though dropped above)
    last_indices = tmp.groupby('searchid')['historydatetime'].idxmax()
    last = tmp.loc[last_indices]

    open_rows = last[last['searchstatus'] != 'REVIEW'].copy()
    if open_rows.empty:
        print("No open rows found for queue depth calculation.")
        return pd.DataFrame(columns=['week', 'total_open', 'open_gt_sla', 'pct_open_gt_sla']) # Return empty with cols

    today = pd.Timestamp.utcnow().normalize()

    # vectorized business-day age
    start = open_rows['historydatetime'].dt.normalize().values.astype('datetime64[D]')
    # Ensure 'today' is also timezone-naive if start is, before converting to numpy
    if start.dtype == '<M8[ns]': # Check if timezone-naive
         today_for_calc = today.tz_localize(None) if today.tz is not None else today
    else: # If start has timezone, ensure today matches or handle appropriately
         # This case might need more specific handling based on expected timezones
         today_for_calc = today # Assuming UTC or compatible timezone handling
    end   = (today_for_calc + pd.Timedelta(days=1)).to_numpy(dtype='datetime64[D]')

    # Ensure HOLIDAYS is the numpy array expected by np.busday_count
    global HOLIDAYS
    if not isinstance(HOLIDAYS, np.ndarray) or HOLIDAYS.dtype != 'datetime64[D]':
         warnings.warn(f"Global HOLIDAYS is not a numpy array of dtype 'datetime64[D]'. Using empty list for calculation. Original type: {type(HOLIDAYS)}")
         holidays_for_calc = [] # Use empty list if global HOLIDAYS is not right format
    else:
         holidays_for_calc = HOLIDAYS

    open_rows['age_bus'] = np.busday_count(start, end, holidays=holidays_for_calc)

    # Ensure 'week' assignment works even with empty data or NaT dates
    try:
        # Convert Period to string immediately
        open_rows['week'] = open_rows['historydatetime'].dt.to_period('W-MON').astype(str)
    except Exception as e:
        warnings.warn(f"Could not assign 'week' column due to error: {e}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['week', 'total_open', 'open_gt_sla', 'pct_open_gt_sla'])

    out = (
        open_rows
        .groupby('week')
        .agg(
            total_open = ('searchid', 'nunique'),
            open_gt_sla = ('age_bus', lambda x: (x > sla_days).sum())
        )
        .reset_index()
    )
    if out.empty:
        return out # Return empty DataFrame if grouping results in nothing

    # Calculate percentage, handle division by zero
    out['pct_open_gt_sla'] = np.where(
        out['total_open'] > 0,
        (out['open_gt_sla'] / out['total_open'] * 100).round(2),
        0.0 # Assign 0 if total_open is 0
    )
    return out


def calculate_all_metrics(df_history):
    """
    Calculate all metrics from the history DataFrame.
    
    Args:
        df_history: Raw history DataFrame
        
    Returns:
        Dictionary of DataFrames with calculated metrics
    """
    print("\n=== Calculating All Metrics ===")
    
    # Prepare the history DataFrame
    df = prepare_history_df(df_history)
    
    # Calculate all metrics
    tat_metrics = calculate_tat(df)
    attempt_metrics = calculate_attempt_counts(df)
    mb_flags = calculate_mb_capability_flags(df)
    contact_metrics = extract_contacts_from_notes(df)
    email_metrics = calculate_email_metrics(df)
    autonomy_metrics = calculate_autonomy_metrics(df)
    contact_plan_metrics = detect_contact_plans(df)
    phase3a_results = calculate_phase3a_metrics(df)

    # Run the new queue depth calculation using the original df_history
    # as it doesn't depend on the 'df' prepared dataframe's specific flags
    queue_depth = queue_depth_weekly(df_history)

    # Return all metrics in a dictionary
    # Ensure phase3a_results is handled correctly (it returns a dict)
    out: dict[str, pd.DataFrame | dict] = {
        'tat': tat_metrics,
        'attempts': attempt_metrics,
        'mb_flags': mb_flags,
        'contacts': contact_metrics,
        'email': email_metrics,
        'autonomy': autonomy_metrics,
        'contact_plan': contact_plan_metrics,
        'phase3a': phase3a_results,
        'queue_depth_weekly': queue_depth
    }
    return out