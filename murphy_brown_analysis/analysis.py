#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py - Analysis functions for Murphy Brown impact assessment

Contains functions for:
- Merging metrics
- Comparative analysis between groups
- Statistical tests
- Time efficiency analysis
- Contact efficiency analysis
- FTE savings estimation
"""

import pandas as pd
import numpy as np
from scipy import stats
import traceback
from datetime import datetime
from pathlib import Path
import os

# Significance level for statistical tests
SIGNIFICANCE_LEVEL = 0.05


def merge_metrics(metrics_dict):
    """
    Merge all metric DataFrames into a single summary DataFrame.
    
    Args:
        metrics_dict: Dictionary of DataFrames with calculated metrics
        
    Returns:
        Merged DataFrame with all metrics
    """
    print("Merging all metrics into summary DataFrame...")
    
    # Start with basic DataFrame of all search IDs
    search_ids = set()
    for metric_name, metric_df in metrics_dict.items():
        if isinstance(metric_df, pd.DataFrame) and 'searchid' in metric_df.columns:
            search_ids.update(metric_df['searchid'].unique())
    
    # Create base DataFrame with all search IDs
    df_summary = pd.DataFrame({'searchid': list(search_ids)})
    
    # Merge each metric DataFrame
    for metric_name, metric_df in metrics_dict.items():
        if isinstance(metric_df, pd.DataFrame) and 'searchid' in metric_df.columns:
            # Skip 'searchid' column in the merge to avoid duplicates
            merge_cols = [col for col in metric_df.columns if col != 'searchid']
            if merge_cols:
                df_summary = pd.merge(
                    df_summary,
                    metric_df[['searchid'] + merge_cols],
                    on='searchid',
                    how='left'
                )
    
    # Handle special case for Phase3A
    if 'phase3a' in metrics_dict and isinstance(metrics_dict['phase3a'], dict):
        phase3a_summary = metrics_dict['phase3a'].get('summary_df')
        if isinstance(phase3a_summary, pd.DataFrame) and 'searchid' in phase3a_summary.columns:
            # Merge phase3a summary data
            phase3a_cols = [col for col in phase3a_summary.columns if col != 'searchid']
            if phase3a_cols:
                df_summary = pd.merge(
                    df_summary,
                    phase3a_summary[['searchid'] + phase3a_cols],
                    on='searchid',
                    how='left'
                )
    
    # Fill NA values in count columns
    count_cols = [
        'attempts_by_row', 'distinct_applicant_contact_count', 'maxminusapplicant',
        'email_count', 'phone_count', 'fax_count', 'total_contacts',
        'email_interaction_count', 'email_method_count', 'human_touch_count'
    ]
    
    for col in count_cols:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].fillna(0).astype('Int64')
    
    # Fill NA in boolean/flag columns
    flag_cols = [
        'mb_touched', 'mb_contact_research', 'mb_email_handling', 
        'is_completed', 'fully_autonomous', 'has_rework', 'had_fallback',
        'contact_plan_provided'
    ]
    
    for col in flag_cols:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].fillna(0).astype(int)
    
    # For contact_plan_provided, ensure it's boolean for certain calculations
    if 'contact_plan_provided' in df_summary.columns:
        df_summary['contact_plan_provided'] = df_summary['contact_plan_provided'].astype(bool)
    
    # Add capability group column
    if all(col in df_summary.columns for col in ['mb_contact_research', 'mb_email_handling']):
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
    
    print(f"Metrics merged. Summary DataFrame shape: {df_summary.shape}")
    return df_summary


def calculate_capability_impact(df_summary):
    """
    Analyze the impact of different MB capabilities on verification metrics.
    
    Args:
        df_summary: Summary DataFrame with MB capability flags
        
    Returns:
        Dictionary with comparative analysis results
    """
    print("Analyzing impact by MB capability group...")
    
    # Required base columns
    required_cols = ['searchid', 'mb_touched', 'mb_contact_research', 'mb_email_handling']
    if not all(col in df_summary.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_summary.columns]
        print(f"Error: Missing required columns for capability analysis: {missing}")
        return None
    
    # Ensure capability_group exists
    if 'capability_group' not in df_summary.columns:
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
    
    # Define metrics to analyze
    base_metrics = [
        'tat_calendar_days', 'tat_business_days', 'attempts_by_row',
        'distinct_applicant_contact_count', 'maxminusapplicant'
    ]
    
    extended_metrics = [
        'total_contacts', 'email_count', 'phone_count', 'fax_count',
        'email_interaction_count', 'email_method_count', 'human_touch_count'
    ]
    
    # Filter to metrics that exist in the DataFrame
    metrics_to_analyze = [m for m in base_metrics + extended_metrics if m in df_summary.columns]
    
    # Convert metrics to numeric type
    for metric in metrics_to_analyze:
        df_summary[metric] = pd.to_numeric(df_summary[metric], errors='coerce')
    
    # Group by capability_group and calculate metrics
    grouped_metrics = df_summary.groupby('capability_group')[metrics_to_analyze].agg([
        'count', 'mean', 'median', 'std'
    ]).reset_index()
    
    # Perform ANOVA for each metric
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
                    'significant': p_val < SIGNIFICANCE_LEVEL
                }
            except Exception as e:
                print(f"Error performing ANOVA for {metric}: {e}")
                anova_results[metric] = {
                    'F_statistic': np.nan,
                    'p_value': np.nan,
                    'significant': np.nan
                }
        else:
            anova_results[metric] = {
                'F_statistic': np.nan,
                'p_value': np.nan,
                'significant': np.nan
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
    
    return {
        'grouped_metrics': grouped_metrics,
        'anova_results': anova_df,
        'group_counts': group_counts
    }


def analyze_time_efficiency(df_summary):
    """
    Calculate time efficiency metrics including TAT comparisons.
    
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
    if all(col in completed_df.columns for col in ['capability_group']):
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
    completed_df['tat_calendar_days'] = pd.to_numeric(completed_df['tat_calendar_days'], errors='coerce')
    completed_df['tat_business_days'] = pd.to_numeric(completed_df['tat_business_days'], errors='coerce')

    mb_completed = completed_df[completed_df['mb_touched'] == 1]
    non_mb_completed = completed_df[completed_df['mb_touched'] == 0]
    
    stat_results = {}
    for metric in ['tat_calendar_days', 'tat_business_days']:
        if metric not in completed_df.columns:
            continue
            
        data_mb = mb_completed[metric].dropna()
        data_non_mb = non_mb_completed[metric].dropna()
        
        if len(data_mb) >= 2 and len(data_non_mb) >= 2:
            try:
                t_stat, p_value = stats.ttest_ind(data_mb, data_non_mb, equal_var=False)
                significant = p_value < SIGNIFICANCE_LEVEL
                
                stat_results[metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significant
                }
            except Exception as e:
                print(f"Error during t-test for {metric}: {e}")
        else:
            print(f"Insufficient data for t-test on {metric}")
    
    return {
        'tat_stats': tat_stats,
        'capability_stats': capability_stats,
        'statistical_tests': stat_results
    }


def analyze_contact_efficiency(df_summary):
    """
    Calculate contact efficiency metrics including outbound contact breadth, channel yield.
    
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
    contact_metrics_available = not missing_contact
    
    # Filter for completed searches
    completed_df = df_summary[df_summary['is_completed'] == 1].copy()
    if completed_df.empty:
        print("No completed searches found for contact efficiency analysis.")
        return None
    
    # Calculate outbound attempts per verification
    attempts_stats = completed_df.groupby('mb_touched').agg(
        search_count=('searchid', 'count'),
        avg_attempts=('attempts_by_row', 'mean'),
        median_attempts=('attempts_by_row', 'median')
    ).reset_index()
    
    # Calculate contact breadth metrics if available
    if contact_metrics_available:
        contact_stats = completed_df.groupby('mb_touched').agg(
            avg_total_contacts=('total_contacts', 'mean'),
            avg_email_contacts=('email_count', 'mean'),
            avg_phone_contacts=('phone_count', 'mean'),
            avg_fax_contacts=('fax_count', 'mean')
        ).reset_index()
    else:
        contact_stats = None
    
    # Calculate yield metrics if email interaction data is available
    if 'email_interaction_count' in completed_df.columns and 'email_method_count' in completed_df.columns:
        # Calculate success rate for email channel
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
    else:
        yield_stats = None
    
    # Statistical tests for attempts
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
                significant = p_value < SIGNIFICANCE_LEVEL
                
                stat_results[metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': significant
                }
            except Exception as e:
                print(f"Error during t-test for {metric}: {e}")
    
    return {
        'attempts_stats': attempts_stats,
        'contact_stats': contact_stats,
        'yield_stats': yield_stats,
        'statistical_tests': stat_results
    }


def estimate_time_savings(df_summary, df_history, time_per_touch=20):
    """
    Estimate the number of agent hours saved by Murphy Brown.
    
    Args:
        df_summary: Summary DataFrame with human_touch_count
        df_history: Raw history DataFrame for date range calculation
        time_per_touch: Average time in minutes per human touch
        
    Returns:
        Dictionary with time savings & FTE equivalent
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
    mb_df = df_summary[df_summary['mb_touched'] == 1]
    non_mb_df = df_summary[df_summary['mb_touched'] == 0]
    
    if mb_df.empty or non_mb_df.empty:
        print("Warning: Cannot calculate time savings—missing MB or non-MB rows")
        return None

    # Ensure count is float for mean calculation
    av_mb = mb_df['human_touch_count'].astype(float).mean()
    av_non = non_mb_df['human_touch_count'].astype(float).mean()

    # Check if means are NaN
    if pd.isna(av_mb) or pd.isna(av_non):
        print("Warning: Could not calculate average touches. Skipping time savings.")
        return None

    reduction = av_non - av_mb
    hrs_per_ver = reduction * time_per_touch_hours
    count_mb = len(mb_df)
    total_hrs = hrs_per_ver * count_mb

    # FTE calculation over the observed date range
    weeks = 1
    weekly_saved = total_hrs
    fte = np.nan
    
    if 'historydatetime' in df_history.columns and pd.api.types.is_datetime64_any_dtype(df_history['historydatetime']):
        dates = df_history['historydatetime'].dropna()
        if not dates.empty:
            min_date, max_date = dates.min(), dates.max()
            if max_date >= min_date:
                days = (max_date - min_date).days
                weeks = max(1, days) / 7
                if weeks > 0:
                    weekly_saved = total_hrs / weeks
                    fte = weekly_saved / 40  # Assuming 40-hour work week
            else:
                print("Warning: Max date is earlier than min date. Cannot calculate FTE accurately.")
    else:
        print("Warning: Cannot calculate FTE—invalid or missing historydatetime.")

    return {
        'avg_touches_mb': av_mb,
        'avg_touches_non_mb': av_non,
        'touch_reduction': reduction,
        'hours_saved_per_verification': hrs_per_ver,
        'mb_verification_count': count_mb,
        'total_hours_saved': total_hrs,
        'weekly_hours_saved': weekly_saved,
        'fte_equivalent': fte
    }


def analyze_contact_plan_impact(df_summary):
    """
    Calculate the impact of contact plans on completion rates and TAT.
    
    Args:
        df_summary: Summary DataFrame with contact_plan_provided flag
    
    Returns:
        DataFrame with delta metrics
    """
    print("Analyzing contact plan impact...")
    
    # Check if contact_plan_provided exists
    if 'contact_plan_provided' not in df_summary.columns:
        print("Error: 'contact_plan_provided' column not found. Cannot analyze plan impact.")
        return None
    
    # Ensure required columns exist
    required_cols = ['searchid', 'is_completed', 'tat_calendar_days', 'attempts_by_row', 'contact_plan_provided']
    if not all(col in df_summary.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_summary.columns]
        print(f"Warning: Missing required columns for delta calculation: {missing}")
        return None
    
    # Convert metrics to numeric
    df_summary = df_summary.copy()
    df_summary['is_completed'] = pd.to_numeric(df_summary['is_completed'], errors='coerce')
    df_summary['tat_calendar_days'] = pd.to_numeric(df_summary['tat_calendar_days'], errors='coerce')
    df_summary['attempts_by_row'] = pd.to_numeric(df_summary['attempts_by_row'], errors='coerce')
    
    # Group by contact_plan_provided
    grp = df_summary.groupby('contact_plan_provided')
    out = grp.agg(
        search_count=('searchid', 'size'),
        completion_rate=('is_completed', 'mean'),
        avg_tat_days=('tat_calendar_days', 'mean'),
        avg_attempts_by_row=('attempts_by_row', 'mean')
    ).reset_index()
    
    # Calculate delta columns (plan - no-plan)
    if True in out['contact_plan_provided'].values and False in out['contact_plan_provided'].values:
        row_with = out[out['contact_plan_provided'] == True]
        row_without = out[out['contact_plan_provided'] == False]
        
        if not row_with.empty and not row_without.empty:
            delta_completion_rate = np.nan
            delta_tat_days = np.nan
            delta_attempts = np.nan
            
            if pd.notna(row_with['completion_rate'].values[0]) and pd.notna(row_without['completion_rate'].values[0]):
                delta_completion_rate = row_with['completion_rate'].values[0] - row_without['completion_rate'].values[0]
                
            if pd.notna(row_with['avg_tat_days'].values[0]) and pd.notna(row_without['avg_tat_days'].values[0]):
                delta_tat_days = row_with['avg_tat_days'].values[0] - row_without['avg_tat_days'].values[0]
                
            if pd.notna(row_with['avg_attempts_by_row'].values[0]) and pd.notna(row_without['avg_attempts_by_row'].values[0]):
                delta_attempts = row_with['avg_attempts_by_row'].values[0] - row_without['avg_attempts_by_row'].values[0]
            
            # Write delta only to the 'True' row
            out.loc[out.contact_plan_provided == True, 'delta_completion_rate'] = delta_completion_rate
            out.loc[out.contact_plan_provided == True, 'delta_tat_days'] = delta_tat_days
            out.loc[out.contact_plan_provided == True, 'delta_attempts'] = delta_attempts
        else:
            # Initialize delta columns
            out['delta_completion_rate'] = np.nan
            out['delta_tat_days'] = np.nan
            out['delta_attempts'] = np.nan
    else:
        out['delta_completion_rate'] = np.nan
        out['delta_tat_days'] = np.nan
        out['delta_attempts'] = np.nan
    
    return out


def calculate_email_performance_verdict(df_summary, min_pp=5, alpha=0.05):
    """
    Determine if MB email yield is materially better than control.
    
    Args:
        df_summary: Summary DataFrame
        min_pp: Minimum difference in percentage points required
        alpha: Significance level for t-test
    
    Returns:
        Dictionary with verdict and supporting metrics
    """
    print("Calculating email performance verdict...")
    
    # Check required columns
    required_cols = ['searchid', 'mb_touched', 'email_method_count', 'email_interaction_count']
    if not all(col in df_summary.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_summary.columns]
        print(f"Warning: Cannot calculate email verdict. Missing columns: {missing}")
        return {
            'verdict': False,
            'reason': f'missing columns: {missing}',
            'mb_yield': np.nan,
            'ctl_yield': np.nan,
            'diff_pp': np.nan,
            'p_value': np.nan
        }
    
    # Convert counts to numeric
    df_summary_copy = df_summary.copy()
    df_summary_copy['email_method_count'] = pd.to_numeric(df_summary_copy['email_method_count'], errors='coerce')
    df_summary_copy['email_interaction_count'] = pd.to_numeric(df_summary_copy['email_interaction_count'], errors='coerce')
    
    # Keep searches with at least one outbound email
    subset = df_summary_copy[df_summary_copy['email_method_count'] > 0].copy()
    
    if subset.empty:
        print("Info: No searches found with email_method_count > 0 for email verdict.")
        return {
            'verdict': False,
            'reason': 'no searches with email method count > 0',
            'mb_yield': np.nan,
            'ctl_yield': np.nan,
            'diff_pp': np.nan,
            'p_value': np.nan
        }
    
    subset['email_yield'] = subset['email_interaction_count'] / subset['email_method_count']
    
    mb = subset[subset['mb_touched'] == 1]['email_yield'].dropna()
    ctl = subset[subset['mb_touched'] == 0]['email_yield'].dropna()
    
    # Calculate means
    mb_mean = mb.mean() if not mb.empty else np.nan
    ctl_mean = ctl.mean() if not ctl.empty else np.nan
    
    # Calculate difference in percentage points
    diff_pp = (mb_mean - ctl_mean) * 100 if pd.notna(mb_mean) and pd.notna(ctl_mean) else np.nan
    
    if len(mb) < 2 or len(ctl) < 2:
        print(f"Info: Insufficient data for email verdict t-test (MB: {len(mb)}, CTL: {len(ctl)}).")
        return {
            'verdict': False,
            'reason': f'insufficient data (MB: {len(mb)}, CTL: {len(ctl)})',
            'mb_yield': mb_mean,
            'ctl_yield': ctl_mean,
            'diff_pp': diff_pp,
            'p_value': np.nan
        }
    
    # Welch t-test
    try:
        t, p = stats.ttest_ind(mb, ctl, equal_var=False, nan_policy='omit')
    except Exception as e:
        print(f"Error during email yield t-test: {e}")
        return {
            'verdict': False,
            'reason': f't-test error: {e}',
            'mb_yield': mb_mean,
            'ctl_yield': ctl_mean,
            'diff_pp': diff_pp,
            'p_value': np.nan
        }
    
    verdict = (diff_pp >= min_pp) and (p < alpha)
    return {
        'verdict': verdict,
        'mb_yield': mb_mean,
        'ctl_yield': ctl_mean,
        'diff_pp': diff_pp,
        'p_value': p,
        'reason': 'OK'
    }


def calculate_contact_plan_verdict(df_summary, plan_delta):
    """
    Determine if contact plans materially improve verification outcomes.
    
    Args:
        df_summary: Summary DataFrame
        plan_delta: Output from analyze_contact_plan_impact
    
    Returns:
        Dictionary with verdict and supporting metrics
    """
    print("Calculating contact plan verdict...")
    
    # Default verdict
    contact_verdict = {'overall_good': False, 'reason': 'Prerequisites not met'}
    
    if plan_delta is None or plan_delta.empty or 'contact_plan_provided' not in plan_delta.columns:
        return contact_verdict
    
    # Check required columns in df_summary
    required_cols = ['tat_calendar_days', 'contact_plan_provided', 'is_completed', 'attempts_by_row']
    if not all(col in df_summary.columns for col in required_cols):
        contact_verdict = {
            'overall_good': False,
            'reason': f'Missing required columns in df_summary: {[col for col in required_cols if col not in df_summary.columns]}'
        }
        return contact_verdict
    
    try:
        # Find the row with contact_plan_provided = True in plan_delta
        row_with_mask = plan_delta['contact_plan_provided'] == True
        
        if not True in plan_delta['contact_plan_provided'].values:
            return {
                'overall_good': False,
                'reason': 'No rows with contact_plan_provided=True in delta table'
            }
        
        row_with = plan_delta.loc[row_with_mask]
        
        # Check if delta calculations were successful
        if row_with[['delta_tat_days', 'delta_completion_rate', 'delta_attempts']].notna().all(axis=None):
            verdict = {}
            
            # 1. TAT must be lower by at least 1 day and p<0.05
            tat_diff = row_with['delta_tat_days'].iat[0]
            
            # Perform t-test using the main df_summary
            group_with_plan = df_summary[df_summary['contact_plan_provided']]['tat_calendar_days'].dropna()
            group_without_plan = df_summary[~df_summary['contact_plan_provided']]['tat_calendar_days'].dropna()
            
            if len(group_with_plan) >= 2 and len(group_without_plan) >= 2:
                t_stat, p_val = stats.ttest_ind(
                    group_with_plan,
                    group_without_plan,
                    equal_var=False,
                    nan_policy='omit'
                )
                verdict['tat_ok'] = pd.notna(tat_diff) and (tat_diff < -1) and (p_val < 0.05)
                verdict['tat_p_value'] = p_val
            else:
                verdict['tat_ok'] = False
                verdict['tat_p_value'] = np.nan
            
            # 2. Completion-rate must be higher by >=3 pp
            compl_diff = row_with['delta_completion_rate'].iat[0] * 100
            verdict['completion_ok'] = compl_diff > 3
            verdict['completion_diff_pp'] = compl_diff
            
            # 3. Attempts must be lower (direction only)
            att_diff = row_with['delta_attempts'].iat[0]
            verdict['attempts_ok'] = att_diff < 0
            verdict['attempts_diff'] = att_diff
            
            # Combine verdicts
            verdict['overall_good'] = all(verdict.get(k, False) for k in ['tat_ok', 'completion_ok', 'attempts_ok'])
            verdict['reason'] = 'OK'
            
            return verdict
        else:
            return {
                'overall_good': False,
                'reason': 'Delta calculations in plan_delta failed (returned NaN)'
            }
    except Exception as e:
        print(f"Error calculating contact plan verdict: {e}")
        traceback.print_exc()
        return {
            'overall_good': False,
            'reason': f'Error: {e}'
        }


def calculate_agent_completion_uplift(df_summary, agent_col='agentname', min_searches_per_group=5):
    """
    Calculate completion rate uplift per agent when a contact plan is provided.
    
    Args:
        df_summary: Summary DataFrame
        agent_col: Column containing agent names
        min_searches_per_group: Minimum searches required per group
        
    Returns:
        DataFrame with agent uplift statistics
    """
    print(f"Calculating agent completion uplift using column: {agent_col}")
    
    required_cols = [agent_col, 'contact_plan_provided', 'is_completed', 'searchid']
    if not all(col in df_summary.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_summary.columns]
        print(f"Warning: Cannot calculate agent uplift. Missing columns: {missing}")
        return pd.DataFrame()
    
    # Ensure proper column types
    df_summary = df_summary.copy()
    df_summary[agent_col] = df_summary[agent_col].fillna('Unknown Agent').astype(str)
    df_summary['contact_plan_provided'] = df_summary['contact_plan_provided'].astype(bool)
    df_summary['is_completed'] = pd.to_numeric(df_summary['is_completed'], errors='coerce')
    
    try:
        # Group by agent and plan status, calculate mean completion
        g = df_summary.groupby([agent_col, 'contact_plan_provided'])['is_completed'].mean().unstack()
        
        # Calculate counts for filtering
        counts = df_summary.groupby([agent_col, 'contact_plan_provided']).size().unstack(fill_value=0)
        g = g.join(counts, rsuffix='_count')
        
        # Filter agents
        g = g.dropna(subset=[True, False])
        g = g[(g['True_count'] >= min_searches_per_group) & (g['False_count'] >= min_searches_per_group)]
    except Exception as e:
        print(f"Error during grouping for agent uplift: {e}")
        return pd.DataFrame()
    
    if g.empty:
        print("No agents found meeting the criteria for completion uplift analysis.")
        return pd.DataFrame()
    
    g['delta_completion_pp'] = (g[True] - g[False]) * 100
    
    # Check significance with t-test
    sig = []
    p_values = []
    agent_list = g.index
    
    df_filtered_agents = df_summary[df_summary[agent_col].isin(agent_list)]
    
    for agent, sub in df_filtered_agents.groupby(agent_col):
        if agent not in agent_list:
            continue
        
        plan = sub[sub.contact_plan_provided]['is_completed'].dropna()
        nolp = sub[~sub.contact_plan_provided]['is_completed'].dropna()
        
        if len(plan) >= min_searches_per_group and len(nolp) >= min_searches_per_group:
            try:
                if plan.var(ddof=0) > 0 and nolp.var(ddof=0) > 0:
                    _, p = stats.ttest_ind(plan, nolp, equal_var=False, nan_policy='omit')
                    sig.append(p < 0.05)
                    p_values.append(p)
                else:
                    sig.append(False)
                    p_values.append(np.nan)
            except Exception as e:
                print(f"Error during t-test for agent {agent}: {e}")
                sig.append(False)
                p_values.append(np.nan)
        else:
            sig.append(False)
            p_values.append(np.nan)
    
    # Add results to DataFrame
    if len(sig) == len(g):
        g['significant'] = sig
        g['p_value'] = p_values
    else:
        g['significant'] = False
        g['p_value'] = np.nan
    
    return g.reset_index()


def calculate_first_day_sla_rate(df_summary):
    """
    Calculate percentage of searches closed in ≤1 calendar day, MB vs. non-MB.
    
    Args:
        df_summary: Summary DataFrame
        
    Returns:
        DataFrame with SLA rates by MB status
    """
    print("Calculating first day SLA rate...")
    
    required = {'tat_calendar_days', 'mb_touched'}
    if not required.issubset(df_summary.columns):
        print(f"Skipping first day SLA rate – missing {required.difference(df_summary.columns)}")
        return None
    
    # Ensure TAT is numeric
    df_copy = df_summary.copy()
    df_copy['tat_calendar_days'] = pd.to_numeric(df_copy['tat_calendar_days'], errors='coerce')
    
    # Add flag
    df_copy['first_day_close'] = np.where(
        df_copy['tat_calendar_days'].notna(),
        (df_copy['tat_calendar_days'] <= 1).astype(int),
        0
    )
    
    try:
        out = (
            df_copy.groupby('mb_touched')['first_day_close']
                  .mean()
                  .rename('sla_rate')
                  .reset_index()
        )
        return out
    except Exception as e:
        print(f"Error during first day SLA grouping: {e}")
        return None


def calculate_throughput(df_summary, period='week', agent_col='agentname'):
    """
    Calculate verified count per agent by time period.
    
    Args:
        df_summary: Summary DataFrame
        period: Time period ('week', 'month', or 'day')
        agent_col: Column containing agent names
        
    Returns:
        DataFrame with throughput statistics
    """
    print(f"Calculating {period}ly verified count per agent...")
    
    required = {agent_col, 'completiondate', 'is_completed'}
    if not required.issubset(df_summary.columns):
        print(f"Skipping {period}ly throughput – missing {required.difference(df_summary.columns)}")
        return pd.DataFrame()
    
    # Ensure proper column types
    df_summary = df_summary.copy()
    df_summary[agent_col] = df_summary[agent_col].fillna('Unknown Agent').astype(str)
    df_summary['is_completed'] = pd.to_numeric(df_summary['is_completed'], errors='coerce')
    
    if not pd.api.types.is_datetime64_any_dtype(df_summary['completiondate']):
        df_summary['completiondate'] = pd.to_datetime(df_summary['completiondate'], errors='coerce')
    
    # Filter to completed searches with valid completion dates
    tmp = df_summary[(df_summary['is_completed'] == 1) & df_summary['completiondate'].notna()].copy()
    
    if tmp.empty:
        print(f"No completed searches with valid completion dates found for {period}ly throughput.")
        return pd.DataFrame()
    
    # Create period column based on specified period
    if period == 'week':
        tmp['period'] = tmp['completiondate'].dt.to_period('W-MON').astype(str)
    elif period == 'month':
        tmp['period'] = tmp['completiondate'].dt.to_period('M').astype(str)
    elif period == 'day':
        tmp['period'] = tmp['completiondate'].dt.date
    else:
        print(f"Invalid period '{period}'. Use 'week', 'month', or 'day'.")
        return pd.DataFrame()
    
    try:
        out = (
            tmp.groupby([agent_col, 'period'], as_index=False)
               .size()
               .rename(columns={'size': 'verified_count'})
        )
        return out
    except Exception as e:
        print(f"Error during {period}ly verified grouping: {e}")
        return pd.DataFrame()


def calculate_mb_category_metrics(df_summary: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    """
    Build category counts, descriptive stats and ANOVA tables.

    Returns {'category_counts': … , 'category_metrics': …}  – or None if
    required columns are missing.
    """
    need = {'mb_contact_research', 'mb_email_handling'}
    if not need.issubset(df_summary.columns):
        return None

    df = df_summary.copy()
    df['mb_category'] = 'No MB'
    df.loc[(df.mb_contact_research == 1) & (df.mb_email_handling == 0), 'mb_category'] = 'Contact Research Only'
    df.loc[(df.mb_contact_research == 0) & (df.mb_email_handling == 1), 'mb_category'] = 'Email Handling Only'
    df.loc[(df.mb_contact_research == 1) & (df.mb_email_handling == 1), 'mb_category'] = 'Both Capabilities'

    counts = df['mb_category'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']

    metrics = [
        'tat_calendar_days', 'tat_business_days', 'attempts_by_row',
        'distinct_applicant_contact_count', 'maxminusapplicant',
        'total_contacts', 'email_count', 'phone_count', 'fax_count',
        'email_interaction_count', 'email_method_count',
        'human_touch_count', 'fully_autonomous', 'has_rework'
    ]
    avail = [m for m in metrics if m in df.columns]

    if not avail:
        return {'category_counts': counts}

    completed = df[df.is_completed == 1] if 'is_completed' in df.columns else df
    rollup = (
        completed
        .groupby('mb_category')
        .agg(
            count=('searchid', 'nunique'),
            **{f'{m}_mean':   (m, 'mean') for m in avail},
            **{f'{m}_median': (m, 'median') for m in avail}
        )
        .reset_index()
    )

    # Simple one-way ANOVA for each metric
    anova = {}
    for m in avail:
        groups = [
            completed.loc[completed.mb_category == g, m].dropna()
            for g in rollup['mb_category']
        ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
            anova[m] = {'F': f, 'p': p, 'sig': p < SIGNIFICANCE_LEVEL}

    return {
        'category_counts': counts,
        'category_metrics': rollup,
        'anova_results': pd.DataFrame.from_dict(anova, orient='index')
                                .reset_index(names=['Metric'])
    }


def export_monthly_reports(df_summary: pd.DataFrame,
                           df_history: pd.DataFrame,
                           out_root: str) -> None:
    """
    Convenience artefacts identical to the ones V8 produced:
      • Excel workbook (overall / by office / by agent) of monthly completions
      • 15-day, completed-only detailed sample CSV
    """
    out_dir = Path(out_root) / "monthly_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Workbook
    comp = df_summary[df_summary.is_completed == 1].copy()
    if not comp.empty and 'completiondate' in comp.columns:
        comp['period'] = comp['completiondate'].dt.to_period('M').astype(str)

        overall = (
            comp.groupby(['period', 'searchtype'])['searchid']
            .count().reset_index(name='completions')
        )

        office = (
            comp.groupby(['period', 'office', 'searchtype'])['searchid']
            .count().reset_index(name='completions')
            if 'office' in comp else pd.DataFrame()
        )

        agent = (
            comp.groupby(['period', 'agentname', 'searchtype'])['searchid']
            .count().reset_index(name='completions')
            if 'agentname' in comp else pd.DataFrame()
        )

        xl = out_dir / f"monthly_completions_{pd.Timestamp.utcnow():%Y%m%d}.xlsx"
        with pd.ExcelWriter(xl, engine='xlsxwriter') as w:
            overall.to_excel(w, 'Overall', index=False)
            if not office.empty: office.to_excel(w, 'By Office', index=False)
            if not agent.empty:  agent.to_excel(w, 'By Agent',  index=False)
        print(f"Saved workbook → {xl}")

    # 2. 15-day sample ending one week before max(historydatetime)
    if 'historydatetime' in df_history.columns:
        max_date = pd.to_datetime(df_history['historydatetime']).max()
        if pd.notna(max_date):
            end   = max_date - pd.Timedelta(days=7)
            start = end - pd.Timedelta(days=14)
            window = df_history[
                (df_history['historydatetime'] >= start) &
                (df_history['historydatetime'] <= end)
            ]
            complete_ids = df_summary.loc[df_summary.is_completed == 1, 'searchid']
            sample = window[window['searchid'].isin(complete_ids)]
            if not sample.empty:
                csv = out_dir / f"sample_15d_completed_{pd.Timestamp.utcnow():%Y%m%d}.csv"
                sample.to_csv(csv, index=False)
                print(f"Saved 15-day sample → {csv}")


def summarize_time_efficiency_by_capability(df_summary: pd.DataFrame
                                            ) -> pd.DataFrame | None:
    """
    Recreates v8's 'capability_stats':
    count, mean, median TAT (calendar / business days) for each capability group
    (No MB, Contact-only, Email-only, Both).
    """
    need_cols = {'mb_contact_research', 'mb_email_handling',
                 'tat_calendar_days', 'tat_business_days'}
    if not need_cols.issubset(df_summary.columns):
        return None

    df = df_summary.copy()
    # Build categorical label
    cat_map = {(0, 0): "No MB",
               (1, 0): "Contact Only",
               (0, 1): "Email Only",
               (1, 1): "Both"}
    df['capability_group'] = df[['mb_contact_research',
                                 'mb_email_handling']].apply(tuple, axis=1).map(cat_map)

    # Ensure numeric
    df['tat_calendar_days'] = pd.to_numeric(df['tat_calendar_days'],  errors='coerce')
    df['tat_business_days'] = pd.to_numeric(df['tat_business_days'], errors='coerce')

    stats_tbl = (df
                 .groupby('capability_group')
                 .agg(n_cal=('tat_calendar_days', 'count'),
                      mean_cal=('tat_calendar_days', 'mean'),
                      median_cal=('tat_calendar_days', 'median'),
                      n_bus=('tat_business_days', 'count'),
                      mean_bus=('tat_business_days', 'mean'),
                      median_bus=('tat_business_days', 'median'))
                 .reset_index())
    return stats_tbl


def summarize_autonomy(df_autonomy: pd.DataFrame) -> dict:
    """Summarize autonomy metrics from the detailed autonomy DataFrame."""
    if df_autonomy is None or df_autonomy.empty:
        return {}
    rate      = (df_autonomy['fully_autonomous'] == 1).mean() * 100
    rework    = (df_autonomy['has_rework']       == 1).mean() * 100
    fallback  = (df_autonomy['had_fallback']     == 1).mean() * 100
    avg_touch = df_autonomy['human_touch_count'].mean()
    return {
        'summary_metrics': {
            'autonomy_rate':   round(rate, 2),
            'rework_rate':     round(rework, 2),
            'fallback_rate':   round(fallback, 2),
            'avg_human_touches': round(avg_touch, 2)
        },
        'detail': df_autonomy
    }


def perform_full_analysis(all_metrics: dict,
                          df_history: pd.DataFrame) -> dict:
    """
    Runs every high-level analysis step and returns a big dict.
    """
    print("\n=== Performing Full Analysis ===")

    # ----- your existing integration code stays here -----
    # Note: Assuming merge_metrics_to_summary, analyze_mb_capability_impact,
    # and summarize_autonomy exist elsewhere or are defined appropriately.
    df_summary = merge_metrics(all_metrics) # Corrected function name
    results    = {}

    # capability, time, contact efficiency … (unchanged sections from user request)
    results['capability_impact']    = calculate_capability_impact(df_summary) # Corrected function name
    results['time_efficiency']      = analyze_time_efficiency(df_summary) # Preserved original name
    results['contact_efficiency']   = analyze_contact_efficiency(df_summary) # Preserved original name
    results['autonomy']             = summarize_autonomy(all_metrics['autonomy']) # Re-enabled this line
    # Extract Phase 3A scalar metrics and the summary DataFrame
    phase3a_payload = all_metrics.get('phase3a', {})
    results['phase3a_metrics']      = phase3a_payload.get('metrics')
    phase3a_summary_df              = phase3a_payload.get('summary_df')
    # Optional: Merge phase3a_summary_df if needed
    # if isinstance(phase3a_summary_df, pd.DataFrame) and not phase3a_summary_df.empty and 'searchid' in phase3a_summary_df.columns:
    #     df_summary = pd.merge(df_summary, phase3a_summary_df[['searchid', 'auto_complete']], on='searchid', how='left')

    # Adding capability_stats call here as per user's block 3 example structure
    cap_stats = summarize_time_efficiency_by_capability(df_summary)
    if cap_stats is not None:
        results['capability_stats'] = cap_stats

    # ---- NEW bits as per user request ----
    results['mb_category_metrics']  = calculate_mb_category_metrics(df_summary)

    # queue depth is already in `all_metrics` (as per user comment)
    if 'queue_depth_weekly' in all_metrics: # Added check for safety
        results['queue_depth_weekly']   = all_metrics['queue_depth_weekly']
    else:
        print("Warning: 'queue_depth_weekly' not found in all_metrics.")
        results['queue_depth_weekly'] = None # Or some default

    # --- Preserving other original calculations ---
    # Note: These were omitted in the user's provided structure for perform_full_analysis,
    # but preserving them as they were part of the original file and not explicitly removed.
    # If they should be removed, please clarify.
    results['time_savings'] = estimate_time_savings(df_summary, df_history) # Preserved original
    results['plan_delta'] = analyze_contact_plan_impact(df_summary) # Preserved original
    results['email_verdict'] = calculate_email_performance_verdict(df_summary) # Preserved original
    results['contact_plan_verdict'] = calculate_contact_plan_verdict(df_summary, results['plan_delta']) # Preserved original
    results['agent_uplift'] = calculate_agent_completion_uplift(df_summary) # Preserved original
    results['sla_rate'] = calculate_first_day_sla_rate(df_summary) # Preserved original
    results['weekly_throughput'] = calculate_throughput(df_summary, 'week') # Preserved original
    results['monthly_throughput'] = calculate_throughput(df_summary, 'month') # Preserved original
    results['daily_throughput'] = calculate_throughput(df_summary, 'day') # Preserved original

    # Monthly Excel & 15-day sample (Placed before final return, after calculations)
    export_monthly_reports(
        df_summary, df_history,
        out_root=os.path.join(os.path.dirname(__file__), "Output") # Using os.path.join for robustness
    )

    # Final catch-all
    results['df_summary'] = df_summary # Original placement seems fine

    # Phase 3a handling - Covered by results['phase3a_metrics'] line above as per user request.
    # Original code below is now redundant based on user's snippet.
    # if 'phase3a' in all_metrics and isinstance(all_metrics['phase3a'], dict):
    #    all_results['phase3a_metrics'] = all_metrics['phase3a'].get('metrics', {})

    print("Full analysis complete.") # Keep original print
    return results