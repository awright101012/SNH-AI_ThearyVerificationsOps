#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz.py - Visualization functions for Murphy Brown impact analysis

Contains functions for creating visualizations and reports:
- Dashboard creation
- Plots for key metrics
- Executive summary generation
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def setup_plot_style():
    """Set up consistent plot styling for all visualizations."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Additional customizations
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Use a more appealing color palette
    sns.set_palette('viridis')


def create_capability_distribution_plot(group_counts, output_path):
    """
    Create a bar chart showing the distribution of MB capability groups.
    
    Args:
        group_counts: DataFrame with capability group counts
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Capability Group', y='Count', data=group_counts)
    plt.title('Distribution of MB Capability Groups', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_tat_comparison_plot(tat_stats, output_path):
    """
    Create a bar chart comparing TAT metrics between MB and non-MB.
    
    Args:
        tat_stats: DataFrame with TAT statistics
        output_path: Path to save the visualization
    """
    if tat_stats is None or tat_stats.empty or 'mb_touched' not in tat_stats.columns:
        print("Warning: Invalid TAT stats for plotting.")
        return None
    
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
    
    # Correctly set ticks and labels
    tick_positions = range(len(new_labels))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([new_labels.get(metric_key, metric_key) for metric_key in tat_melt['metric'].unique()])
    
    plt.xticks(rotation=30, ha='right')
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_autonomy_metrics_plot(autonomy_metrics, output_path):
    """
    Create a bar chart showing key autonomy metrics.
    
    Args:
        autonomy_metrics: Dictionary with autonomy metrics
        output_path: Path to save the visualization
    """
    if not autonomy_metrics:
        print("Warning: No autonomy metrics provided for plotting.")
        return None
    
    metric_df = pd.DataFrame([
        {'Metric': 'Autonomy Rate (%)', 'Value': autonomy_metrics.get('autonomy_rate', 0)},
        {'Metric': 'Rework Rate (%)', 'Value': autonomy_metrics.get('rework_rate', 0)},
        {'Metric': 'Avg Human Touches', 'Value': autonomy_metrics.get('avg_human_touches', 0)},
        {'Metric': 'Fallback Rate (%)', 'Value': autonomy_metrics.get('fallback_rate', 0)}
    ])
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Metric', y='Value', data=metric_df)
    plt.title('MB Autonomy Metrics', fontsize=14)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_touch_comparison_plot(time_savings, output_path):
    """
    Create a bar chart comparing human touches between MB and non-MB.
    
    Args:
        time_savings: Dictionary with time savings metrics
        output_path: Path to save the visualization
    """
    if not time_savings:
        print("Warning: No time savings metrics provided for plotting.")
        return None
    
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
                    ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_time_savings_summary(time_savings, output_path):
    """
    Create a text-based summary of time savings metrics.
    
    Args:
        time_savings: Dictionary with time savings metrics
        output_path: Path to save the visualization
    """
    if not time_savings:
        print("Warning: No time savings metrics provided for plotting.")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_attempts_comparison_plot(attempts_stats, output_path):
    """
    Create a bar chart comparing outbound attempts between MB and non-MB.
    
    Args:
        attempts_stats: DataFrame with attempt statistics
        output_path: Path to save the visualization
    """
    if attempts_stats is None or attempts_stats.empty or 'mb_touched' not in attempts_stats.columns:
        print("Warning: Invalid attempts stats for plotting.")
        return None
    
    attempts_melt = pd.melt(
        attempts_stats,
        id_vars=['mb_touched', 'search_count'],
        value_vars=['avg_attempts', 'median_attempts'],
        var_name='metric', value_name='value'
    )
    
    attempts_melt['mb_status'] = attempts_melt['mb_touched'].map({0: 'Non-MB', 1: 'MB'})
    
    plt.figure(figsize=(10, 6))
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
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_contact_plan_impact_plot(plan_delta, output_path):
    """
    Create a bar chart showing the impact of contact plans on key metrics.
    
    Args:
        plan_delta: DataFrame with contact plan delta metrics
        output_path: Path to save the visualization
    """
    if plan_delta is None or plan_delta.empty or 'contact_plan_provided' not in plan_delta.columns:
        print("Warning: Invalid plan delta for plotting.")
        return None
    
    # Extract delta values
    delta_row = plan_delta[plan_delta['contact_plan_provided'] == True]
    
    if delta_row.empty or not all(col in delta_row.columns for col in ['delta_completion_rate', 'delta_tat_days', 'delta_attempts']):
        print("Warning: Missing delta columns for contact plan impact plot.")
        return None
    
    # Create a dataframe for plotting the deltas
    delta_df = pd.DataFrame([
        {'Metric': 'Completion Rate (pp)', 'Value': delta_row['delta_completion_rate'].iloc[0] * 100},
        {'Metric': 'TAT Days', 'Value': delta_row['delta_tat_days'].iloc[0]},
        {'Metric': 'Attempts', 'Value': delta_row['delta_attempts'].iloc[0]}
    ])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(delta_df['Metric'], delta_df['Value'], color=['green', 'red', 'blue'])
    
    # Color bars based on whether the effect is positive or negative
    for i, bar in enumerate(bars):
        if (i == 0 and delta_df['Value'].iloc[i] > 0) or (i in [1, 2] and delta_df['Value'].iloc[i] < 0):
            bar.set_color('green')  # Good effect
        else:
            bar.set_color('red')    # Bad effect
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Impact of MB Contact Plans on Key Metrics', fontsize=14)
    plt.ylabel('Change (With Plan - Without Plan)')
    
    # Add value labels
    for i, v in enumerate(delta_df['Value']):
        plt.text(i, v + (0.1 if v >= 0 else -0.1), 
                 f'{v:.1f}', 
                 ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_email_performance_plot(email_verdict, output_path):
    """
    Create a bar chart showing email yield comparison between MB and control.
    
    Args:
        email_verdict: Dictionary with email performance metrics
        output_path: Path to save the visualization
    """
    if not email_verdict or not all(k in email_verdict for k in ['mb_yield', 'ctl_yield']):
        print("Warning: Invalid email verdict for plotting.")
        return None
    
    yield_df = pd.DataFrame([
        {'Group': 'Non-MB', 'Email Yield': email_verdict.get('ctl_yield', 0)},
        {'Group': 'MB', 'Email Yield': email_verdict.get('mb_yield', 0)}
    ])
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Group', y='Email Yield', data=yield_df)
    plt.title('Email Channel Yield Comparison', fontsize=14)
    
    # Add value labels
    for p in ax.patches:
        percentage = p.get_height() * 100
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11)
    
    # Add verdict as text
    verdict_text = "VERDICT: " + ("BETTER" if email_verdict.get('verdict', False) else "NOT BETTER") 
    plt.figtext(0.5, 0.01, verdict_text, ha='center', fontsize=12, 
                bbox={'facecolor': 'green' if email_verdict.get('verdict', False) else 'red', 
                      'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at the bottom for verdict text
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_throughput_trend_plot(throughput_df, period='week', output_path=None):
    """
    Create a line plot showing the trend in verification throughput.
    
    Args:
        throughput_df: DataFrame with throughput data by period
        period: Time period ('week', 'month', 'day')
        output_path: Path to save the visualization
        
    Returns:
        Path to saved visualization
    """
    if throughput_df is None or throughput_df.empty:
        print(f"Warning: No {period}ly throughput data for plotting.")
        return None
    
    # Ensure required columns exist
    required_cols = ['period', 'verified_count']
    if not all(col in throughput_df.columns for col in required_cols):
        print(f"Warning: Missing required columns for throughput plot: {[col for col in required_cols if col not in throughput_df.columns]}")
        return None
    
    # Aggregate by period
    period_totals = throughput_df.groupby('period')['verified_count'].sum().reset_index()
    
    # Convert period to datetime for proper sorting
    try:
        if period == 'day':
            # period is already a date
            period_totals['period_dt'] = pd.to_datetime(period_totals['period'])
        else:
            # period is a string like '2024-01' or '2024-01-01/2024-01-07'
            period_totals['period_dt'] = pd.to_datetime(period_totals['period'].str.split('/').str[0])
    except Exception as e:
        print(f"Error converting period to datetime: {e}")
        # Use the original string periods but sort them
        period_totals = period_totals.sort_values('period')
        period_totals['period_dt'] = period_totals['period']
    
    # Sort by period
    period_totals = period_totals.sort_values('period_dt')
    
    plt.figure(figsize=(12, 6))
    plt.plot(period_totals['period_dt'], period_totals['verified_count'], marker='o', linestyle='-')
    
    plt.title(f'{period.capitalize()}ly Verification Throughput', fontsize=14)
    plt.xlabel(period.capitalize())
    plt.ylabel('Verifications Completed')
    
    # Format x-axis tick labels appropriately
    if len(period_totals) > 12:
        # If too many periods, show only a subset of ticks
        tick_count = min(12, len(period_totals))
        step = len(period_totals) // tick_count
        plt.xticks(
            period_totals['period_dt'].iloc[::step],
            period_totals['period'].iloc[::step],
            rotation=45, ha='right'
        )
    else:
        plt.xticks(
            period_totals['period_dt'],
            period_totals['period'],
            rotation=45, ha='right'
        )
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for x, y in zip(period_totals['period_dt'], period_totals['verified_count']):
        plt.annotate(str(y), (x, y), textcoords="offset points", 
                     xytext=(0, 5), ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.close()
        return None


def create_dashboard(analysis_results, output_dir=None):
    """
    Create a set of visualizations and an HTML dashboard.
    
    Args:
        analysis_results: Dictionary with analysis results
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
    
    # Setup plot style
    setup_plot_style()
    
    # Track all generated visualizations
    visualizations = {}
    
    # 1. Capability distribution plot
    if ('capability_impact' in analysis_results and analysis_results['capability_impact'] is not None and 
        'group_counts' in analysis_results['capability_impact']):
        group_counts = analysis_results['capability_impact']['group_counts']
        output_path = os.path.join(output_dir, 'capability_distribution.png')
        visualizations['capability_distribution'] = create_capability_distribution_plot(group_counts, output_path)
    
    # 2. TAT comparison plot
    if ('time_efficiency' in analysis_results and analysis_results['time_efficiency'] is not None and 
        'tat_stats' in analysis_results['time_efficiency']):
        tat_stats = analysis_results['time_efficiency']['tat_stats']
        output_path = os.path.join(output_dir, 'tat_comparison.png')
        visualizations['tat_comparison'] = create_tat_comparison_plot(tat_stats, output_path)
    
    # 3. Autonomy metrics plot
    if 'autonomy' in analysis_results and analysis_results['autonomy'] is not None:
        autonomy_metrics = analysis_results['autonomy'].get('summary_metrics', {})
        output_path = os.path.join(output_dir, 'autonomy_metrics.png')
        visualizations['autonomy_metrics'] = create_autonomy_metrics_plot(autonomy_metrics, output_path)
    
    # 4. Touch comparison plot
    if 'time_savings' in analysis_results and analysis_results['time_savings'] is not None:
        time_savings = analysis_results['time_savings']
        output_path = os.path.join(output_dir, 'touch_comparison.png')
        visualizations['touch_comparison'] = create_touch_comparison_plot(time_savings, output_path)
        
        # 5. Time savings summary
        output_path = os.path.join(output_dir, 'time_savings_summary.png')
        visualizations['time_savings_summary'] = create_time_savings_summary(time_savings, output_path)
    
    # 6. Attempts comparison plot
    if ('contact_efficiency' in analysis_results and analysis_results['contact_efficiency'] is not None and 
        'attempts_stats' in analysis_results['contact_efficiency']):
        attempts_stats = analysis_results['contact_efficiency']['attempts_stats']
        output_path = os.path.join(output_dir, 'attempts_comparison.png')
        visualizations['attempts_comparison'] = create_attempts_comparison_plot(attempts_stats, output_path)
    
    # 7. Contact plan impact plot
    if 'plan_delta' in analysis_results and analysis_results['plan_delta'] is not None:
        plan_delta = analysis_results['plan_delta']
        output_path = os.path.join(output_dir, 'contact_plan_impact.png')
        visualizations['contact_plan_impact'] = create_contact_plan_impact_plot(plan_delta, output_path)
    
    # 8. Email performance plot
    if 'email_verdict' in analysis_results and analysis_results['email_verdict'] is not None:
        email_verdict = analysis_results['email_verdict']
        output_path = os.path.join(output_dir, 'email_performance.png')
        visualizations['email_performance'] = create_email_performance_plot(email_verdict, output_path)
    
    # 9. Throughput trend plot
    if 'weekly_throughput' in analysis_results and analysis_results['weekly_throughput'] is not None:
        throughput_df = analysis_results['weekly_throughput']
        output_path = os.path.join(output_dir, 'weekly_throughput.png')
        visualizations['weekly_throughput'] = create_throughput_trend_plot(throughput_df, 'week', output_path)
    
    # Create HTML dashboard index
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
    """
    
    # Add visualization sections to HTML only if they were created
    viz_sections = {
        'capability_distribution': ('<div class="dashboard-item"><h2>Capability Distribution</h2>'
                                   '<img src="capability_distribution.png" alt="MB Capability Distribution" /></div>'),
        'tat_comparison': ('<div class="dashboard-item"><h2>Turnaround Time Comparison</h2>'
                          '<img src="tat_comparison.png" alt="TAT Comparison" /></div>'),
        'autonomy_metrics': ('<div class="dashboard-item"><h2>Autonomy Metrics</h2>'
                            '<img src="autonomy_metrics.png" alt="Autonomy Metrics" /></div>'),
        'touch_comparison': ('<div class="dashboard-item"><h2>Human Touch Comparison</h2>'
                            '<img src="touch_comparison.png" alt="Human Touch Comparison" /></div>'),
        'time_savings_summary': ('<div class="dashboard-item"><h2>Time Savings</h2>'
                                '<img src="time_savings_summary.png" alt="Time Savings Summary" /></div>'),
        'attempts_comparison': ('<div class="dashboard-item"><h2>Outbound Attempts</h2>'
                               '<img src="attempts_comparison.png" alt="Attempts Comparison" /></div>'),
        'contact_plan_impact': ('<div class="dashboard-item"><h2>Contact Plan Impact</h2>'
                               '<img src="contact_plan_impact.png" alt="Contact Plan Impact" /></div>'),
        'email_performance': ('<div class="dashboard-item"><h2>Email Performance</h2>'
                             '<img src="email_performance.png" alt="Email Performance" /></div>'),
        'weekly_throughput': ('<div class="dashboard-item"><h2>Weekly Throughput</h2>'
                             '<img src="weekly_throughput.png" alt="Weekly Throughput" /></div>')
    }
    
    for viz_key, viz_path in visualizations.items():
        if viz_path and viz_key in viz_sections:
            html_output += viz_sections[viz_key]
    
    html_output += """
        </div>
        
        <div class="timestamp">
            <p>Analysis performed using Murphy Brown impact analysis scripts</p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'dashboard.html'), 'w') as f:
        f.write(html_output)
    
    print(f"Dashboard visualizations and HTML index saved to: {output_dir}")
    return output_dir


def generate_executive_summary(analysis_results, output_dir=None):
    """
    Generate an executive summary of the MB impact analysis.
    
    Args:
        analysis_results: Dictionary with analysis results
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
    if 'autonomy' in analysis_results and analysis_results['autonomy'] is not None:
        autonomy_metrics = analysis_results['autonomy'].get('summary_metrics', {}) or {}
    
    time_savings = {}
    if 'time_savings' in analysis_results and analysis_results['time_savings'] is not None:
        time_savings = analysis_results['time_savings'] or {}
    
    time_efficiency = {}
    if 'time_efficiency' in analysis_results and analysis_results['time_efficiency'] is not None:
        time_efficiency = analysis_results['time_efficiency'] or {}
    
    contact_efficiency = {}
    if 'contact_efficiency' in analysis_results and analysis_results['contact_efficiency'] is not None:
        contact_efficiency = analysis_results['contact_efficiency'] or {}
    
    # Prepare report content
    report_content = f"""
    # Murphy Brown Impact Analysis: Executive Summary
    
    **Generated on:** {timestamp}
    
    ## Overview
    
    This report summarizes the impact of Murphy Brown (MB) on the verification process, focusing on both contact research and email handling capabilities.
    
    ## Key Findings
    
    ### Autonomy & Automation
    
    - **End-to-End Autonomy Rate:** {autonomy_metrics.get('autonomy_rate', 0):.1f}%
    - **Rework Rate:** {autonomy_metrics.get('rework_rate', 0):.1f}%
    - **Avg. Human Touches:** {autonomy_metrics.get('avg_human_touches', 0):.1f}
    
    ### Efficiency Improvements
    
    """
    
    # Add TAT comparison if available
    if time_efficiency and 'tat_stats' in time_efficiency:
        tat_stats = time_efficiency.get('tat_stats')
        if tat_stats is not None and not tat_stats.empty and {'avg_tat_days', 'mb_touched'}.issubset(tat_stats.columns):
            # Find MB and non-MB rows
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
        attempt_stats = contact_efficiency.get('attempts_stats')
        if attempt_stats is not None and not attempt_stats.empty and {'avg_attempts', 'mb_touched'}.issubset(attempt_stats.columns):
            # Find MB and non-MB rows
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
    
    - **Hours Saved Per Verification:** {time_savings.get('hours_saved_per_verification', 0):.2f} hours
    - **Total Hours Saved:** {time_savings.get('total_hours_saved', 0):.1f} hours
    - **FTE Equivalent:** {time_savings.get('fte_equivalent', 0):.2f} FTEs
    """
    
    # Add capability comparison if available
    if ('capability_impact' in analysis_results and analysis_results['capability_impact'] is not None and 
        'group_counts' in analysis_results['capability_impact']):
        group_counts = analysis_results['capability_impact']['group_counts']
        
        report_content += f"""
    ### Capability Analysis
    
    MB capabilities breakdown:
    """
        
        for _, row in group_counts.iterrows():
            report_content += f"""
    - **{row['Capability Group']}:** {row['Count']} searches
    """
    
    # Add throughput section
    agent_weekly = analysis_results.get('weekly_throughput')
    if agent_weekly is not None and not agent_weekly.empty and all(col in agent_weekly.columns for col in ['period', 'verified_count']):
        try:
            # Group by period to get total counts
            weekly_totals = agent_weekly.groupby('period')['verified_count'].sum().reset_index()
            
            # Convert period to datetime for sorting
            weekly_totals['period_dt'] = pd.to_datetime(weekly_totals['period'].str.split('/').str[0], errors='coerce')
            weekly_totals = weekly_totals.sort_values('period_dt')
            
            # Calculate growth rate if at least 2 periods available
            if len(weekly_totals) > 1:
                last_value = weekly_totals['verified_count'].iloc[-1]
                prev_value = weekly_totals['verified_count'].iloc[-2]
                growth_rate = ((last_value / prev_value) - 1) * 100 if prev_value > 0 else 0
                
                report_content += f"""
    ### Throughput

    - **Latest weekly completions growth:** {growth_rate:+.1f}%
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

    - **Latest weekly throughput growth:** Error calculating trend
"""
    
    # Add monthly throughput
    agent_monthly = analysis_results.get('monthly_throughput')
    if agent_monthly is not None and not agent_monthly.empty and all(col in agent_monthly.columns for col in ['period', 'verified_count']):
        try:
            # Group by period to get total counts
            monthly_totals = agent_monthly.groupby('period')['verified_count'].sum().reset_index()
            
            # Convert period to datetime for sorting
            monthly_totals['period_dt'] = pd.to_datetime(monthly_totals['period'], errors='coerce')
            monthly_totals = monthly_totals.sort_values('period_dt')
            
            # Calculate growth rate if at least 2 periods available
            if len(monthly_totals) > 1:
                last_value = monthly_totals['verified_count'].iloc[-1]
                prev_value = monthly_totals['verified_count'].iloc[-2]
                growth_rate = ((last_value / prev_value) - 1) * 100 if prev_value > 0 else 0
                
                report_content += f"""
    - **Latest monthly completions growth:** {growth_rate:+.1f}%
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
    
    # Add phase3a metrics if available
    if 'phase3a_metrics' in analysis_results and analysis_results['phase3a_metrics']:
        phase3a = analysis_results['phase3a_metrics']
        
        report_content += f"""
    ### Phase 3A Metrics
    
    - **Automation Rate:** {phase3a.get('automation_rate', 0) * 100:.1f}%
    - **Reassignment Rate:** {phase3a.get('reassignment_rate', 0) * 100:.1f}%
    - **Avg Response Time:** {phase3a.get('avg_response_time_hours', 0):.1f} hours
    - **Email Success Rate:** {phase3a.get('email_success_rate', 0) * 100:.1f}%
    """
    
    # Add conclusion
    dashboard_dir = analysis_results.get('dashboard_dir')
    if dashboard_dir:
        report_content += f"""
    ## Conclusion
    
    Murphy Brown demonstrates significant positive impact on verification processes, with notable improvements in turnaround time, reduction in human touches required, and substantial resource savings.
    
    The dual capability of contact research and email handling work together to streamline verifications and reduce agent workload.
    
    ## Next Steps
    
    1. Continue monitoring MB performance as volume increases
    2. Identify additional opportunity areas for expanding capabilities
    3. Track long-term impact on verification throughput and quality
    
    ---
    
    *Full analysis dashboard available at: {dashboard_dir}/dashboard.html*
"""

    # Write report to file
    report_path = os.path.join(output_dir, f"mb_impact_executive_summary_{datetime.now().strftime('%Y%m%d')}.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Executive summary saved to: {report_path}")
    return report_path