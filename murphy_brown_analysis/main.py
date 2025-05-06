#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - Main script for Murphy Brown impact analysis

This script ties together the metrics, analysis, and viz modules
to provide a complete pipeline for Murphy Brown impact analysis.
"""

import os
import sys
import argparse
import traceback
from datetime import datetime

# Import modules
import metrics
import analysis
import viz


def save_dataframe(df, name, output_dir):
    """Save a DataFrame to a CSV file in the output directory."""
    if df is None or df.empty:
        print(f"Skipping CSV save for '{name}' as the DataFrame is empty or None.")
        return None
    
    current_time = datetime.now().strftime("%Y%m%d")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_{current_time}.csv")
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Saved CSV: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving CSV file to {output_path}: {e}")
        traceback.print_exc()
        return None


def save_json(data, name, output_dir):
    """Save a dictionary to a JSON file in the output directory."""
    import json
    
    if not isinstance(data, dict):
        print(f"Skipping JSON save for '{name}' as input is not a dictionary.")
        return None
    
    current_time = datetime.now().strftime("%Y%m%d")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_{current_time}.json")
    
    try:
        # Convert numpy types to native Python types for JSON serialization
        serializable_dict = {k: (v.item() if hasattr(v, 'item') else v) for k, v in data.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable_dict, f, indent=4)
        print(f"Saved JSON: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving JSON file to {output_path}: {e}")
        traceback.print_exc()
        return None


def get_output_dir(output_dir_path, subdirectory=None):
    """Get output directory, creating it if necessary."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    base_output_dir = output_dir_path if output_dir_path else os.path.join(script_dir, "Output")
    
    if subdirectory:
        output_dir = os.path.join(base_output_dir, subdirectory)
    else:
        output_dir = base_output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results(analysis_results, output_dir):
    """Save analysis results to CSV and JSON files."""
    print("\n--- Saving Analysis Results ---")
    
    # Create output subdirectories
    data_dir = os.path.join(output_dir, "data")
    reports_dir = os.path.join(output_dir, "reports")
    verdicts_dir = os.path.join(output_dir, "verdicts")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(verdicts_dir, exist_ok=True)
    
    # Save main summary DataFrame
    if 'df_summary' in analysis_results:
        save_dataframe(analysis_results['df_summary'], "mb_analysis_summary", data_dir)
    
    # Save time efficiency and TAT stats
    if 'time_efficiency' in analysis_results and analysis_results['time_efficiency']:
        if 'tat_stats' in analysis_results['time_efficiency']:
            save_dataframe(analysis_results['time_efficiency']['tat_stats'], "tat_stats", data_dir)
        
        if 'statistical_tests' in analysis_results['time_efficiency'] and analysis_results['time_efficiency']['statistical_tests']:
            try:
                import pandas as pd
                ttest_df = pd.DataFrame.from_dict(
                    analysis_results['time_efficiency']['statistical_tests'], 
                    orient='index'
                )
                ttest_df.index.name = 'Metric'
                save_dataframe(ttest_df.reset_index(), "mb_time_efficiency_ttests", data_dir)
            except Exception as e:
                print(f"Error saving time efficiency t-tests: {e}")
    
    # Save contact efficiency stats
    if 'contact_efficiency' in analysis_results and analysis_results['contact_efficiency']:
        if 'attempts_stats' in analysis_results['contact_efficiency']:
            save_dataframe(analysis_results['contact_efficiency']['attempts_stats'], "attempts_stats", data_dir)
        
        if 'contact_stats' in analysis_results['contact_efficiency']:
            save_dataframe(analysis_results['contact_efficiency']['contact_stats'], "contact_stats", data_dir)
        
        if 'yield_stats' in analysis_results['contact_efficiency']:
            save_dataframe(analysis_results['contact_efficiency']['yield_stats'], "email_yield_stats", data_dir)
        
        if 'statistical_tests' in analysis_results['contact_efficiency'] and analysis_results['contact_efficiency']['statistical_tests']:
            try:
                import pandas as pd
                ttest_df = pd.DataFrame.from_dict(
                    analysis_results['contact_efficiency']['statistical_tests'], 
                    orient='index'
                )
                ttest_df.index.name = 'Metric'
                save_dataframe(ttest_df.reset_index(), "mb_contact_efficiency_ttests", data_dir)
            except Exception as e:
                print(f"Error saving contact efficiency t-tests: {e}")
    
    # Save capability impact stats
    if 'capability_impact' in analysis_results and analysis_results['capability_impact']:
        if 'grouped_metrics' in analysis_results['capability_impact']:
            save_dataframe(analysis_results['capability_impact']['grouped_metrics'], "mb_capability_metrics", data_dir)
        
        if 'anova_results' in analysis_results['capability_impact']:
            save_dataframe(analysis_results['capability_impact']['anova_results'], "mb_capability_anova", data_dir)
        
        if 'group_counts' in analysis_results['capability_impact']:
            save_dataframe(analysis_results['capability_impact']['group_counts'], "mb_capability_counts", data_dir)

    # Save capability_stats (from summarize_time_efficiency_by_capability)
    if 'capability_stats' in analysis_results and analysis_results['capability_stats'] is not None:
        save_dataframe(analysis_results['capability_stats'], "capability_tat_stats", data_dir)

    # Save queue depth
    if 'queue_depth_weekly' in analysis_results and analysis_results['queue_depth_weekly'] is not None:
        save_dataframe(analysis_results['queue_depth_weekly'], "queue_depth_weekly", data_dir)
    
    # Save mb_category_metrics results
    if 'mb_category_metrics' in analysis_results and analysis_results['mb_category_metrics']:
        cat = analysis_results['mb_category_metrics']
        if isinstance(cat, dict):
            if 'category_counts' in cat and cat['category_counts'] is not None:
                save_dataframe(cat['category_counts'], "mb_category_counts", data_dir)
            if 'category_metrics' in cat and cat['category_metrics'] is not None:
                save_dataframe(cat['category_metrics'], "mb_category_rollup", data_dir)
            if 'anova_results' in cat and cat['anova_results'] is not None:
                save_dataframe(cat['anova_results'], "mb_category_anova", data_dir)
        else:
             print("Warning: mb_category_metrics is not a dictionary. Skipping save.")

    # Save contact plan results
    if 'plan_delta' in analysis_results and analysis_results['plan_delta'] is not None:
        save_dataframe(analysis_results['plan_delta'], "mb_contact_plan_delta", data_dir)
    
    # Save agent uplift results
    if 'agent_uplift' in analysis_results and analysis_results['agent_uplift'] is not None:
        save_dataframe(analysis_results['agent_uplift'], "agent_completion_uplift", data_dir)
    
    # Save SLA rate results
    if 'sla_rate' in analysis_results and analysis_results['sla_rate'] is not None:
        save_dataframe(analysis_results['sla_rate'], "first_day_sla_rate", data_dir)
    
    # Save throughput metrics
    if 'weekly_throughput' in analysis_results and analysis_results['weekly_throughput'] is not None:
        save_dataframe(analysis_results['weekly_throughput'], "verified_by_agent_week", data_dir)
    
    if 'monthly_throughput' in analysis_results and analysis_results['monthly_throughput'] is not None:
        save_dataframe(analysis_results['monthly_throughput'], "verified_by_agent_month", data_dir)
    
    if 'daily_throughput' in analysis_results and analysis_results['daily_throughput'] is not None:
        save_dataframe(analysis_results['daily_throughput'], "verified_by_agent_day", data_dir)
    
    # Save verdicts as JSON
    if 'email_verdict' in analysis_results and analysis_results['email_verdict'] is not None:
        save_json(analysis_results['email_verdict'], "mb_email_performance_verdict", verdicts_dir)
    
    if 'contact_plan_verdict' in analysis_results and analysis_results['contact_plan_verdict'] is not None:
        save_json(analysis_results['contact_plan_verdict'], "mb_contact_plan_verdict", verdicts_dir)
    
    # Save phase3a metrics if available
    if 'phase3a_metrics' in analysis_results and analysis_results['phase3a_metrics'] is not None:
        # Directly save the metrics dictionary without looking for a nested 'metrics' key
        if isinstance(analysis_results['phase3a_metrics'], dict) and analysis_results['phase3a_metrics']:
            save_json(analysis_results['phase3a_metrics'], "phase3a_metrics", data_dir)
        else:
            print("Warning: 'phase3a_metrics' data is empty or not a dictionary. Skipping save.")
            
    print("All results saved successfully.")


def main():
    """Main execution function."""
    # --- Configuration Start ---
    # Define parameters directly instead of using argparse
    input_file = r"C:\Users\awrig\Downloads\Fullo3Query.csv"  # Specify your input CSV path here
    output_dir_param = None  # Set to a specific path or None to use default './Output'
    filter_searchtype = 'empv,eduv' # Default filter
    mb_agent = 'murphy.brown' # Default MB agent
    skip_dashboard = False # Set to True to skip dashboard
    skip_summary = False # Set to True to skip summary
    # --- Configuration End ---

    print(f"\n=== Murphy Brown Impact Analysis ===")
    print(f"Input file: {input_file}") # Use variable
    print(f"Output directory: {output_dir_param or './Output'}") # Use variable
    print(f"Search type filter: {filter_searchtype}") # Use variable
    print(f"MB agent identifier: {mb_agent}\n") # Use variable
    
    start_time = datetime.now()
    
    try:
        # 1. Load history data
        print("Step 1: Loading history data...")
        import pandas as pd
        
        try:
            # Use the input_file variable directly
            df_history = pd.read_csv(input_file, low_memory=False)
            print(f"Loaded {len(df_history)} rows from {input_file}")
            
            # Standardize column names to lowercase
            df_history.columns = df_history.columns.str.lower().str.strip()
            
            # Basic data validation
            if 'searchid' not in df_history.columns:
                raise ValueError("Input file missing required 'searchid' column")
                
        except Exception as e:
            print(f"Error loading input file: {e}")
            traceback.print_exc()
            return 1
        
        # 2. Filter search types if specified
        if filter_searchtype: # Use variable
            search_types = [st.strip().lower() for st in filter_searchtype.split(',')] # Use variable
            print(f"Filtering for search types: {search_types}")
            
            original_count = len(df_history)
            if 'searchtype' in df_history.columns:
                search_type_filter = df_history['searchtype'].astype(str).str.lower().isin(search_types)
                df_history = df_history.loc[search_type_filter].copy()
                print(f"Filtered for {search_types} search types. Kept {len(df_history)} rows out of {original_count}.")
                
                if df_history.empty:
                    print(f"No matching records found after filtering. Exiting.")
                    return 1
            else:
                print("Warning: 'searchtype' column not found. Cannot filter by search type.")
        
        # 4. Calculate all metrics
        print("\nStep 2: Calculating all metrics...")
        all_metrics = metrics.calculate_all_metrics(df_history)
        if not all_metrics:
            print("Error: Metrics calculation failed.")
            return 1
        
        # 5. Perform analysis
        print("\nStep 3: Performing analysis...")
        analysis_results = analysis.perform_full_analysis(all_metrics, df_history)
        if not analysis_results:
            print("Error: Analysis failed.")
            return 1
        
        # 6. Save results
        output_dir = get_output_dir(output_dir_param) # Pass variable to updated function
        save_results(analysis_results, output_dir)
        
        # 7. Create dashboard
        if not skip_dashboard: # Use variable
            print("\nStep 4: Creating visualizations and dashboard...")
            dashboard_dir = viz.create_dashboard(analysis_results, 
                                                 os.path.join(output_dir, "dashboard"))
            # Safely add dashboard_dir to results
            if analysis_results is not None:
                analysis_results['dashboard_dir'] = dashboard_dir
            print(f"Dashboard created at: {dashboard_dir}")
        else:
            print("\nSkipping dashboard creation as requested.")
        
        # 8. Generate executive summary
        if not skip_summary: # Use variable
            print("\nStep 5: Generating executive summary...")
            summary_path = viz.generate_executive_summary(analysis_results, 
                                                         os.path.join(output_dir, "reports"))
            print(f"Executive summary created at: {summary_path}")
        else:
            print("\nSkipping executive summary creation as requested.")
        
        # Done!
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        print(f"\n=== Analysis Complete! ===")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())