#!/usr/bin/env python3
"""
ChatISA Log Analysis Tool
Demonstrates the differences between usage_log.json and session_log.json
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def analyze_log_differences():
    """Analyze the new privacy-compliant activity log."""
    
    logs_dir = Path("logs")
    activity_file = logs_dir / "activity_log.json"
    backup_usage_file = logs_dir / "usage_log.json.backup"
    backup_session_file = logs_dir / "session_log.json.backup"
    
    print("="*80)
    print("         ChatISA Log Analysis - Privacy-Compliant System")
    print("="*80)
    
    # Check if new activity file exists
    if not activity_file.exists():
        print("No activity_log.json found")
        return
        
    print("\nNOTE: ChatISA now uses privacy-compliant logging!")
    print("Personal information has been removed and logs consolidated.")
    if backup_usage_file.exists():
        print("Legacy logs backed up as *.backup files")
    
    # Load data
    with open(activity_file) as f:
        activity_data = json.load(f)
    
    print(f"\nFILE OVERVIEW:")
    print(f"  activity_log.json: {len(activity_data)} entries")
    
    # Separate activities by type
    ai_activities = [a for a in activity_data if a.get('activity_type') == 'ai_generation']
    session_activities = [a for a in activity_data if a.get('activity_type') != 'ai_generation']
    
    print(f"  - AI interactions: {len(ai_activities)}")
    print(f"  - User activities: {len(session_activities)}")
    
    if ai_activities:
        ai_df = pd.DataFrame(ai_activities)
        print(f"\nAI INTERACTION TRACKING:")
        print(f"  Purpose: Tracks actual AI model usage and content generation")
        print(f"  Total Cost Tracked: ${ai_df['cost'].sum():.4f}")
        print(f"  Models Used: {ai_df['model_used'].nunique()} different models")
        print(f"  Pages with AI Usage: {', '.join(ai_df['page'].unique())}")
        
        # Calculate average response time safely
        response_times = []
        for perf in ai_df['performance']:
            if isinstance(perf, dict) and 'response_time_ms' in perf:
                response_times.append(perf['response_time_ms'])
        avg_response = sum(response_times) / len(response_times) if response_times else 0
        print(f"  Avg Response Time: {avg_response:.0f}ms")
        
        # Token analysis
        total_tokens = sum([t['total'] for t in ai_df['tokens'] if isinstance(t, dict)])
        print(f"  Total Tokens Used: {total_tokens:,}")
        
        # Cost breakdown by model
        print(f"\n  Cost by Model:")
        model_costs = ai_df.groupby('model_used')['cost'].sum().sort_values(ascending=False)
        for model, cost in model_costs.head(5).items():
            print(f"    * {model}: ${cost:.4f}")
    
    if session_activities:
        session_df = pd.DataFrame(session_activities)
        print(f"\nUSER ACTIVITY TRACKING:")
        print(f"  Purpose: Tracks navigation, features, and system interactions")
        print(f"  Unique Sessions: {session_df['session_id'].nunique()}")
        print(f"  Activity Types: {', '.join(session_df['activity_type'].unique())}")
        
        # Page visit analysis
        page_visits = session_df[session_df['activity_type'] == 'page_visit']
        if not page_visits.empty:
            print(f"  Page Visits: {len(page_visits)} total")
            print(f"\n  Most Visited Pages:")
            visit_counts = page_visits['page'].value_counts()
            for page, count in visit_counts.head(5).items():
                print(f"    * {page}: {count} visits")
        
        # Model changes
        model_changes = session_df[session_df['activity_type'] == 'model_change']
        if not model_changes.empty:
            print(f"  Model Changes: {len(model_changes)} switches")
        
        # PDF exports
        pdf_exports = session_df[session_df['activity_type'] == 'pdf_export']
        if not pdf_exports.empty:
            print(f"  PDF Exports: {len(pdf_exports)} downloads (PRIVACY-COMPLIANT - no personal data)")
    
    print(f"\nKEY FEATURES OF NEW PRIVACY-COMPLIANT SYSTEM:")
    print(f"  activity_log.json:")
    print(f"    + Tracks BOTH AI usage AND user interactions in one file")
    print(f"    + Records costs, tokens, and performance metrics")
    print(f"    + NO personal information (names, emails, etc.) logged")
    print(f"    + Session IDs allow analytics without privacy invasion")
    print(f"    + Supports all previous analysis capabilities")
    print(f"    + Critical for cost management and UX analysis")
    
    print(f"\nANALYSIS RECOMMENDATIONS:")
    print(f"  For Educators: Monitor AI usage and engagement patterns")
    print(f"    * AI costs and usage by course/topic (no personal data)")
    print(f"    * Learning pathways and feature adoption")
    print(f"    * Response times and model performance")
    print(f"  ")
    print(f"  For Researchers: Privacy-compliant learning analytics")
    print(f"    * Anonymous usage patterns across different AI models")
    print(f"    * Navigation patterns and feature adoption")
    print(f"    * Performance studies without personal identifiers")
    print(f"  ")
    print(f"  For System Admins: Operational health monitoring")
    print(f"    * API costs, model performance, errors")
    print(f"    * User load, feature usage, peak times")
    print(f"    * All without storing personal information")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_log_differences()