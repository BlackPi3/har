#!/usr/bin/env python3
"""
Analyze hyperparameter search results

Usage:
    python analyze_hyperparameter_results.py outputs/hyperparameter_search/coarse_search_12345/
    python analyze_hyperparameter_results.py outputs/hyperparameter_search/fine_search_67890/ --plot
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_dir):
    """Load all trial results from directory."""
    results_dir = Path(results_dir)
    results = []
    
    for result_file in results_dir.glob("trial_*_results.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
    
    return results


def analyze_results(results):
    """Analyze hyperparameter search results."""
    if not results:
        print("No results found!")
        return None
    
    # Convert to DataFrame for analysis
    rows = []
    for result in results:
        row = {
            'trial_id': result['trial_id'],
            'val_loss': result['metrics']['val_loss'],
            'train_time': result['metrics']['train_time_seconds'],
            'best_epoch': result['metrics']['best_epoch'],
            'converged': result['metrics']['converged']
        }
        
        # Add hyperparameters
        for key, value in result['hyperparameters'].items():
            row[f'hp_{key}'] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Analysis
    print("=== Hyperparameter Search Results Analysis ===")
    print(f"Total trials: {len(df)}")
    print(f"Successful trials: {len(df[df['val_loss'] != float('inf')])}")
    print(f"Failed trials: {len(df[df['val_loss'] == float('inf')])}")
    print()
    
    # Best result
    if len(df[df['val_loss'] != float('inf')]) > 0:
        valid_df = df[df['val_loss'] != float('inf')]
        best_idx = valid_df['val_loss'].idxmin()
        best_result = valid_df.loc[best_idx]
        
        print("=== Best Result ===")
        print(f"Trial ID: {best_result['trial_id']}")
        print(f"Validation Loss: {best_result['val_loss']:.4f}")
        print(f"Training Time: {best_result['train_time']:.1f}s")
        print(f"Best Epoch: {best_result['best_epoch']}")
        print("Hyperparameters:")
        for col in df.columns:
            if col.startswith('hp_'):
                param_name = col[3:]  # Remove 'hp_' prefix
                print(f"  {param_name}: {best_result[col]}")
        print()
        
        # Top 5 results
        print("=== Top 5 Results ===")
        top_5 = valid_df.nsmallest(5, 'val_loss')[['trial_id', 'val_loss', 'train_time'] + 
                                                   [col for col in df.columns if col.startswith('hp_')]]
        print(top_5.to_string(index=False))
        print()
        
        # Parameter importance (correlation with validation loss)
        print("=== Parameter Importance (Correlation with Val Loss) ===")
        hp_cols = [col for col in df.columns if col.startswith('hp_')]
        correlations = []
        
        for col in hp_cols:
            if df[col].dtype in ['int64', 'float64']:
                corr = valid_df[col].corr(valid_df['val_loss'])
                correlations.append((col[3:], abs(corr), corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        for param, abs_corr, corr in correlations:
            direction = "↑" if corr > 0 else "↓"
            print(f"  {param}: {abs_corr:.3f} {direction}")
        print()
        
        return df, best_result
    
    else:
        print("No successful trials found!")
        return df, None


def plot_results(df, output_dir):
    """Create visualization plots."""
    if df is None or len(df) == 0:
        return
    
    valid_df = df[df['val_loss'] != float('inf')]
    if len(valid_df) == 0:
        print("No valid results to plot")
        return
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Validation loss distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.hist(valid_df['val_loss'], bins=20, alpha=0.7)
    plt.xlabel('Validation Loss')
    plt.ylabel('Frequency')
    plt.title('Distribution of Validation Loss')
    
    # 2. Training time vs validation loss
    plt.subplot(2, 3, 2)
    plt.scatter(valid_df['train_time'], valid_df['val_loss'], alpha=0.6)
    plt.xlabel('Training Time (s)')
    plt.ylabel('Validation Loss')
    plt.title('Training Time vs Val Loss')
    
    # 3. Convergence analysis
    plt.subplot(2, 3, 3)
    converged_counts = valid_df['converged'].value_counts()
    plt.pie(converged_counts.values, labels=['Not Converged', 'Converged'], autopct='%1.1f%%')
    plt.title('Convergence Rate')
    
    # 4-6. Hyperparameter vs validation loss (for numeric parameters)
    hp_cols = [col for col in df.columns if col.startswith('hp_')]
    numeric_hp_cols = [col for col in hp_cols if valid_df[col].dtype in ['int64', 'float64']]
    
    for i, col in enumerate(numeric_hp_cols[:3]):  # Plot first 3 numeric parameters
        plt.subplot(2, 3, 4 + i)
        plt.scatter(valid_df[col], valid_df['val_loss'], alpha=0.6)
        plt.xlabel(col[3:])  # Remove 'hp_' prefix
        plt.ylabel('Validation Loss')
        plt.title(f'{col[3:]} vs Val Loss')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'hyperparameter_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    
    # Heatmap for parameter combinations (if we have 2+ numeric parameters)
    if len(numeric_hp_cols) >= 2:
        plt.figure(figsize=(10, 8))
        
        # Create pivot table for heatmap
        param1, param2 = numeric_hp_cols[0], numeric_hp_cols[1]
        
        # Group by parameter combinations and take mean validation loss
        pivot_data = valid_df.groupby([param1, param2])['val_loss'].mean().reset_index()
        pivot_table = pivot_data.pivot(index=param1, columns=param2, values='val_loss')
        
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis_r')
        plt.title(f'Validation Loss Heatmap: {param1[3:]} vs {param2[3:]}')
        
        heatmap_path = Path(output_dir) / 'parameter_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {heatmap_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('results_dir', help='Directory containing trial results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-csv', help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Load and analyze results
    results = load_results(args.results_dir)
    df, best_result = analyze_results(results)
    
    if df is not None:
        # Save CSV if requested
        if args.save_csv:
            df.to_csv(args.save_csv, index=False)
            print(f"Results saved to: {args.save_csv}")
        
        # Generate plots if requested
        if args.plot:
            plot_results(df, args.results_dir)
        
        # Save best configuration
        if best_result is not None:
            best_config = {
                'val_loss': best_result['val_loss'],
                'hyperparameters': {col[3:]: best_result[col] for col in df.columns if col.startswith('hp_')}
            }
            
            best_config_path = Path(args.results_dir) / 'best_hyperparameters.json'
            with open(best_config_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            print(f"Best hyperparameters saved to: {best_config_path}")


if __name__ == "__main__":
    main()