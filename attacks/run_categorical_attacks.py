#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import pandas as pd
import sys; sys.path.insert(0, '..')
from load_data import load_data

def run_multiple_category_attacks(data_name, secret_bit, reps=32, n_procs=4, data_dir="results/"):
    """
    Run separate reconstruction attacks for each category of a multi-valued attribute.
    
    Parameters
    ----------
    data_name: str
        Dataset name
    secret_bit: str
        Column with multiple categories to attack
    reps: int
        Number of repetitions
    n_procs: int
        Number of processors to use
    data_dir: str
        Directory for results
    """
    # Load the dataset once to get all categories
    df = load_data(data_name, 1000, secret_bit, randomize=False)
    categories = df[secret_bit].unique()
    
    results = {}
    
    for category in categories:
        print(f"Running attack for category: {category}")
        
        # Create directory for this category
        category_dir = f"{data_dir}/{data_name}_{category}"
        os.makedirs(f"{category_dir}/reps", exist_ok=True)
        
        # Prepare datasets with binary indicator for this category
        for rep in range(reps):
            rep_dir = f"{category_dir}/reps/rep_{rep}"
            os.makedirs(rep_dir, exist_ok=True)
            
            # Copy the dataset with binary encoding for this category
            df_rep = load_data(data_name, 1000, secret_bit, randomize=True)
            df_rep[secret_bit] = (df_rep[secret_bit] == category).astype(int)
            df_rep.to_csv(f"{rep_dir}/df.csv.gz", compression="gzip", index=False)
            
            # Select a random user
            user = np.random.randint(len(df_rep))
            np.savetxt(f"{rep_dir}/user.csv", np.array([user]), fmt="%d")
        
        # Run the attack pipeline for this category
        subprocess.run([
            "python3", "gen_queries.py", 
            "--data_name", f"{data_name}_{category}", 
            "--query_type", "simple", 
            "--k", "3", 
            "--reps", str(reps), 
            "--n_procs", str(n_procs), 
            "--data_dir", data_dir, 
            "--secret_bit", secret_bit
        ])
        
        subprocess.run([
            "python3", "run_attack.py", 
            "--data_name", f"{data_name}_{category}", 
            "--synth_model", "NonPrivate", 
            "--n_rows", "1000", 
            "--scale_type", "cond", 
            "--attack_name", "recon", 
            "--reps", str(reps), 
            "--n_procs", str(n_procs), 
            "--data_dir", data_dir, 
            "--k", "3", 
            "--secret_bit", secret_bit
        ])
        
        # Analyze the results for this category
        analyze_proc = subprocess.run([
            "python3", "analyze_privacy_utility.py", 
            "--data_name", f"{data_name}_{category}", 
            "--synth_model", "NonPrivate", 
            "--n_rows", "1000", 
            "--k", "3", 
            "--scale_type", "cond", 
            "--n_queries", "-1", 
            "--attack_name", "recon", 
            "--secret_bit", secret_bit, 
            "--reps", str(reps), 
            "--data_dir", data_dir
        ], capture_output=True, text=True)
        
        print(analyze_proc.stdout)
        results[category] = analyze_proc.stdout
    
    # Print a summary of all category results
    print("\n===== SUMMARY OF RESULTS =====")
    for category, result in results.items():
        print(f"Category {category}: {result}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run reconstruction attacks for each category of a multi-valued attribute")
    parser.add_argument("--data_name", type=str, default="nist", help="Dataset name")
    parser.add_argument("--secret_bit", type=str, default="F1", help="Column with multiple categories to attack")
    parser.add_argument("--reps", type=int, default=32, help="Number of repetitions")
    parser.add_argument("--n_procs", type=int, default=4, help="Number of processors to use")
    parser.add_argument("--data_dir", type=str, default="results/", help="Directory for results")
    
    args = parser.parse_args()
    run_multiple_category_attacks(args.data_name, args.secret_bit, args.reps, args.n_procs, args.data_dir)
