#!/usr/bin/env python3
import os
import numpy as np
import subprocess
import pandas as pd
import sys; sys.path.insert(0, '..')
from load_data import load_data
import time
import re
import json

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def extract_accuracy_from_output(output):
    """Extract accuracy percentage from analysis output"""
    # Try to find the accuracy line with regex
    accuracy_match = re.search(r'Attack Accuracy: (\d+\.?\d*)%', output)
    if accuracy_match:
        return float(accuracy_match.group(1))
    
    # Try to find "n_users_evaluated" in the output
    users_match = re.search(r'Number of users evaluated: (\d+)', output)
    n_users = int(users_match.group(1)) if users_match else None
    
    # If regex fails, try parsing line by line
    for line in output.split('\n'):
        if 'Attack Accuracy:' in line:
            try:
                return float(line.split('%')[0].split(':')[1].strip())
            except:
                pass
    
    return "Failed to extract accuracy", n_users

def run_multiple_category_attacks_all_users(data_name, secret_bit, reps=32, n_procs=4, data_dir="results", 
                                           generate=False, process=False, run_attack=False, analyze=True):
    """
    Run separate reconstruction attacks for each category of a multi-valued attribute,
    evaluating against ALL users in the dataset.

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
    generate: bool
        Whether to generate queries
    process: bool
        Whether to process queries
    run_attack: bool
        Whether to run the attack
    analyze: bool
        Whether to analyze results
    """
    # Load the dataset once to get all categories
    df = load_data(data_name, 1000, secret_bit, randomize=False, balance=False)
    categories = sorted(df[secret_bit].unique())

    print(f"Found {len(categories)} unique values in column {secret_bit}: {categories}")
    results = {}
    
    for category in categories:
        category_str = str(category).replace(" ", "_").replace("/", "_")
        print(f"\n[{time.strftime('%Y-%m-%d_%H:%M:%S')}] Running attack for category: {category}")

        # Create a category-specific dataset name to avoid path conflicts
        category_data_name = f"{data_name}_{category_str}"

        # Create directory for this category
        category_dir = f"{data_dir}/{category_data_name}"
        os.makedirs(f"{category_dir}/reps", exist_ok=True)

        # Prepare datasets with binary indicator for this category
        print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S')}] Preparing dataset...")
        for rep in range(reps):
            rep_dir = f"{category_dir}/reps/rep_{rep}"
            os.makedirs(rep_dir, exist_ok=True)

            # Copy the dataset with binary encoding for this category
            df_rep = load_data(data_name, 1000, secret_bit, randomize=True)
            df_rep[secret_bit] = (df_rep[secret_bit] == category).astype(int)
            df_rep.to_csv(f"{rep_dir}/df.csv.gz", compression="gzip", index=False)
            
            # We still need a user.csv file for compatibility with some scripts,
            # but we'll ignore it in our all-users analysis
            user = 0  # This value won't be used in all-users analysis
            np.savetxt(f"{rep_dir}/user.csv", np.array([user]), fmt="%d")

        # Generate queries
        if generate:
            print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S')}] Generating queries...")
            gen_queries_cmd = [
                "python3", "gen_queries.py",
                "--data_name", category_data_name,
                "--query_type", "simple",
                "--k", "3",
                "--reps", str(reps),
                "--n_procs", str(n_procs),
                "--data_dir", data_dir,
                "--secret_bit", secret_bit
            ]
            subprocess.run(gen_queries_cmd, check=True)

        # Process queries
        if process:
            print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S')}] Processing queries...")
            process_queries_cmd = [
                "python3", "process_queries.py",
                "--data_name", category_data_name,
                "--synth_model", "NonPrivate",
                "--n_rows", "1000",
                "--query_type", "simple",
                "--k", "3",
                "--scale_type", "cond",
                "--reps", str(reps),
                "--n_procs", str(n_procs),
                "--data_dir", data_dir,
                "--secret_bit", secret_bit
            ]
            subprocess.run(process_queries_cmd, check=True)

        # Run the attack
        if run_attack:
            print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S')}] Running attack...")
            run_attack_cmd = [
                "python3", "run_attack.py",
                "--data_name", category_data_name,
                "--synth_model", "NonPrivate",
                "--n_rows", "1000",
                "--scale_type", "cond",
                "--attack_name", "recon",
                "--reps", str(reps),
                "--n_procs", str(n_procs),
                "--data_dir", data_dir,
                "--k", "3",
                "--secret_bit", secret_bit
            ]
            subprocess.run(run_attack_cmd, check=True)

        # Analyze the results
        if analyze:
            print(f"[{time.strftime('%Y-%m-%d_%H:%M:%S')}] Analyzing results (all users)...")
            
            # Custom implementation to evaluate all users
            all_users_results = evaluate_all_users(
                category_data_name, 
                "NonPrivate", 
                1000, 
                3, 
                "cond", 
                -1, 
                "recon", 
                secret_bit, 
                reps, 
                data_dir
            )
            
            accuracy = all_users_results["acc"] * 100 if "acc" in all_users_results else "Failed to calculate"
            n_users = all_users_results.get("n_users_evaluated", 0)
            
            # Convert to standard Python types to avoid JSON serialization issues
            results[str(category)] = {
                "accuracy": float(accuracy) if isinstance(accuracy, (int, float, np.number)) else accuracy,
                "n_users_evaluated": int(n_users) if isinstance(n_users, (int, np.integer)) else n_users
            }
            
            print(f"All-users analysis for category {category}: {accuracy}% accuracy over {n_users} users")
            
    # Print a summary of all category results
    print("\n===== SUMMARY OF RESULTS (ALL USERS) =====")
    for category, result in results.items():
        print(f"Category {category}: {result['accuracy']}% accuracy ({result['n_users_evaluated']} users evaluated)")

    # Save results to JSON file
    results_file = f"{data_dir}/{data_name}_{secret_bit}_all_users_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {results_file}")
    
    return results

def evaluate_all_users(data_name, synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_bit, reps, data_dir):
    """Custom implementation to evaluate attack accuracy on all users"""
    
    data_dir_path = f'{data_dir}/{data_name}/reps'
    
    all_exp_bits = []
    all_est_bits = []
    
    # Process each repetition
    for rep in range(reps):
        rep_dir = f'{data_dir_path}/rep_{rep}'
        
        # Check if files exist
        if not os.path.exists(rep_dir):
            print(f"Directory {rep_dir} does not exist")
            continue
            
        df_path = f'{rep_dir}/df.csv.gz'
        if not os.path.exists(df_path):
            print(f"File {df_path} does not exist")
            continue
            
        # Load the dataset
        df = pd.read_csv(df_path, compression='gzip')
        
        # Get paths to results
        query_subdir = f'simple/{k}way_{scale_type}'
        synth_df_dir = f'{rep_dir}/{synth_model}/{n_rows_names.get(n_rows, "1K")}/'
        synth_result_dir = f'{synth_df_dir}/{query_subdir}/'
        
        synth_result_path = f'{synth_result_dir}/est_secret_bits_{n_queries}_{attack_name}.npz'
        synth_score_path = f'{synth_result_dir}/sol_{n_queries}_{attack_name}.npz'
        
        if not os.path.exists(synth_result_path) or not os.path.exists(synth_score_path):
            print(f"Missing result files in {synth_result_dir}")
            continue
            
        # Get expected and estimated secret bits
        try:
            exp_secret_bits = df[secret_bit].values
            est_secret_bits = np.load(synth_result_path)['arr_0']
            
            # Check if sizes match
            if len(exp_secret_bits) != len(est_secret_bits):
                print(f"Size mismatch in rep {rep}: exp={len(exp_secret_bits)}, est={len(est_secret_bits)}")
                continue
                
            # Add to our collection
            all_exp_bits.extend(exp_secret_bits)
            all_est_bits.extend(est_secret_bits)
            
        except Exception as e:
            print(f"Error processing rep {rep}: {e}")
            continue
    
    # Calculate accuracy
    if not all_exp_bits:
        return {"error": "No valid data found"}
        
    all_exp_bits = np.array(all_exp_bits)
    all_est_bits = np.array(all_est_bits)
    
    successes = (all_exp_bits == all_est_bits).sum()
    total = len(all_exp_bits)
    
    return {
        "acc": float(successes / total) if total > 0 else 0,
        "n_users_evaluated": int(total)
    }

# Define n_rows_names dictionary here for standalone functionality
n_rows_names = {
    10: '10',
    100: '100',
    1000: '1K',
    10000: '10K',
    100000: '100K',
    1000000: '1M'
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run reconstruction attacks for each category of a multi-valued attribute (ALL USERS)")
    parser.add_argument("--data_name", type=str, default="nist", help="Dataset name")
    parser.add_argument("--secret_bit", type=str, default="F1", help="Column with multiple categories to attack")
    parser.add_argument("--reps", type=int, default=32, help="Number of repetitions")
    parser.add_argument("--n_procs", type=int, default=4, help="Number of processors to use")
    parser.add_argument("--data_dir", type=str, default="results", help="Directory for results")
    parser.add_argument("--generate", action="store_true", help="Run n-way queries")
    parser.add_argument("--process", action="store_true", help="Run processing queries")
    parser.add_argument("--run_attack", action="store_true", help="Run attack")
    parser.add_argument("--analyze", action="store_true", help="Run analyze")

    args = parser.parse_args()
    run_multiple_category_attacks_all_users(
        args.data_name, 
        args.secret_bit, 
        args.reps, 
        args.n_procs, 
        args.data_dir,
        args.generate,
        args.process,
        args.run_attack,
        args.analyze
    )
