#!/bin/bash
REPS=32
N_PROCS=4
DATA_DIR=results/
DATA_NAME=nist
SECRET_BIT=F1  # Your target column from NIST dataset

echo "[$(date +%F_%T)] Setting up NIST dataset for attack..."

# Create a Python script to prepare the dataset and domain file
cat > prepare_nist.py << 'EOF'
import pandas as pd
import numpy as np
import json
import os

def prepare_binary_secret_bit(df, secret_bit):
    """Ensure the secret attribute is binary (0/1)."""
    unique_vals = df[secret_bit].unique()
    
    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
        return df
    
    # One-hot encode the column
    first_category = unique_vals[0]
    new_col_name = f"{secret_bit}_binary"
    df[new_col_name] = (df[secret_bit] == first_category).astype(int)
    
    # Drop the original column and rename the new one
    df = df.drop(columns=[secret_bit])
    df = df.rename(columns={new_col_name: secret_bit})
    
    return df

def generate_domain_json(df, output_file):
    """Automatically generate a domain JSON file."""
    domain = {}
    for column in df.columns:
        n_unique = df[column].nunique()
        domain[column] = max(2, n_unique)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(domain, f, indent=4)
    
    return domain

# Load the NIST dataset
df = pd.read_csv('datasets/25_PracticeProblem/25_Demo_25f_OriginalData.csv')

# Ensure the secret bit is binary
secret_bit = 'F1'
df = prepare_binary_secret_bit(df, secret_bit)

# Generate the domain JSON file
os.makedirs('datasets/domain', exist_ok=True)
domain = generate_domain_json(df, 'datasets/domain/nist-domain.json')
print(f"Generated domain file with {len(domain)} attributes")

# Save the processed dataset
df.to_csv('datasets/nist_processed.csv', index=False)
print(f"Processed dataset saved with {len(df)} rows and {len(df.columns)} columns")

# Create directories for the attack
data_dir = 'results/nist/reps'
os.makedirs(data_dir, exist_ok=True)

# Generate files for each repetition
for rep in range(32):  # Adjust as needed
    rep_dir = f'{data_dir}/rep_{rep}'
    os.makedirs(rep_dir, exist_ok=True)
    
    # Copy the dataset
    df_copy = df.copy()
    df_copy.to_csv(f'{rep_dir}/df.csv.gz', compression='gzip', index=False)
    
    # Select a random user
    user = np.random.randint(len(df))
    np.savetxt(f'{rep_dir}/user.csv', np.array([user]), fmt='%d')
    
    print(f"Prepared repetition {rep}: selected user {user}")
EOF

# Run the preparation script
python3 prepare_nist.py

# Continue with the attack pipeline
echo "[$(date +%F_%T)] Generating queries for attack..."
python3 gen_queries.py --data_name $DATA_NAME --query_type simple --k 3 --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --secret_bit $SECRET_BIT

# Run the attack
echo "[$(date +%F_%T)] Running recon attack..."
python3 run_attack.py --data_name $DATA_NAME --synth_model NonPrivate --n_rows 1000 --scale_type cond --attack_name recon --reps $REPS --n_procs $N_PROCS --data_dir $DATA_DIR --k 3 --secret_bit $SECRET_BIT

# Analyze the results
echo "[$(date +%F_%T)] Analyzing results..."
python3 analyze_privacy_utility.py --data_name $DATA_NAME --synth_model NonPrivate --n_rows 1000 --k 3 --scale_type cond --n_queries -1 --attack_name recon --secret_bit $SECRET_BIT --reps $REPS --data_dir $DATA_DIR

echo "[$(date +%F_%T)] Complete"
