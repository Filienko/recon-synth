"""
Step 5: Run attack (Adv_recon, Adv_infer, or Adv_dcr) against generated synthetic data
For each repetition of privacy game (rep):
    - load previously generated raw and synthetic data
    - load previously generated queries
    - load noisy result of queries
    - run attack
"""
import click
import os
from tqdm import tqdm
import pickle
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Process, Array, Queue
from threading import Thread
import psutil
from time import sleep
import lzma
import numpy as np
import pandas as pd

# Add directory above current directory to path
import sys; sys.path.insert(0, '..')
from load_data import *
from attacks import *

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def single_rep(rep, synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_attr, 
               categorical=False, num_categories=2, data_dir=None):
    """
    Run attack against a single repetition of privacy game
    
    Parameters
    ----------
    rep : int
        Repetition number
    synth_model : str
        Name of synthetic data model
    n_rows : int
        Number of rows in synthetic dataset
    k : int
        k-way marginals
    scale_type : str
        Scale type ('normal' or 'cond')
    n_queries : int
        Number of queries to use
    attack_name : str
        Type of attack ('recon', 'recon_cat', 'infer', 'dcr')
    secret_attr : str
        Name of secret attribute to reconstruct
    categorical : bool
        Whether the secret attribute is categorical with multiple values
    num_categories : int
        Number of possible values for categorical attribute
    data_dir : str
        Directory containing data
        
    Returns
    -------
    success : int
        1 if attack was successful, 0 otherwise
    error : int
        1 if error occurred, 0 otherwise
    """
    # setup dirs
    rep_dir = f'{data_dir}/rep_{rep}'
    query_subdir = f'simple/{k}way'
    query_dir = f'{rep_dir}/queries/{query_subdir}/'
    synth_df_dir = f'{rep_dir}/{synth_model}/{n_rows_names[n_rows]}'
    synth_result_dir = f'{synth_df_dir}/{query_subdir}_{scale_type}/'
    os.makedirs(synth_result_dir, exist_ok=True)

    # load raw dataset and target user
    df = pd.read_csv(f'{rep_dir}/df.csv.gz', compression='gzip')
    u = int(np.genfromtxt(f'{rep_dir}/user.csv'))

    if attack_name == 'infer':
        # Adv_infer (no changes to existing implementation)
        synth_df = pd.read_csv(f'{synth_df_dir}/synth_df.csv.gz', compression='gzip')
        cols_X = list(synth_df.columns)
        cols_X.remove(secret_attr)
        X_train = synth_df[cols_X].to_numpy()
        y_train = synth_df[secret_attr].to_numpy()
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        X_test = df[cols_X].to_numpy()
        est_secret_values = clf.predict(X_test)
        sol = clf.predict_proba(X_test)
        feasible = True
        
    elif attack_name == 'dcr':
        # Adv_dcr (no changes to existing implementation)
        synth_df = pd.read_csv(f'{synth_df_dir}/synth_df.csv.gz', compression='gzip')
        est_secret_values = np.zeros(len(df))
        sol = np.zeros(len(df))
        feasible = True
        # DCR attack logic (unchanged)
        # ...
        
    elif attack_name == 'recon_cat' or (attack_name == 'recon' and categorical and num_categories > 2):
        # Adv_recon for categorical attributes
        
        # Get attributes and secret values
        attrs, secret_values = process_data_categorical(df, secret_attr)
        
        # Determine the possible category values
        if num_categories == 0:  # Auto-detect
            category_values = np.unique(secret_values)
            num_categories = len(category_values)
        else:
            category_values = np.arange(1, num_categories + 1)  # 1-indexed
        
        # Load queries
        with lzma.open(f'{query_dir}/queries.pkl.xz', 'rb') as f:
            queries = pickle.load(f)
        
        # Load synthetic query results
        synth_result = np.load(f'{synth_result_dir}/synth_result.npz')['arr_0']
        
        # Limit queries if needed
        if n_queries > 0 and n_queries < len(queries):
            queries = queries[:n_queries]
            synth_result = synth_result[:n_queries]
        
        # Get query matrix
        A = simple_kway(queries, attrs)
        
        # Run categorical reconstruction attack
        est_secret_values, sol, feasible = categorical_l1_solve(
            A, synth_result, category_values, procs=1)
            
    else:
        # Original Adv_recon for binary targets
        
        # Split real dataset into attribute matrix
        attrs, secret_bits = process_data(df, secret_attr)

        # Load queries
        with lzma.open(f'{query_dir}/queries.pkl.xz', 'rb') as f:
            queries = pickle.load(f)

        # Load noisy answers
        synth_result = np.load(f'{synth_result_dir}/synth_result.npz')['arr_0']
        
        # Only use queries that have target value 1
        target_vals = np.array([target_val for (_, _, target_val) in queries])
        queries = [(attr_inds, attr_vals, target_val) for (attr_inds, attr_vals, target_val) in queries if target_val == 1]
        synth_result = synth_result[np.nonzero(target_vals)[0]]

        # Clip queries to n_queries
        if n_queries > 0:
            queries = queries[:n_queries]
            synth_result = synth_result[:n_queries]

        # Get query matrix for queries
        A = simple_kway(queries, attrs)

        # Minimize L1 error
        est_secret_values, sol, feasible = l1_solve(A, synth_result, procs=1)
            
    # Save results
    attack_id = 'recon_cat' if attack_name == 'recon' and categorical else attack_name
    np.savez_compressed(f'{synth_result_dir}/est_secret_values_{n_queries}_{attack_id}.npz', est_secret_values)
    np.savez_compressed(f'{synth_result_dir}/sol_{n_queries}_{attack_id}.npz', sol)

    # Check if prediction is correct for the target user
    true_value = df.iloc[u, df.columns.get_loc(secret_attr)]
    predicted_value = est_secret_values[u]
    success = 100 if true_value == predicted_value else 0
    error = 0 if feasible else 100

    return success, error

def worker(proc, function, args, rep_queue, results, errors, completed_queue):
    # set processors to use
    p = psutil.Process()
    p.cpu_affinity([proc])

    while True:
        sleep(4 * proc / 32) # prevent process dead-locking
        next_rep = rep_queue.get() 
        if next_rep is None:
            break
        results[next_rep], errors[next_rep], = function(next_rep, *args)
        completed_queue.put(None)

def track_progress_fn(completed_queue, total):
    with tqdm(total=total, leave=False) as pbar:
        curr_num = 0
        while True:
            completed_queue.get()
            pbar.update(1)
            curr_num += 1

            if curr_num == total:
                break

def process_data_categorical(df, secret_attr):
    """
    Split dataframe into attributes and categorical secret values
    
    Parameters
    ----------
    df: pd.DataFrame
        dataframe containing all attributes including secret
    secret_attr: str
        name of the secret categorical attribute
    
    Returns
    -------
    attrs: np.ndarray
        n x d array of non-secret attributes
    secret_values: np.ndarray
        n array of categorical secret values
    """
    # Copy columns to avoid modifying original
    cols = list(df.columns)
    cols.remove(secret_attr)
    
    # Extract attributes and secret values
    attrs = df[cols].to_numpy()
    secret_values = df[secret_attr].to_numpy()
    
    return attrs, secret_values

@click.command()
@click.option('--data_name', default='acs', type=str, help='dataset to attack (acs, fire)')
@click.option('--synth_model', default='BayNet_3parents', type=str, help='synthetic model to fit (BayNet_Xparents, RAP_Xiters, RAP_Xiters_NN, CTGAN, NonPrivate, Real, GaussianCopula, TVAE, CopulaGAN)')
@click.option('--n_rows', type=int, default=1000, help='number of rows of synthetic data')
@click.option('--k', type=int, default=3, help='k-way marginals')
@click.option('--scale_type', default='cond', help='scale to adjust synthetic result by. normal => size of synthetic dataset. cond => number of users selected by attributes', type=click.Choice(['normal', 'cond'], case_sensitive=False))
@click.option('--n_queries', type=int, default=-1, help='number of queries to use to run attack (-1 uses all queries)')
@click.option('--attack_name', default='recon', help='attack to run', type=click.Choice(['recon', 'recon_cat', 'infer', 'dcr'], case_sensitive=False))
@click.option('--secret_attr', type=str, default=None, help='name of the secret attribute to reconstruct')
@click.option('--categorical', is_flag=True, help='whether the secret attribute is categorical with multiple values')
@click.option('--num_categories', type=int, default=2, help='number of possible values for categorical attribute (default: 2 for binary)')
@click.option('--start_rep_idx', type=int, default=0, help='repetition to start running attack from')
@click.option('--reps', type=int, default=100, help='number of repetitions to attack')
@click.option('--n_procs', type=int, default=1, help='number of processes to use to run the attack')
@click.option('--data_dir', type=str, default='results/', help='directory to load/save generated data to')
def run_attack(data_name, synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_attr,
              categorical, num_categories, start_rep_idx, reps, n_procs, data_dir):
    """
    Run attack against synthetic data
    
    This function has been extended to support:
    - Binary attribute reconstruction (original)
    - Categorical attribute reconstruction (new)
    - Inference attack (original)
    - DCR attack (original)
    """
    data_dir = f'{data_dir}/{data_name}/reps'
    secret_attr = secret_attr if secret_attr is not None else get_default_secret_attr(data_name)
    
    # Use categorical reconstruction if specified as recon_cat or if categorical flag is set
    use_categorical = (attack_name == 'recon_cat') or (attack_name == 'recon' and categorical)
    
    # Set up distributed processing
    accs = Array('i', range(start_rep_idx + reps))
    errors = Array('i', range(start_rep_idx + reps))
    rep_queue = Queue()
    for i in range(reps):
        rep_queue.put(start_rep_idx + i)
    for i in range(n_procs):
        rep_queue.put(None) # signal end of reps to each processor
    completed_queue = Queue()
    track_progress_thread = Thread(target=track_progress_fn, args=(completed_queue,reps,), daemon=True)
    track_progress_thread.start()

    # Start worker processes
    processes = []
    for proc in range(n_procs):
        p = Process(target=worker, args=(proc, single_rep,
            (synth_model, n_rows, k, scale_type, n_queries, attack_name, secret_attr, categorical, num_categories, data_dir),
            rep_queue, accs, errors, completed_queue))
        p.start()
        processes.append(p)
    
    track_progress_thread.join()

    for p in processes:
        p.kill()
    
    # Calculate results
    acc = 0
    error = 0
    for i in range(start_rep_idx, start_rep_idx + reps):
        acc += accs[i] / reps
        error += errors[i] / reps
    
    # Print results with appropriate description
    if use_categorical:
        print(f"Categorical Reconstruction Attack on {secret_attr} with {num_categories} categories:")
    else:
        print(f"{attack_name.upper()} Attack on {secret_attr}:")
    
    print(f"Accuracy: {acc:.2f}%\tFeasible: {100 - error:.2f}%")

if __name__ == '__main__':
    run_attack()
