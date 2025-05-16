"""
Utility function to solve L1 minimization problem using Gurobi solver
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def l1_solve(A, noisy_result, procs=1):
    """
    Solve linear program that minimizes L1 norm of error
    i.e. min_c ||A @ c - noisy_result||_1 s.t. 0 <= c <= 1
    by solving the dual linear program
    TODO: insert citation for conversion to dual LP
    
    Parameters
    ----------
    A: np.ndarray
        q x n query matrix
    noisy_result: np.ndarray
        q length vector of (noisy) results to queries
    procs: int
        number of processors to use
    
    Returns
    -------
    est_secret_bits: np.ndarray
        n length {0, 1} vector of estimated 
    sol: np.ndarray
        n length [0, 1] vector of "scores"
    success: bool
        whether Gurobi successfully solved the LP
    """
    m, n = A.shape

    # prepare linear program
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('Threads', procs)
        env.start()
        with gp.Model(env=env) as model:
            # create vars
            x = model.addMVar(shape=n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
            e1 = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="e1")
            e2 = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=0, name="e2")

            c1 = np.ones(m)
            c2 = np.ones(m)
            model.setObjective(c1 @ e1 - c2 @ e2, GRB.MINIMIZE)
            
            A_x = A.copy()
            model.addConstr(A_x @ x - e1 - e2 == noisy_result)

            # solve linear program
            model.optimize()
            if model.status == 2:
                # success
                sol = np.array(model.X)[:n]
            else:
                # failure, solution is random
                sol = np.random.randint(0, 2, size=n)
            est_secret_bits = np.where(sol >= 0.5, 1, 0)

            return est_secret_bits, sol, model.status == 2

def categorical_l1_solve(A, noisy_result, num_categories, procs=1):
    """
    Solve linear program that minimizes L1 norm of error for multi-class reconstruction
    based on the mathematical formulation:
    
    Variables:
    - t_im for each record i and category m
    
    Objective: 
    - Minimize sum of absolute errors |e_j| across all queries j
    
    Constraints:
    - e_j = b_j - Q_j({X|t}) for each query j
    - 0 <= t_im <= 1 for all i, m
    - sum(t_im) = 1 for all i (each record has exactly one category)
    
    Parameters
    ----------
    A: np.ndarray
        q x n query matrix (each row selects records matching query conditions)
    noisy_result: np.ndarray
        q length vector of query results from synthetic data
    num_categories: int
        number of possible values the categorical variable can take
    procs: int
        number of processors to use
    
    Returns
    -------
    est_categories: np.ndarray
        n length vector of estimated categorical values
    sol: np.ndarray
        n x num_categories matrix of "scores" for each category
    success: bool
        whether Gurobi successfully solved the LP
    """
    m, n = A.shape  # m queries, n records

    # Setup Gurobi environment with appropriate parameters
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('Threads', procs)
        env.start()
        
        with gp.Model(env=env) as model:
            # Create variables t_im for each record i and category m
            t = model.addMVar(shape=(n, num_categories), vtype=GRB.CONTINUOUS, lb=0, ub=1, name="t")
            
            # Error variables (positive and negative components for absolute value)
            e_pos = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="e_pos")
            e_neg = model.addMVar(shape=m, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name="e_neg")

            # Objective: minimize sum of absolute errors
            model.setObjective(e_pos.sum() + e_neg.sum(), GRB.MINIMIZE)
            
            # Constraint: Each record must have a valid probability distribution over categories
            for i in range(n):
                model.addConstr(t[i, :].sum() == 1)
            
            # Constraints for query errors:
            # For each query j: e_j = b_j - Q_j({X|t})
            for j in range(m):
                # Get the records selected by this query
                selected_records = A[j, :]
                
                # For each query, we have a target category value
                target_category = j % num_categories + 1  # Cycle through categories
                target_idx = target_category - 1  # Convert to 0-indexed
                
                # Calculate expected result for this query: sum of t_im where i has X_QI and m is target category
                query_result = 0
                for i in range(n):
                    if selected_records[i] > 0:  # If record i is selected by this query
                        query_result += t[i, target_idx]
                
                # Add constraint: e_pos_j - e_neg_j = b_j - query_result
                model.addConstr(e_pos[j] - e_neg[j] == noisy_result[j] - query_result)
            
            # Solve the LP
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                # Extract solution
                sol = np.zeros((n, num_categories))
                for i in range(n):
                    for m in range(num_categories):
                        sol[i, m] = t[i, m].X
                
                # Determine the most likely category for each record
                est_categories = np.argmax(sol, axis=1) + 1  # Convert to 1-indexed
                success = True
            else:
                # Random solution if optimization failed
                sol = np.random.rand(n, num_categories)
                sol = sol / sol.sum(axis=1, keepdims=True)  # Normalize rows
                est_categories = np.random.randint(1, num_categories+1, size=n)
                success = False

            return est_categories, sol, success
