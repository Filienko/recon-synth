"""
Utility functions to select and evaluate large number of queries batch by batch
"""
import numpy as np
from tqdm import tqdm

from .simple_kway_queries import *
from .any_kway_queries import *

def gen_kway(attrs, n_queries, query_type, k, select_unique=True):
    """
    Generate unique k-way queries batch by batch

    Parameters
    ----------
    attrs: np.ndarray
        n x d array of attributes for each user
    n_queries: int 
        number of queries to generate
        if n_queries < 0, generate all possible queries over the domain of attrs
    query_type: "simple" | "any"
        "simple" queries are k-way queries that must involve the secret attribute
        "any" queries are k-way queries that can involve any combination of k attributes
    k: int
        number of attributes involved in the query (including secret attribute)
    select_unique: bool
        only keep queries that select unique subsets of users (based on their attributes)
    
    Returns
    -------
    queries: list[(list[int], list[int], int)] | list[(list[int], list[int])]
        (if "simple") list of queries encoded in the form (attribute ids, attribute values, secret attribute value)
        (if "any") list of queries encoded in the form (attribute ids, attribute values)
    """
    queries = set()
    
    pbar = tqdm(total=n_queries, desc='# queries', leave=False) if n_queries > 0 else None 
    while True:
        prev_n_queries = len(queries)

        if query_type == 'simple':
            if n_queries < 0:
                # generate all possible queries over the domain of attrs
                curr_queries = gen_all_simple_kway(attrs, k)
            else:
                # generate queries at random
                curr_queries = gen_rand_simple_kway(attrs, n_queries, k)

            # attach subset of users selected to queries
            subsets = simple_kway(curr_queries, attrs).astype(int)
        elif query_type == 'any':
            if n_queries < 0:
                # generate all possible queries over the domain of attrs
                curr_queries = gen_all_any_kway(attrs, k)
            else:
                # generate queries at random
                curr_queries = gen_rand_any_kway(attrs, n_queries, k)

        if select_unique:
            # compress representation of subsets
            subsets = np.packbits(subsets, axis=1)
            curr_queries = [Query(curr_query, subset) for curr_query, subset in zip(curr_queries, subsets)]

            # add queries to set of queries filtering out queries with same subset
            queries.update(curr_queries)

            # calculate how many "new" queries generated and update progress bar
            update_n_queries = len(queries) - prev_n_queries if len(queries) <= n_queries else n_queries - prev_n_queries
            if n_queries > 0:
                pbar.update(update_n_queries)

            # enough queries have been generated, extract raw query out and break loop
            if n_queries < 0 or len(queries) >= n_queries:
                queries = [query.query for query in queries][:n_queries] 
                break
        else:
            return curr_queries
    
    if pbar is not None:
        pbar.close()
    
    return queries

def get_result(attrs, secret_bits, queries, query_type, batch_size=20000, show_progress=True):
    """
    Calculate results of queries batch by batch (total number of queries and total number of users can be very large)

    Parameters
    ----------
    attrs: np.ndarray
        n x d array of attributes for each user
    secret_bits: np.ndarray
        n array of secret bits for each user
    queries: list[(list[int], list[int], int)] | list[(list[int], list[int])] 
        list of queries generated by gen_kway
    query_type: "simple" | "any"
        "simple" queries are k-way queries that must involve the secret attribute
        "any" queries are k-way queries that can involve any combination of k attributes
    batch_size: int
        size of batch to process
    show_progress: bool
        show progress bar
    
    Returns
    -------
    n_user: np.ndarray
        m array of number of users selected by each query (based on attrs only)
    results: np.ndarray
        m array of results for queries
    """
    # process queries batch by batch
    results, n_users = [], []
    query_range = range(0, len(queries), batch_size)
    if show_progress and len(queries) > batch_size:
        query_range = tqdm(query_range, leave=False, desc='batch_query')

    for query_start_idx in query_range:
        # process users batch by batch
        result, n_user = None, None
        user_range = range(0, len(attrs), batch_size)
        if show_progress and len(attrs) > batch_size:
            user_range = tqdm(user_range, leave=False, desc='batch_user')

        for users_start_idx in user_range:
            curr_attrs = attrs[users_start_idx:users_start_idx+batch_size]
            curr_secret_bits = secret_bits[users_start_idx:users_start_idx+batch_size]
            curr_queries = queries[query_start_idx:query_start_idx+batch_size]
            if query_type == 'simple':
                curr_n_users, curr_result = get_result_simple_kway(curr_attrs, curr_secret_bits, curr_queries)
            elif query_type == 'any':
                curr_n_users, curr_result = get_result_any_kway(curr_attrs, curr_queries)

            if result is None:
                result = curr_result
                n_user = curr_n_users
            else:
                result = result + curr_result
                n_user = n_user + curr_n_users
               
        results.append(result)
        n_users.append(n_user)

    # combine all batches of results together
    result = np.concatenate(results)
    n_user = np.concatenate(n_users)

    return n_user, result

class Query(object):
    """Utility class to encapsulate query and corresponding subset of users selected by query (based on attrs)"""
    def __init__(self, query, subset):
        self.query = query
        self.subset = tuple(subset)
    def __eq__(self, other):
        if isinstance(other, Query):
            return (self.subset == other.subset)
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash(self.subset)
