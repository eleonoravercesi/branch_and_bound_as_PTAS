def is_integer_sol(X, tol=1e-6):
    """
    Check if the solution is integer

    Parameters:
    - X: dictionary with the solutions (job, machine) -> value
    - tol: tolerance for the comparison

    Returns:
    - True if the solution is integer, False otherwise
    """
    for (j, i) in X.keys():
        if abs(X[j, i] - round(X[j, i])) > tol:
            return False
    return True