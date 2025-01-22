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


def is_integer_val(v, tol=1e-6):
    """
    Check if a value is integer

    Parameters:
    - v: value to check
    - tol: tolerance for the comparison

    Returns:
    - True if the value is integer, False otherwise
    """
    return abs(v - round(v)) < tol
