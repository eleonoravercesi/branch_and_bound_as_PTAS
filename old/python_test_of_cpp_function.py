from algorithms import JS_ILP
from old.parse_files import parse_instance

if __name__ == '__main__':
    # instance = "01.txt"
    # directory_name = "/home/vercee/Documents/beb_cpp/data/simple_instances/"
    instance = "111.txt"
    directory_name = "/home/vercee/Documents/branch_and_bound_for_job_scheduling/data/instancias1a100/"
    P = parse_instance(instance, directory_name)
    n_jobs, n_machines = P.shape
    # Make the P a column vector -- we are in the identical case
    P = P[:, 0].reshape(-1, 1)
    #val, X = list_scheduling_algorithm_identical(P, n_machines = 10)
    out = JS_ILP(P, n_machines = n_machines)
    print("Optimal value: ", out[0])
