# Branch-and-Bound Algorithms as Polynomial-time Approximation Schemes

### Koppány István Encz, Monaldo Mastrolilli, and Eleonora Vercesi

This is the accompaning code of the paper "Branch-and-Bound Algorithms as Polynomial-time Approximation Schemes" by Koppány István Encz, Monaldo Mastrolilli, and Eleonora Vercesi accepted at ICALP 2025.

You can find an extended version of the paper also in this repository, named as `main.pdf`

## How this repository is structured
The structure of this repository is as follows:

```text
.
├── BeB
│   ├── multi_knapsack.py
│   ├── unrelated_job_scheduling.py
├── bounds
│   ├── multi_knapsack.py
│   ├── unrelated_job_scheduling.py
├── exact_models
│   ├── multi_knapsack.py
│   ├── unrelated_job_scheduling.py
├── output
│   ├── results_multiknapsack_random_instances.csv
│   ├── results_unrelated_job_scheduling_random_instances.csv
├── main_multi_knapsack.py
├── main_unrelated_job_scheduling.py
├── plots.ipynb
├── README.md
├── requirements.txt
├── SCIP_BeB.py
├── utils.py
```

-  In the folder `BeB`, you can find the implementation of the Branch-and-Bound algorithm.
- In the folder `bounds`, you can find the implementation of the upper and lower bounds we used.
- In the folder `exact_models`, you can find the implementation of the exact models we used for comparison. We used both Google OR-Tools and SCIP. Note that, when using SCIP, we disable heuristics, presolving, and cuts to make it run as a pure branch-and-bound algorithm.
- In the folder `output`, you can find the results of the experiments we ran. If you wish, you can re-run your experiments, but this will overwrite the files in this folder.

## How to use the code
Install the packages in the `requirements.txt` file. You may want to create a new `conda` environment or `virtualenv` for this project. You can install the packages by running the following command:
```bash
pip install -r requirements.txt
```

If you want to fully reproduce our experiments, you need to run the two main files:

```bash
python main_multi_knapsack.py
python main_unrelated_job_scheduling.py
```
These lines create two `.csv` files in the `output` folder. If you don't want to recreate the experiments from scratch, you can use the `.csv` files in the `output` folder.
It may take some time to run all the experiments. If you wish to just do a reduced version of the experiments, you can modify the line 9 of `main_multi_knapsack.py` and line 8 of `main_unrelated_job_scheduling.py` files.

If you want to visualize the results, you can run the following jupyter notebook
```bash
jupyter-notebook plots.ipynb
```
To better visualize the plots, you can use your mouse as follows (tested on a Linux machine):
- Right click and move the mouse right → Zoom in
- Right click and move the mouse left → Zoom out
- Right click and scroll → Move the plot

If you wish to asses the performances of the B&B as implemented in SCIP, you can run the following command:
```bash
python SCIP_BeB.py multi_knapsack 0 29 100 5
python SCIP_BeB.py unrelated_job_scheduling 0 29 50 5
```
This will run the B&B algorithm on 30 instances of the multi-knapsack problem with 100 items and 5 knapsacks and on 30 instances of the unrelated job scheduling problem with 50 jobs and 5 machines. Mean running time and standard deviation will be printed.

## How to modify the code
The code is structured in a way that you can easily modify the experiments. 
More specifically, in the folder `BeB`, there is a class `BranchAndBound`, that contains all the components we use to tune our algorithm. 

- `upper_bound/lower_bound` are the functions that compute the upper and lower bounds of the problem.
- `branching_variable` select the variable to branch on.
- `rounding` is the function that rounds the solution of the LP relaxation.

You can implement your own functions and pass them to the `BranchAndBound` class. Just be sure that the input and output of the functions are the same as the ones we implemented.
