# Branch-and-Bound Algorithms as Polynomial-time Approximation Schemes

### Koppány István Encz, Monaldo Mastrolilli, and Eleonora Vercesi

This is the accompaning code of the paper "Branch-and-Bound Algorithms as Polynomial-time Approximation Schemes" by Koppány István Encz, Monaldo Mastrolilli, and Eleonora Vercesi.


## How to use the code
1. Install the packages in the requirements.txt file
```bash
pip install -r requirements.txt
```
2. If you want to fully reproduce our experiments, you need to run the two main files

```bash
python main_multi_knapsack.py
python main_unrelated_job_scheduling.py
```
These lines create two `.csv` files in the `output` folder. If you don't want to recreate the experiments from scratch, you can use the `.csv` files in the `output` folder.

3. If you want to visualize the results, you can run the following jupyter notebook
```bash
jupyter-notebook plots.ipynb
```
To better visualize the plots, you can use your mouse as follows (tested on a Linux machine):
- Right click and move the mouse right --> Zoom in
- Right click and move the mouse left --> Zoom out
- Right click and scroll --> Move the plot

## How to modify the code
The code is structured in a way that you can easily modify the experiments. 
More specifically, in the folder `BeB`, there is a class `BranchAndBound`, that contains all the components we use to tune our algorithm. 
- `upper_bound/lower_bound` are the functions that compute the upper and lower bounds of the problem.
- `branching_variable` select the variable to branch on.
- `rounding` is the function that rounds the solution of the LP relaxation.