Source code for the paper: A Divide and Learn approach for Predict+optimize with Combinatorial Problems

*There are example files for running experiments for each model in the main folder. 

*Results are saved in the "Tests" folder and you will find python scripts to plot graphs, construct tables in the same folder. DNL uses
regression to warmstart, and there is not a seperate regression file. The first epoch of dnl results is also the result of regression.

*The Dataset is in the "data" folder, Problem instances for benchmark problem are also in the same folder.

**Instances(Loads) {30-39} are scheduling benchmarks with 1 machines
**Instances(Loads) {40-48,400} are scheduling benchmarks with 2 machines
**Instances(Loads) {50-57,500,501} are scheduling benchmarks with 3 machines



Different Problems
*******************
To use Dnl for different problems:
* new solvers need to be defined for in the dnl folder (Similar to Icon Solver and Knapsack Solver). 
* Then these solver methods should be integrated in the Solver.py file.



