# Pattern-and-Anomaly-Detection-Lab

# Hypreparameters
Hyperparameters can affect the speed and also the accuracy of the final model. Hyperparameter optimization finds a tuple of hyperparameters that lead to the model which better solves the problem. 

Here, a list of the three most widespread algorithms to perform hyperparameters optimization:
1. Grid search: It performs an exhaustive search by evaluating any candidates’ combinations. Obviously, it could result in an unfeasible computing cost, so grid search is an option only when the number of candidates is limited enough.
2. Random search: Providing a cheaper alternative, random search tests only as many tuples as you choose. The selection of the values to evaluate is completely random. Logically the required time decreases significantly. Apart from speed, Random search takes advantage of randomization in the case of continuous hyperparameters that must be discretized when optimized by Grid search. 
3. Bayesian optimization: Contrary to Grid and random search, Bayesian optimization uses previous iterations to guide the next ones. It consists of building a distribution of functions (Gaussian Process) that best describes the function to optimize. In this case, hyperparameter optimization, the function to optimize is those which, given the hyperparameters, returns the performance of the trained model they would lead to. After every step, this distribution of functions is updated and the algorithm detects which regions in the hyperparameter space are more interesting to explore and which are not.
