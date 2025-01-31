import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_weights(I_util, I_fair, lambda_param=1.0):
    """
    Solve the linear programming problem:
    minimize sum(I_util * epsilon) + lambda * sum(epsilon)
    subject to:
        sum(I_fair * epsilon) <= 0
        epsilon <= epsilon_bar
        epsilon >= -epsilon_bar
    
    Args:
        I_util: Utility influence values
        I_fair: Fairness influence values
        lambda_param: Regularization parameter (positive)
    
    Returns:
        epsilon: Optimal weights
    """
    try:
        # Create a new model
        m = gp.Model("weight_optimization")
        
        # Number of data points
        n = len(I_util)
        
        # Create variables
        epsilon = m.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="epsilon")
        
        # Set objective: minimize sum(I_util * epsilon) + lambda * sum(epsilon)
        obj = gp.quicksum(I_util[i] * epsilon[i] for i in range(n)) + \
              lambda_param * gp.quicksum(epsilon[i] for i in range(n))
        m.setObjective(obj, GRB.MINIMIZE)
        
        # Add fairness constraint: sum(I_fair * epsilon) <= 0
        m.addConstr(gp.quicksum(I_fair[i] * epsilon[i] for i in range(n)) <= 0, "fairness")
        
        # Add bound constraints for each epsilon
        epsilon_bar = 1.0  # You can adjust this value
        for i in range(n):
            m.addConstr(epsilon[i] <= epsilon_bar, f"upper_bound_{i}")
            m.addConstr(epsilon[i] >= -epsilon_bar, f"lower_bound_{i}")
        
        # Solve the model
        m.optimize()
        
        # Get the solution
        epsilon_values = np.array([epsilon[i].X for i in range(n)])
        
        return epsilon_values
        
    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
        return None

def main():
    # Load influence values
    util_infl = np.load('util_infl.npy')
    fair_infl = np.load('fair_infl.npy')
    
    # Solve for different lambda values
    lambda_values = [0.1, 1.0, 10.0]
    for lambda_param in lambda_values:
        print(f"\nSolving with lambda = {lambda_param}")
        epsilon = solve_weights(util_infl, fair_infl, lambda_param)
        
        if epsilon is not None:
            non_zero_weights = np.sum(np.abs(epsilon) > 1e-6)
            max_weight = np.max(epsilon)
            min_weight = np.min(epsilon)
            util_influence = np.sum(util_infl * epsilon)
            fair_influence = np.sum(fair_infl * epsilon)
            
            # Save results to .npy files
            np.save(f'weights_lambda_{lambda_param}_epsilon.npy', epsilon)
            np.save(f'weights_lambda_{lambda_param}_summary.npy', np.array([
                non_zero_weights, max_weight, min_weight, util_influence, fair_influence
            ]))
            
            print(f"Number of non-zero weights: {non_zero_weights}")
            print(f"Max weight: {max_weight:.4f}")
            print(f"Min weight: {min_weight:.4f}")
            print(f"Utility influence: {util_influence:.4f}")
            print(f"Fairness influence: {fair_influence:.4f}")

if __name__ == "__main__":
    main()
