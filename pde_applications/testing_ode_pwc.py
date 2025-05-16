import numpy as np
import pickle
import matplotlib.pyplot as plt

def eval_cheby(x, val, index):
    # Evaluate the Chebyshev polynomial
    res = np.zeros((x.shape[0]),dtype=np.complex_)
    scal = 2.0 ** (np.sum(index != 0, axis=1) / 2)
    for j in range(x.shape[0]):
        res[j] = np.sum(val * scal * np.prod(np.cos(np.arccos(x[j,:]) * index), axis=1)) 
    return res

def eval_for_x(x):
    a_given = np.array([0, 0, 0, 0, -2, -2, -2, -2])  # B-spline coefficients for given RHS f
    return eval_cheby(np.append(x, 1/2*a_given).reshape(1, -1), results['val'], results['index'])

# Define the true solution
def true_solution(x):
    if -1 <= x < 0:
        return -2/3 * x - 2/3
    elif 0 <= x <= 1:
        return x**2 - 1/3 * x - 2/3
    else:
        return 0.0

# Load approximation results
file_path = 'results_ode_pwc/s100n16.pickle'
with open(file_path, 'rb') as file:
    results = pickle.load(file)
    
# Evaluate the true solution at corresponding x-values
x_values = np.linspace(-1,1,1000)
true_values = np.array([true_solution(x) for x in x_values])

# Solution by our approximation
computed_values = np.array([eval_for_x(x) for x in x_values]).reshape(-1)

# Compute absolute error
abs_error = np.abs(computed_values - true_values)

# Print error statistics
print(f"Max absolute error: {np.max(abs_error)}")
print(f"Mean absolute error: {np.mean(abs_error)}")

# Plot comparison (FEniCS Solution vs True Solution)
plt.figure(figsize=(10, 6))
plt.plot(x_values, computed_values, 'o', label='FEniCS Solution')
plt.plot(x_values, true_values, '-', label='True Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Comparison of FEniCS and True Solution')
plt.show()

# Plot absolute error
plt.figure(figsize=(10, 6))
plt.plot(x_values, abs_error, label='Absolute Error', color='r')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.legend()
plt.title('Absolute Error')
plt.show()