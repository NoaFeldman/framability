"""
Generate unitary matrices from two-qubit Pauli string coefficients.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable


def pauli_matrices():
    """
    Return the four Pauli matrices: I, X, Y, Z.
    
    Returns:
        dict: Dictionary with 'I', 'X', 'Y', 'Z' keys containing 2x2 matrices
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    return {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def two_qubit_pauli_strings():
    """
    Generate all 16 two-qubit Pauli strings as 4x4 matrices.
    
    Returns:
        list: List of 16 two-qubit Pauli matrices ordered as:
              II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
    """
    paulis = pauli_matrices()
    pauli_labels = ['I', 'X', 'Y', 'Z']
    
    pauli_strings = []
    for p1 in pauli_labels:
        for p2 in pauli_labels:
            # Tensor product of two Pauli matrices
            pauli_2q = np.kron(paulis[p1], paulis[p2])
            pauli_strings.append(pauli_2q)
    
    return pauli_strings


def u_gradient_descent(params):
    """
    Create a 4x4 unitary matrix from 16 real parameters.
    
    The parameters serve as coefficients for the 16 two-qubit Pauli strings.
    The Hamiltonian is formed as a linear combination of these Pauli strings,
    then exponentiated to produce a unitary matrix.
    
    Args:
        params (array-like): Array of 16 real parameters (coefficients for each Pauli string)
        
    Returns:
        np.ndarray: 4x4 complex unitary matrix
    """
    if len(params) != 16:
        raise ValueError(f"Expected 16 parameters, got {len(params)}")
    
    params = np.array(params, dtype=float)
    
    # Get all two-qubit Pauli strings
    pauli_strings = two_qubit_pauli_strings()
    
    # Build the Hamiltonian as linear combination of Paulis
    H = np.zeros((4, 4), dtype=complex)
    for i, param in enumerate(params):
        H += param * pauli_strings[i]
    
    # Exponentiate to create unitary: U = exp(-i * H)
    # Using eigendecomposition for numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues)) @ eigenvectors.conj().T
    
    return U


def gradient_descent_unitary(target_fn: Callable, init_params=None, learning_rate=0.01,
                            max_iterations=1000, tolerance=1e-6, method='BFGS',
                            silent=False, display_fn: Callable = None):
    """
    Perform gradient descent to find a unitary matrix that minimizes a target function.
    
    Uses scipy.optimize to minimize the target function over the space of 4x4 unitaries,
    parametrized by 16 real coefficients for two-qubit Pauli strings.
    
    Args:
        target_fn (callable): Function that takes a 4x4 unitary matrix and returns a scalar loss.
                            Should be minimized.
        init_params (array-like, optional): Initial 16 parameters. If None, random values are used.
        learning_rate (float): Learning rate for gradient descent (used only with 'GD' method).
        max_iterations (int): Maximum number of optimization iterations.
        tolerance (float): Convergence tolerance.
        method (str): Optimization method ('BFGS', 'L-BFGS-B', 'GD' for vanilla gradient descent,
                     or any scipy.optimize.minimize method). Default is 'BFGS'.
        silent (bool): If True, print the target function value at each iteration and
                       suppress all other output. If False (default), no output is printed.
        display_fn (callable, optional): Function that takes a 4x4 unitary and returns the
                       value to print when silent=True. If None, uses target_fn. Useful when
                       optimizing a smooth surrogate but wanting to display the exact metric.
    
    Returns:
        dict: Dictionary containing:
            - 'params': Optimized 16 parameters
            - 'unitary': Optimized 4x4 unitary matrix
            - 'loss': Final loss value
            - 'result': Full scipy OptimizeResult object
    """
    if init_params is None:
        init_params = np.random.randn(16)
    else:
        init_params = np.array(init_params, dtype=float)
    
    if len(init_params) != 16:
        raise ValueError(f"Expected 16 initial parameters, got {len(init_params)}")
    
    # Define cost function for optimization
    def cost_fn(params):
        U = u_gradient_descent(params)
        return target_fn(U)

    # Function used for display when silent=True
    _display = display_fn if display_fn is not None else target_fn
    def display_cost(params):
        U = u_gradient_descent(params)
        return _display(U)

    # Define gradient function using numerical differentiation
    def grad_fn(params):
        eps = 1e-5
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            grad[i] = (cost_fn(params_plus) - cost_fn(params_minus)) / (2 * eps)
        return grad
    
    # Handle vanilla gradient descent
    if method == 'GD':
        params = init_params.copy()
        for iteration in range(max_iterations):
            grad = grad_fn(params)
            params_new = params - learning_rate * grad
            cost_new = cost_fn(params_new)
            cost_old = cost_fn(params)

            if silent:
                print(display_cost(params))

            if np.abs(cost_new - cost_old) < tolerance:
                break

            params = params_new

        result = type('OptimizeResult', (), {
            'x': params,
            'fun': cost_fn(params),
            'nit': iteration + 1,
            'success': True
        })()
    else:
        # Callback prints the current loss when silent=True
        def _callback(params):
            if silent:
                print(display_cost(params))

        # Use scipy.optimize.minimize for other methods
        result = minimize(
            cost_fn,
            init_params,
            method=method,
            jac=grad_fn,
            callback=_callback,
            options={
                'maxiter': max_iterations,
                'gtol': tolerance,
                'disp': False
            }
        )
    
    # Extract optimized unitary
    opt_params = result.x
    opt_unitary = u_gradient_descent(opt_params)
    opt_loss = result.fun
    
    return {
        'params': opt_params,
        'unitary': opt_unitary,
        'loss': opt_loss,
        'result': result
    }


if __name__ == "__main__":
    # Example 1: Generate a random unitary
    print("=" * 60)
    print("Example 1: Generate Random Unitary")
    print("=" * 60)
    params = np.random.randn(16)
    U = u_gradient_descent(params)
    
    print("Generated unitary matrix:")
    print(U)
    print("\nVerifying unitarity (U @ U† ≈ I):")
    is_unitary = np.allclose(U @ U.conj().T, np.eye(4))
    print(is_unitary)
    
    # Example 2: Optimize a unitary to match a target
    print("\n" + "=" * 60)
    print("Example 2: Gradient Descent Optimization")
    print("=" * 60)
    
    # Create a target unitary (e.g., a random one)
    target_params = np.random.randn(16)
    target_U = u_gradient_descent(target_params)
    
    # Define a loss function: minimize distance to target
    def target_loss_fn(U):
        """Loss is Frobenius norm of difference from target unitary."""
        return np.linalg.norm(U - target_U, 'fro') ** 2
    
    print("Target unitary generated.")
    print("Optimizing to find matching unitary...")
    
    # Run gradient descent
    result = gradient_descent_unitary(
        target_loss_fn,
        init_params=np.random.randn(16),
        method='BFGS',
        max_iterations=500,
        tolerance=1e-8
    )
    
    print(f"\nOptimization converged: {result['result'].success}")
    print(f"Final loss: {result['loss']:.2e}")
    print(f"Iterations: {result['result'].nit}")
    
    print("\nVerifying optimized unitary is unitary:")
    U_opt = result['unitary']
    is_unitary_opt = np.allclose(U_opt @ U_opt.conj().T, np.eye(4))
    print(is_unitary_opt)
    
    print("\nDistance to target:")
    distance = np.linalg.norm(U_opt - target_U, 'fro')
    print(f"{distance:.2e}")
