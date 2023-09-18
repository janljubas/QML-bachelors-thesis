from qiskit.circuit.library import RealAmplitudes
import numpy as np

# defining the unitary operator U(φ)        ( that is, |Ψ(φ)> = U(φ) * |00...0> ) -> a parameterized state
ansatz = RealAmplitudes(num_qubits=2, reps=1, entanglement='linear').decompose()
print(ansatz.draw())


# defining the Hamiltonian
from qiskit.opflow import Z, I
hamiltonian = Z ^ Z


# putting them together to define the expectation value, <Ψ(φ)| H |Ψ(φ)>
from qiskit.opflow import StateFn, PauliExpectation
expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
pauli_basis = PauliExpectation().convert(expectation)



# a function to simulate the measurement of the expectation value
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler

quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                   # we'll set a seed for reproducibility
                                   shots = 8192, seed_simulator = 2718,
                                   seed_transpiler = 2718)
sampler = CircuitSampler(quantum_instance)

def evaluate_expectation(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(pauli_basis, params=value_dict).eval()
    return np.real(result)


# We'll choose a random point and index i = 2
point = np.random.random(ansatz.num_parameters)
INDEX = 2




""" FINITE DIFFERENCE GRADIENTS """

print("Finite difference gradient:")

EPS = 0.2   # -> ε

# make identity vector with a 1 at index ``INDEX``, otherwise 0
e_i = np.identity(point.size)[:, INDEX]

plus = point + EPS * e_i
minus = point - EPS * e_i

finite_difference = ( evaluate_expectation(plus) - evaluate_expectation(minus) ) / (2 * EPS)    # the F.D. approx. method
print(finite_difference)



# Instead of doing this manually, we can use Qiskit's Gradient class for this:

from qiskit.opflow import Gradient

shifter = Gradient('fin_diff', analytic=False, epsilon=EPS)
grad = shifter.convert(expectation, params=ansatz.parameters[INDEX])
# print(grad)

value_dict = dict(zip(ansatz.parameters, point))
print(sampler.convert(grad, value_dict).eval().real)




""" ANALYTIC GRADIENTS """

print("Analytic gradients:")

EPS = np.pi / 2
e_i = np.identity(point.size)[:, INDEX]

plus = point + EPS * e_i
minus = point - EPS * e_i

finite_difference = ( evaluate_expectation(plus) - evaluate_expectation(minus) ) / 2
print(finite_difference)


shifter = Gradient()  # parameter-shift rule is the default
grad = shifter.convert(expectation, params=ansatz.parameters[INDEX])
# print(grad)
print(sampler.convert(grad, value_dict).eval().real)




# 1. Choosing an initial point ~ a starting point for the algorithm
initial_point = np.random.random(ansatz.num_parameters)
# initial_point = np.array([0.43253681, 0.09507794, 0.42805949, 0.34210341])



# 2. Creating a function which evaluates the gradient at each step -> needed to reduce its error and guide the optimization process

gradient = Gradient().convert(expectation)  # calculate the gradient of the expectation value, <Ψ(φ)| H |Ψ(φ)>  (converted into a gradient expression)
gradient_in_pauli_basis = PauliExpectation().convert(gradient)  # converting the gradient expression into the Pauli basis representation:
                                                                # necessary because the gradient is now represented in the form of Pauli operators,
                                                                # which are the basis of representing quantum states and operators

sampler = CircuitSampler(quantum_instance)  # `CircuitSampler` object is used to evaluate an operator on a quantum device or simulator
                                            # Here, it is instantiated with the quantum_instance variable (qiskit.Aer backend - qasm simulator)

def evaluate_gradient(theta):   # `theta` are the values of the parameters of the quantum circuit

    value_dict = dict(zip(ansatz.parameters, theta))    # Zipping together the parameters of a param. quant. circuit with the `theta` values.
                                                        # This creates a dictionary that maps each parameter to its corresponding value.
    
    result = sampler.convert(gradient_in_pauli_basis, params=value_dict).eval() # the sampler object is then used to evaluate the gradient
    
    return np.real(result)




# 3. To compare the convergence of the optimizers, we can keep track of the loss at each step by using a callback function.

from qiskit.algorithms.optimizers import GradientDescent
from utils import OptimizerLog
gd_log = OptimizerLog()
gd = GradientDescent(maxiter=300,
                     learning_rate=0.01,
                     callback=gd_log.update)
# A callback function is a user-defined function that is called after each iteration of the optimization algorithm.
# It allows us to monitor and store intermediate results or perform additional tasks during the optimization process.
    # -> Optimizer will call this function after each iteration, passing the relevant values (including the loss) as arguments.

'''
The `OptimizerLog` class in this code snippet allows us to keep track of the loss values during the optimization process.
After running the optimization, we can access the loss list from the gd_log instance to examine the convergence behavior of the optimizer.
'''

result = gd.minimize(
    fun=evaluate_expectation,  # function to minimize
    x0=initial_point,          # initial point
    jac=evaluate_gradient      # function to evaluate gradient
)

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 3))
plt.plot(gd_log.loss, label='vanilla gradient descent')     # here we plot the loss function with each step of the optimization process
plt.axhline(-1, ls='--', c='C3', label='target')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.legend()

plt.show()

input("Press Enter to close the window...")