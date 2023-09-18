'''
3. Quantum Natural gradient
'''

"""
Instead of assuming all the variables are equally sensitive to the small variations (which we do for vanilla gradient),
it would be more correct to take into account the relative steepest descent -> helps in finding the optimum!

In other words, we should calculate the distance depending on the model space, not on the Euclidian parameter space (where all var. are equal)

+ using the so called Quantum Fisher Information (Jacobian which transforms the Euclidian space in our model space) -> g^(-1) (φ_n)

=> quantum natural gradient uses all this in the following formula:
    φ_n+1 = φ_n - η * g^(-1)(φ_n) * ∇f(φ_n)
"""



""" We can evaluate the natural gradient in Qiskit using the NaturalGradient instead of the Gradient. """

from qiskit.opflow import NaturalGradient

""" Start point of old code """

from qiskit.circuit.library import RealAmplitudes
import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler
from qiskit.opflow import Gradient
from qiskit.opflow import Z
from qiskit.opflow import StateFn, PauliExpectation

ansatz = RealAmplitudes(num_qubits=2, reps=1, entanglement='linear').decompose()

hamiltonian = Z ^ Z

expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
pauli_basis = PauliExpectation().convert(expectation)

quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots = 8192, seed_simulator = 2718, seed_transpiler = 2718)
sampler = CircuitSampler(quantum_instance)

natural_gradient = (NaturalGradient(regularization='ridge').convert(expectation))
natural_gradient_in_pauli_basis = PauliExpectation().convert(natural_gradient)
sampler = CircuitSampler(quantum_instance, caching="all")

gradient = Gradient().convert(expectation)
gradient_in_pauli_basis = PauliExpectation().convert(gradient)

def evaluate_expectation(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(pauli_basis, params=value_dict).eval()
    return np.real(result)

def evaluate_gradient(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(gradient_in_pauli_basis, params=value_dict).eval()
    return np.real(result)


""" End point of old code """

def evaluate_natural_gradient(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(natural_gradient, params=value_dict).eval()
    return np.real(result)

initial_point = np.random.random(ansatz.num_parameters)

print('Vanilla gradient:', evaluate_gradient(initial_point))
print('Natural gradient:', evaluate_natural_gradient(initial_point))

# They differ a lot!



from qiskit.algorithms.optimizers import GradientDescent
from utils import OptimizerLog
import matplotlib.pyplot as plt

qng_log = OptimizerLog()
qng = GradientDescent(maxiter=300,
                      learning_rate=0.01,
                      callback=qng_log.update)

result = qng.minimize(evaluate_expectation,
                      initial_point,
                      evaluate_natural_gradient)

# Plot loss
plt.figure(figsize=(7, 3))
# plt.plot(gd_log.loss, 'C0', label='vanilla gradient descent')
plt.plot(qng_log.loss, 'C1', label='quantum natural gradient')
plt.axhline(-1, c='C3', ls='--', label='target')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.legend()