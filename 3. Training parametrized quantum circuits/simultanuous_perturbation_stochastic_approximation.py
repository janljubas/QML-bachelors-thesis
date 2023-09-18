"""
4. Simultaneous Perturbation Stochastic Approximation (SPSA)

Looking at our function f()as a vector, if we want to evaluate the gradient , we need to calculate the partial derivation of 
 with respect to each parameter, meaning we would need 2N function evaluations for N parameters to calculate the gradient.


SPSA is an optimization technique where we randomly sample from the gradient, to reduce the number of evaluations. Since we don't care
about the exact values (but only about the convergence), an unbiased sampling should suffice.

In practice, while the exact gradient follows a smooth path to the minimum, SPSA will jump around due to the random sampling, but it will
converge, goven the samoe boundary conditions as the gradient.

Performance: It basically follows the gradient descent curve, and at a fraction of the cost!

"""















"""
OLD CODE, BUT NECESSARY FOR THE FINAL PLOT
"""

from qiskit.circuit.library import RealAmplitudes
import numpy as np
from qiskit.opflow import Z
from qiskit.opflow import StateFn, PauliExpectation
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler
from qiskit.opflow import Gradient
from qiskit.algorithms.optimizers import GradientDescent
from utils import OptimizerLog
from qiskit.opflow import NaturalGradient
from qiskit.algorithms.optimizers import GradientDescent
from utils import OptimizerLog
import matplotlib.pyplot as plt

ansatz = RealAmplitudes(num_qubits=2, reps=1, entanglement='linear').decompose()

hamiltonian = Z ^ Z

expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
pauli_basis = PauliExpectation().convert(expectation)

quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),  shots = 8192, seed_simulator = 2718, seed_transpiler = 2718)
sampler = CircuitSampler(quantum_instance)

def evaluate_expectation(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(pauli_basis, params=value_dict).eval()
    return np.real(result)

point = np.random.random(ansatz.num_parameters)
INDEX = 2

EPS = 0.2 

e_i = np.identity(point.size)[:, INDEX]

plus = point + EPS * e_i
minus = point - EPS * e_i

finite_difference = ( evaluate_expectation(plus) - evaluate_expectation(minus) ) / (2 * EPS)    # the F.D. approx. method


shifter = Gradient('fin_diff', analytic=False, epsilon=EPS)
grad = shifter.convert(expectation, params=ansatz.parameters[INDEX])

value_dict = dict(zip(ansatz.parameters, point))

EPS = np.pi / 2
e_i = np.identity(point.size)[:, INDEX]

plus = point + EPS * e_i
minus = point - EPS * e_i

finite_difference = ( evaluate_expectation(plus) - evaluate_expectation(minus) ) / 2

shifter = Gradient()
grad = shifter.convert(expectation, params=ansatz.parameters[INDEX])

initial_point = np.random.random(ansatz.num_parameters)

gradient = Gradient().convert(expectation) 
gradient_in_pauli_basis = PauliExpectation().convert(gradient)

sampler = CircuitSampler(quantum_instance)

def evaluate_gradient(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(gradient_in_pauli_basis, params=value_dict).eval()
    return np.real(result)

gd_log = OptimizerLog()
gd = GradientDescent(maxiter=300, learning_rate=0.01, callback=gd_log.update)

result = gd.minimize(    fun=evaluate_expectation,    x0=initial_point,       jac=evaluate_gradient )

natural_gradient = (NaturalGradient(regularization='ridge').convert(expectation))
natural_gradient_in_pauli_basis = PauliExpectation().convert(natural_gradient)
sampler = CircuitSampler(quantum_instance, caching="all")

gradient = Gradient().convert(expectation)
gradient_in_pauli_basis = PauliExpectation().convert(gradient)


def evaluate_natural_gradient(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(natural_gradient, params=value_dict).eval()
    return np.real(result)

initial_point = np.random.random(ansatz.num_parameters)

qng_log = OptimizerLog()
qng = GradientDescent(maxiter=300, learning_rate=0.01, callback=qng_log.update)

result = qng.minimize(evaluate_expectation, initial_point, evaluate_natural_gradient)














""" Finally, the SPSA code """


from qiskit.algorithms.optimizers import SPSA
spsa_log = OptimizerLog()
spsa = SPSA(maxiter=300, learning_rate=0.01, perturbation=0.01, callback=spsa_log.update)

result = spsa.minimize(evaluate_expectation, initial_point)

# Plot loss
# plt.figure(figsize=(7, 3))
# plt.plot(gd_log.loss, 'C0', label='vanilla gradient descent')
# plt.plot(qng_log.loss, 'C1', label='quantum natural gradient')
# plt.plot(spsa_log.loss, 'C0', ls='--', label='SPSA')
# plt.axhline(-1, c='C3', ls='--', label='target')
# plt.ylabel('loss')
# plt.xlabel('iterations')
# plt.legend()
# plt.show()

"""
We can do the same for natural gradients as well, as described in Reference 3. 
We'll skip the details here, but the idea is to sample not only from the gradient, but to extend this to the 
quantum Fisher information and thus to the natural gradient.

Qiskit implements this as the QNSPSA algorithm. Let's compare its performance:
"""

""" QNSPSA code """

from qiskit.algorithms.optimizers import QNSPSA
qnspsa_log = OptimizerLog()

fidelity = QNSPSA.get_fidelity(ansatz, quantum_instance, expectation=PauliExpectation())

qnspsa = QNSPSA(fidelity, maxiter=300, learning_rate=0.01, perturbation=0.01, callback=qnspsa_log.update)

result = qnspsa.minimize(evaluate_expectation, initial_point)

# Plot loss
# plt.figure(figsize=(7, 3))
# plt.plot(gd_log.loss, 'C0', label='vanilla gradient descent')
# plt.plot(qng_log.loss, 'C1', label='quantum natural gradient')
# plt.plot(spsa_log.loss, 'C0', ls='--', label='SPSA')
# plt.plot(qnspsa_log.loss, 'C1', ls='--', label='QN-SPSA')
# plt.axhline(-1, c='C3', ls='--', label='target')
# plt.ylabel('loss')
# plt.xlabel('iterations')
# plt.legend()
# plt.show()

"""
We can see that QNSPSA somewhat follows the natural gradient descent curve.

The vanilla and natural gradient costs are linear and quadratic in terms of the number of parameters, while the costs 
for SPSA and QNSPSA are constant, i.e. independent of the number of parameters.

There is the small offset between the costs for SPSA and QNSPSA as more evaluations are required to approximate the natural gradient.
"""


"""
The Gradient training updates the circuit parameters using the gradient of the loss function.

The Natural Gradient updates the circuit parameters using the quantum natural gradient of the loss function.

The SPSA training method updates the circuit parameters using the approx. gradient of the loss function, calculated using random sampling.

The QNSPSA training method updates the circuit parameters using the approx. quantum natural gradient of the loss function, calculated using random sampling.
"""









""" Training in practice"""


"""
In this era of near-term quantum computing, circuit evaluations are expensive, and readouts are not perfect due to the noisy nature
of the devices.
Therefore in practice, people often resort to using SPSA. To improve convergence, we don't use a constant learning rate,
but an exponentially decreasing one.

The diagram below shows the typical convergence between a constant learning rate (dotted lines) versus an exponentially decreasing one (solid lines).

We see that the convergence for a constant learning rate is smooth decreasing line, while the convergence for an exponentially decreasing one
is steeper and more staggered.
This works well if you know what your loss function looks like.


Qiskit will try to automatically calibrate the learning rate to the model if you don't specify the learning rate.

"""

autospsa_log = OptimizerLog()
autospsa = SPSA(maxiter=300,
                learning_rate=None,     # here the power law gradient method is applied automatically
                perturbation=None,
                callback=autospsa_log.update)

result = autospsa.minimize(evaluate_expectation, initial_point)

# Plot loss
plt.figure(figsize=(7, 3))
plt.plot(gd_log.loss, 'C0', label='vanilla gradient descent')
plt.plot(qng_log.loss, 'C1', label='quantum natural gradient')
plt.plot(spsa_log.loss, 'C0', ls='--', label='SPSA')
plt.plot(qnspsa_log.loss, 'C1', ls='--', label='QN-SPSA')
plt.plot(autospsa_log.loss, 'C3', label='Power-law SPSA')
plt.axhline(-1, c='C3', ls='--', label='target')
plt.ylabel('loss')
plt.xlabel('iterations')
plt.legend()

"""
We see here that it works the best of all the methods for this small model. For larger models,
the convergence will probably be more like the natural gradient.

"""