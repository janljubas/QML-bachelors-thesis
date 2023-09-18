import math
from qiskit import QuantumCircuit


"""
Data representation is crucial for the success of machine learning models. For classical machine learning,
the problem is how to represent the data numerically, so that it can be best processed by a classical machine learning algorithm.
"""


"""
For quantum machine learning, this question is similar, but more fundamental:
how to represent and efficiently input the data into a quantum system, so that it can be processed by a quantum machine learning algorithm?

This is usually referred to as data encoding, but is also called data embedding or loading.

This process is a critical part of quantum machine learning algorithms and directly affects their computational power.
"""

"""
1. Basis encoding
"""

# if X = {x1 = 011, x2 = 110}   ->   only a few of the values in the state space (2 out of 2^3)
# import math
# from qiskit import QuantumCircuit
desired_state = [ 0, 0, 0, 1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2), 0 ]
qc = QuantumCircuit(3)
qc.initialize(desired_state, [0,1,2])


print(qc.draw())
qc.draw(output='mpl').show()
# print(qc.decompose().draw())
# print(qc.decompose().decompose().draw())
# print(qc.decompose().decompose().decompose().draw())
# print(qc.decompose().decompose().decompose().decompose().draw())

print(qc.decompose().decompose().decompose().decompose().decompose().draw())
qc.decompose().decompose().decompose().decompose().decompose().draw(output='mpl').show()

input("a")