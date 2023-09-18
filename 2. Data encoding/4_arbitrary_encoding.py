import math
from qiskit import QuantumCircuit



"""
3. Angle encoding       ->      no example
"""

qc = QuantumCircuit(3)
qc.ry(0, 0)
qc.ry(2*math.pi/4, 1)
qc.ry(2*math.pi/2, 2)
qc.draw(output='mpl').show()

"""
4. Arbitrary encoding
"""

# using only 3 qubits to encode 12 features!

from qiskit.circuit.library import EfficientSU2
circuit = EfficientSU2(num_qubits=3, reps=1, insert_barriers=True)
print(circuit.decompose().draw())
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
encode = circuit.bind_parameters(x)
print(encode.decompose().draw())
encode.decompose().draw(output='mpl').show()




# ZZFeatureMap circuit with 3 qubits only encodes a data point od 3 features, despoite having 6 parameterized gates:

from qiskit.circuit.library import ZZFeatureMap
circuit = ZZFeatureMap(3, reps=1, insert_barriers=True)
print(circuit.decompose().draw())
x = [0.1, 0.2, 0.3]
encode = circuit.bind_parameters(x)
print(encode.decompose().draw())
encode.decompose().draw(output='mpl').show()


"""
If a parameterized quantum circuit has N parameters, the largest number of features it can encode is N !

The performance of different parameterized quantum circuits on different types of data is an active area od investigation.
"""

input("a")