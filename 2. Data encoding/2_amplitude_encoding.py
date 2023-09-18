import math
from qiskit import QuantumCircuit
"""
2. Amplitude encoding
"""
w_norm = math.sqrt(17.25)
desired_state = [
    1 / w_norm * 1.5,
    1 / w_norm * 0,
    1 / w_norm * -2,
    1 / w_norm * 3,
    1 / w_norm * 1,
    1 / w_norm * -1,
    0,
    0
    ]

qc = QuantumCircuit(3)
qc.initialize(desired_state, [0,1,2])

print( qc.decompose().decompose().decompose().decompose().decompose().draw() )

qc.decompose().decompose().decompose().decompose().decompose().draw(output='mpl').show()
input("a")
