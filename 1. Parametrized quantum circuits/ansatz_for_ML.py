from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap

"""
In quantum machine learning, parameterized quantum circuits tend to be used for two things:

To encode data, where the parameters are determined by the data being encoded
As a quantum model, where the parameters are determined by an optimization process.

"""

qc_zz = ZZFeatureMap(3, reps=1, insert_barriers=True)
print(qc_zz.decompose().draw())

qc_zz.decompose().draw(output='mpl').show()



"""
A hardware efficient circuit as a quantum model, consisting of alternating layers of single-qubit rotation gates,
followed by two-qubit gates.
In particular, they use y-and-z-rotation gates, and controlled-z gates, which we can build using the TwoLocal circuit:
"""


from qiskit.circuit.library import TwoLocal
qc_twolocal = TwoLocal(num_qubits=3, reps=2, rotation_blocks=['ry','rz'],
                entanglement_blocks='cz', skip_final_rotation_layer=True,
                insert_barriers=True)

print(qc_twolocal.decompose().draw())
qc_twolocal.decompose().draw(output='mpl').show()





"""
The TwoLocal circuit in Qiskit can create many parameterized circuits, such as:
"""

qc_13 = TwoLocal(3, rotation_blocks='ry',
                 entanglement_blocks='crz', entanglement='sca',
                 reps=3, skip_final_rotation_layer=True,
                 insert_barriers=True)

qc_13.decompose().draw(output='mpl').show()





"""
Qiskit's NLocal circuit can also create more general parameterized circuits with alternating rotation and entanglement layers.

Here is a NLocal circuit, with a rotation block on 2 qubits and an entanglement block on 4 qubits using linear entanglement:

"""

from qiskit.circuit.library import NLocal

# rotation block:
rot = QuantumCircuit(2)
params = ParameterVector('r', 2)
rot.ry(params[0], 0)
rot.rz(params[1], 1)

# entanglement block:
ent = QuantumCircuit(4)
params = ParameterVector('e', 3)
ent.crx(params[0], 0, 1)
ent.crx(params[1], 1, 2)
ent.crx(params[2], 2, 3)

# num_qubits predstavlja koliki se broj puta ponavlja ry + rz gate, a kasnije i 3 crx gate
qc_nlocal = NLocal(num_qubits=10, rotation_blocks=rot,
                   entanglement_blocks=ent, entanglement='linear',
                   skip_final_rotation_layer=True, insert_barriers=True)

qc_nlocal.decompose().draw(output='mpl').show()


input("Press Enter to close the window...")

