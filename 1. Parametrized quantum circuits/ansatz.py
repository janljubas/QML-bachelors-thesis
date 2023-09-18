"""
In the literature, depending on the context, parameterized quantum circuits are also called
parameterized trial states, variational forms, or ANSATZES.

"""

from qiskit.circuit import QuantumCircuit, Parameter

# creating the variable (parameter) which will be a part of the "dynamically-defined" input of the circuit
theta = Parameter('θ')  # fun fact: parameter name can be defined by any unicode symbol string

circuit = QuantumCircuit(2)
circuit.rz(theta, 0)
circuit.crz(theta, 0, 1)
circuit.draw(output='mpl').show()   # ili print(circuit.draw())  ili ovo :|




"""
If we want the gates to have different parameters, we can use two Parameters,
or we create a ParameterVector, which acts like a list of Parameters:
"""
from qiskit.circuit import ParameterVector
theta_list = ParameterVector('θ', length=2)

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.rz(theta_list[0], 0)
qc.rx(theta_list[1], 1)
qc.draw(output='mpl').show()




"""
As all quantum gates used in a quantum circuit are unitary, a parameterized circuit itself can be described as a 
unitary operation on n qubits, U_φ, acting on some initial state |φ_0⟩, often set to |0⟩^⊗n.
The resulting parameterized quantum state is |ψ0⟩ = U_φ*|φ_0⟩ , where φ is a set of tunable parameters.
"""
"""
Why are parameterized quantum circuits useful for near-term machine learning algorithms?

They offer a way to implement algorithms on near-term quantum devices.
"""





"""
To use parameterized quantum circuits as a machine learning model, we need them to generalize well.
This means that the circuit should be able to generate a significant subset of the states within the output Hilbert space.
To avoid being easy to simulate on a classical computer, the circuit should also ENTANGLE QUBITS.

"""



"""
Some authors propose the measures of EXPRESSIBILITY and ENTANGLING capability to discriminate between different
parameterized quantum circuits.

We can think of the expressibility of a circuit as the extent to which it can generate states within the Hilbert space.
It can be quantified by computing the extent to which the states generated from the circuit deviate from the uniform distribution.

On the other hand, the entangling capability of a circuit describes its ability to generate entangled states.

[cool example with 2 independent parameters giving the circuit the ability to access many more states]

[Meyer-Wallach measure of state entanglement -> fully-separable has a measure of 0, a Bell state has a M-W measure of 1]

[Hardware efficient quantum circuits]

"""

input("Press Enter to close the window...")
