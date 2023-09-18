from qiskit import BasicAer
from qiskit.utils import algorithm_globals
import numpy as np
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from sklearn.preprocessing import OneHotEncoder
from qiskit.algorithms.optimizers import SPSA
from matplotlib import pyplot as plt


algorithm_globals.random_seed = 1558
np.random.seed(algorithm_globals.random_seed)

TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = (
    ad_hoc_data(training_size=20,
                test_size=5,
                n=2,
                gap=0.3,
                one_hot=False)
)


FEATURE_MAP = ZZFeatureMap(feature_dimension=2, reps=2)
VAR_FORM = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)

AD_HOC_CIRCUIT = FEATURE_MAP.compose(VAR_FORM)
AD_HOC_CIRCUIT.measure_all()
print(AD_HOC_CIRCUIT.decompose().draw())


encoder = OneHotEncoder()
train_labels_oh = encoder.fit_transform(TRAIN_LABELS.reshape(-1, 1)).toarray()
test_labels_oh = encoder.fit_transform(TEST_LABELS.reshape(-1, 1)).toarray()



initial_point = np.random.random(VAR_FORM.num_parameters)
# initial_point = np.array([0.3200227 , 0.6503638 , 0.55995053,
#                           0.96566328, 0.38243769, 0.90403094,
#                           0.82271449, 0.26810137, 0.61076489,
#                           0.82301609, 0.11789148, 0.29667125])
class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        """Save intermediate results. Optimizer passes five values
        but we ignore the last two."""
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)


log = OptimizerLog()

vqc = VQC(feature_map=FEATURE_MAP,
          ansatz=VAR_FORM,
          loss='cross_entropy',
          optimizer=SPSA(callback=log.update),
          initial_point=initial_point,
          quantum_instance=BasicAer.get_backend('qasm_simulator'))

vqc.fit(TRAIN_DATA, train_labels_oh)

fig = plt.figure()
plt.plot(log.evaluations, log.costs)
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.show()

# score == accuracy
print(vqc.score(TEST_DATA, test_labels_oh))



from matplotlib.lines import Line2D
plt.figure(figsize=(9, 6))

for feature, label in zip(TRAIN_DATA, train_labels_oh):
    
    COLOR = 'C1' if label[0] == 0 else 'C0'
    
    plt.scatter(feature[0], feature[1], marker='o', s=100, color=COLOR)

for feature, label, pred in zip(TEST_DATA, test_labels_oh, vqc.predict(TEST_DATA)):
    
    COLOR = 'C1' if pred[0] == 0 else 'C0'
    
    plt.scatter(feature[0], feature[1], marker='s', s=100, color=COLOR)
    
    if not np.array_equal(label,pred):  # mark wrongly classified
        plt.scatter(feature[0], feature[1], marker='o', s=500, linewidths=2.5, facecolor='none', edgecolor='C3')

legend_elements = [
    Line2D([0], [0], marker='o', c='w', mfc='C1', label='A', ms=10),
    Line2D([0], [0], marker='o', c='w', mfc='C0', label='B', ms=10),
    Line2D([0], [0], marker='s', c='w', mfc='C1', label='predict A', ms=10),
    Line2D([0], [0], marker='s', c='w', mfc='C0', label='predict B', ms=10),
    Line2D([0], [0], marker='o', c='w', mfc='none', mec='C3', label='wrongly classified', mew=2, ms=15)
]

plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')

plt.title('Training & Test Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

input("...")