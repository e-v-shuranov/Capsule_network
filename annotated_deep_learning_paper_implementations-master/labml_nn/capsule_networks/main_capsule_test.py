

import sys
import platform
import numpy as np

import os
import torch

from labml import experiment
from labml_nn.capsule_networks.mnist import Configs

def main():
    experiment.create(name="capsule_networks")
    conf = Configs()
    experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                              'optimizer.learning_rate': 1e-3,
                              'inner_iterations': 5})
    experiment.add_pytorch_models({'model': conf.model})
    with experiment.start():
        conf.run()

def main_test():
    print(sys.version)
    print(sys.executable)
    print(platform.python_version())
    print("env: ", os.getenv('p38'))
    a = np.arange(6)
    print(a)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=0)
    X = [[ 1,  2,  3], [11, 12, 13]]
    y = [0, 1]  # classes of each sample
    clf.fit(X, y)



if __name__ == '__main__':
    main()