import pandas as pd
import numpy as np


with open('/shared/hj14/cifar10-dataset/sparsity-34.log') as f:
    lines = (line for line in f if line[0].isdigit())
    FH = np.loadtxt(lines, delimiter=',')


tmp1 = FH.reshape(200,32)
print(tmp1.shape)
tmp2 = np.transpose(tmp1)
print(tmp2.shape)
np.savetxt('sparsity-34.csv', tmp2, delimiter = ',')
