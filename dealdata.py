import pandas as pd
import numpy as np
import argparse

# parser = argparse.ArgumentParser(description='process sparsity')
# parser.add_argument('--depth', default=20, type=int, help='resnet depth')

with open('/shared/hj14/cifar10-dataset/sparsity-32.log') as f:
    lines = (line for line in f if line[0].isdigit())
    FH = np.loadtxt(lines, delimiter=',')


tmp1 = FH.reshape(200,30)
print(tmp1.shape)
tmp2 = np.transpose(tmp1)
print(tmp2.shape)
np.savetxt('sparsity-32.csv', tmp2, delimiter = ',')
