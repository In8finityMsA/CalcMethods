import matplotlib.pyplot as plt
import numpy as np
import sys

m1 = np.loadtxt(sys.argv[1])
m2 = np.loadtxt(sys.argv[2])
m3 = np.loadtxt(sys.argv[3])

plt.semilogy(np.arange(0, len(m1), 1), m1, 'r', color='r', label='w=0.8')
plt.semilogy(np.arange(0, len(m2), 1), m2, 'r', color='g', label='w=1.0')
plt.semilogy(np.arange(0, len(m3), 1), m3, 'r', color='b', label='w=1.2')
plt.ylabel('norm')
plt.xlabel('iteration')
plt.legend(fontsize=14)
plt.show()