import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc
from matplotlib.pyplot import figure

font = {'size'   : 18}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

plt.figure(figsize=(8, 6))
eb_n0=np.arange(-6,20,3)

# cubic=np.load('6bit_eval/hp_ber.npy')
linear35=np.load('updated_6bit_eval/pedestrian/0.00035/hp_ber.npy')
linear35i=np.load('updated_6bit_eval/pedestrian/0.00035/ideal_ber.npy')

linear1=np.load('updated_6bit_eval/pedestrian/0.001/hp_ber.npy')
linear1i=np.load('updated_6bit_eval/pedestrian/0.001/ideal_ber.npy')

plt.yscale('log')
plt.ylabel('BER')
plt.xlabel(r'$E_{b}/N_{0}$ (dB)')

plt.plot(eb_n0,linear35, 'r--o', lw=3, ms=10, label=r'$3.5 \times 10^{-4}$')
plt.plot(eb_n0,linear1, 'g--o', lw=3, ms=10, label=r'$1.0 \times 10^{-3}$')
plt.plot(eb_n0,linear35i,'b--o', lw=3, ms=10, label=r'Ideal')
# plt.plot(eb_n0,linear1i,'b--o', lw=3, ms=10, label=r'Ideal $1.0 \times 10^{-3}$'))


# plt.plot(eb_n0,lineari, label='3.5e-4 ideal', marker='+')
# plt.plot(eb_n0,linear17i, 'r-', lw=3, ms=10,label='1.7e-4 ideal', marker='+')
# plt.plot(eb_n0,linear1i, label='1e-4 ideal', marker='+')
# plt.plot(eb_n0,nearest, label='nearest', marker='1', color='r')

plt.grid(which='major', linewidth='0.5', color='black');
plt.grid(which='minor', linewidth='0.5', color='grey');

plt.legend(fontsize=14.5)
plt.show()

# pylab.rc('text', usetex=True)
# pylab.rc('font', size=45)
# pylab.rc('axes', labelsize=40)
# pylab.rc('legend', fontsize=45)
# pylab.figure(figsize=(8,6))
