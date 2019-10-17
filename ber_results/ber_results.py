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
eb_n0=np.arange(-3,30,3)

# cubic=np.load('6bit_eval/hp_ber.npy')
linear0=np.load('updated_6bit_eval/pedestrian/0.001/hp_ber.npy')
linear1=np.load('updated_6bit_eval/pedestrian/0.00035/hp_ber.npy')
linear2=np.load('updated_6bit_eval/agrim_manifold_pedestrian/0.00035/hp_ber.npy')
linear4=np.load('updated_6bit_eval/pedestrian/0.00035/ideal_ber.npy')
linear0=linear0[:11]
linear1=linear1[:11]
linear2=linear2[:11]
linear4=linear4[:11]

plt.yscale('log')
plt.ylabel('BER')
plt.xlabel(r'$E_{b}/N_{0}$ (dB)')
# # pdb.set_trace()
# plt.plot(eb_n0,linear0, 'r--o', lw=3, ms=10, label=r'$\times 10^{-1}$')
plt.plot(eb_n0,linear4, 'c--o', lw=3, ms=10, label=r'Ideal - Complete Feedback')
plt.plot(eb_n0,linear0, 'r--o', lw=3, ms=10, label=r'$f_{d}T_{s}=10^{-3}$, Scalar param.')
# plt.plot(eb_n0,linear1,'b--o', lw=3, ms=10, label=r'$f_{d}T_{s}3.5=\times 10^{-4}$-Parameter'))
plt.plot(eb_n0,linear2,'g--o', lw=3, ms=10, label=r'$f_{d}T_{s}=3.5\times 10^{-4}$, Manifold')

v = plt.axis()
plt.axis([-3.4, 27.4, v[2], v[3]])
# plt.plot(eb_n0,lineari, label='3.5e-4 ideal', marker='+')
# plt.plot(eb_n0,linear17i, 'r-', lw=3, ms=10,label='1.7e-4 ideal', marker='+')
# plt.plot(eb_n0,linear1i, label='1e-4 ideal', marker='+')
# plt.plot(eb_n0,nearest, label='nearest', marker='1', color='r')

plt.grid(which='major', linewidth='0.5', color='#888888');
plt.grid(which='minor', linewidth='0.5', color='#cccccc');

plt.legend(fontsize=14.5)
plt.show()

# pylab.rc('text', usetex=True)
# pylab.rc('font', size=45)
# pylab.rc('axes', labelsize=40)
# pylab.rc('legend', fontsize=45)
# pylab.figure(figsize=(8,6))
