import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc
from matplotlib.pyplot import figure

font = {'size'   : 18}
plt.rc('font', **font)
# plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

plt.figure(figsize=(8, 6))
eb_n0=np.arange(-3,25,3)

# cubic=np.load('6bit_eval/hp_ber.npy')
# linear0=np.load('updated_6bit_eval/pedestrian/0.1/hp_cap.npy')
# linear00=np.load('updated_6bit_eval/pedestrian/0.1/hp_cap.npy')
# linear0=linear0/linear00

# linear1=np.load('updated_6bit_eval/pedestrian/0.01/hp_cap.npy')
# linear11=np.load('updated_6bit_eval/pedestrian/0.01/max_cap.npy')
# linear1=linear1/linear11

# linear2=np.load('updated_6bit_eval/pedestrian/0.001/hp_cap.npy')
# linear22=np.load('updated_6bit_eval/pedestrian/0.001/max_cap.npy')
# linear2=linear2/linear22
veh_after5=np.array([2.23854902e-02, 5.85075132e-03, 9.43905410e-04, 1.79587744e-04, 7.84925235e-05, 4.36302923e-05, 2.41909232e-05, 1.23645413e-05, 5.51285282e-06, 1.96887601e-06])
ped_after5=np.load('ped_after5/0.00035/hp_ber.npy')[:10]
ped_par=np.load('pedestrian/0.00035/hp_ber.npy')[:10]
ped_man=np.load('agrim_manifold_pedestrian/0.00035/hp_ber.npy')[:10]
ped_ideal=np.load('pedestrian/0.00035/ideal_ber.npy')[:10]

plt.yscale('log')
plt.ylabel('BER')
plt.xlabel(r'$E_{b}/N_{0}$ (dB)')
# pdb.set_trace()
# plt.plot(eb_n0,linear0, 'r--o', lw=3, ms=10, label=r'$\times 10^{-1}$')
# plt.plot(eb_n0,linear1, 'g--o', lw=3, ms=10, label=r'$\times 10^{-2}$')
# plt.plot(eb_n0,linear2,'b--o', lw=3, ms=10, label=r'$\times 10^{-3}$')
# plt.plot(eb_n0,linear3,'y--o', lw=3, ms=10, label=r'$\times 10^{-4}$')
plt.plot(eb_n0, ped_after5, label=r'scalar_new')
plt.plot(eb_n0, veh_after5, label=r'vehicular')
plt.plot(eb_n0, ped_par, label=r'scalar')
plt.plot(eb_n0, ped_man, label=r'manifold')
plt.plot(eb_n0, ped_ideal, label='ideal')

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
