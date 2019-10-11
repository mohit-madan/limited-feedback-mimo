import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc
from matplotlib.pyplot import figure

# font = {'size'   : 22}
# rc('font', **font)
# rc('text', usetex=True)
# rc('xtick', labelsize=40)
# rc('ytick', labelsize=40)
plt.figure(figsize=(8, 6))
eb_n0=np.arange(-6,20,3)

# cubic=np.load('6bit_eval/hp_ber.npy')
linear17=np.load('6bit_eval/0.00017-linear/hp_ber.npy')
linear1=np.load('6bit_eval/0.0001-linear/hp_ber.npy')
linear=np.load('6bit_eval/linear/hp_ber.npy')

linear17i=np.load('6bit_eval/0.00017-linear/ideal_ber.npy')
linear1i=np.load('6bit_eval/0.0001-linear/ideal_ber.npy')
lineari=np.load('6bit_eval/ideal_ber.npy')

# nearest=np.load('6bit_eval/nearest/hp_ber.npy')
# vehicular_mani=np.load('vehicular_manifold/hp_ber.npy')
# vehicular_scal=np.load('vehicular/hp_ber.npy')
plt.yscale('log')
plt.ylabel('BER', fontsize=14)
plt.xlabel(r'$E_{b}/N_{o}(dB)$', fontsize=14)
plt.title('Pedestrian Channel', fontsize=14)
# plt.plot(eb_n0,vehicular_scal, label='Vehicular Scalar Paramesters', marker='+')
# plt.plot(eb_n0, vehicular_mani, label='Vehicular Manifold', marker='o')
# plt.plot(eb_n0,hpang_ber,label='hop_givens_7.5')
# plt.plot(eb_n0, cubic, label='Cubic', marker='o', color='b')
plt.plot(eb_n0,linear, label='3.5e-4', marker='+')
plt.plot(eb_n0,linear17, label='1.7e-4', marker='+')
plt.plot(eb_n0,linear1, label='1e-4', marker='+')

# plt.plot(eb_n0,lineari, label='3.5e-4 ideal', marker='+')
plt.plot(eb_n0,linear17i, label='1.7e-4 ideal', marker='+')
plt.plot(eb_n0,linear1i, label='1e-4 ideal', marker='+')
# plt.plot(eb_n0,nearest, label='nearest', marker='1', color='r')

plt.grid()
plt.legend(fontsize=14.5)
plt.show()

# pylab.rc('text', usetex=True)
# pylab.rc('font', size=45)
# pylab.rc('axes', labelsize=40)
# pylab.rc('legend', fontsize=45)
# pylab.figure(figsize=(8,6))
