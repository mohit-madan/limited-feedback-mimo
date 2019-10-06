import numpy as np 
import matplotlib.pyplot as plt 

eb_n0=np.arange(-6,20,3)

# cubic=np.load('6bit_eval/hp_ber.npy')
# linear=np.load('6bit_eval/linear/hp_ber.npy')
# nearest=np.load('6bit_eval/nearest/hp_ber.npy')
vehicular_mani=np.load('vehicular_manifold/hp_ber.npy')
vehicular_scal=np.load('vehicular/hp_ber.npy')
plt.yscale('log')
plt.ylabel('BER')
plt.xlabel(r'$E_{b}/N_{o}(dB)$')
plt.title('Pedestrian Channel')
plt.plot(eb_n0,vehicular_scal, label='Vehicular Scalar Paramesters', marker='+')
plt.plot(eb_n0, vehicular_mani, label='Vehicular Manifold', marker='o')
# plt.plot(eb_n0,hpang_ber,label='hop_givens_7.5')
# plt.plot(eb_n0, cubic, label='Cubic', marker='o', color='b')
# plt.plot(eb_n0,linear, label='Linear', marker='+', color='g')
# plt.plot(eb_n0,nearest, label='nearest', marker='1', color='r')

plt.grid()
plt.legend()
plt.show()

