import numpy as np 
import matplotlib.pyplot as plt 

eb_n0=np.arange(-6,20,3)

hpang_ber=np.load('eval/hp_ber.npy')
hpmani_ber=np.load('manifold_interp/hp_ber.npy')
hpang_ber6=np.load('6bit_eval/hp_ber.npy')
plt.yscale('log')
plt.ylabel('BER')
plt.xlabel(r'$E_{b}/N_{o}(dB)$')
# plt.plot(eb_n0,hpang_ber,label='hop_givens_7.5')
plt.plot(eb_n0, hpang_ber6, label='Pedestrian Manifold', marker='o')
plt.plot(eb_n0,hpmani_ber,label='Pedestrian Scalar Parameters', marker='+')
plt.grid()
plt.legend()
plt.show()

