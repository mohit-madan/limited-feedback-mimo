import numpy as np 
import matplotlib.pyplot as plt 

eb_n0=np.arange(-6,20,3)

hpang_ber=np.load('eval/hp_ber.npy')
hpmani_ber=np.load('manifold_interp/hp_ber.npy')
hpang_ber6=np.load('6bit_eval/hp_ber.npy')
plt.yscale('log')
plt.plot(eb_n0,hpang_ber,label='hop_givens_7.5')
plt.plot(eb_n0, hpmani_ber, label='hop_manifold')
plt.plot(eb_n0,hpang_ber6,label='hop_givens_6')

plt.legend()
plt.show()

