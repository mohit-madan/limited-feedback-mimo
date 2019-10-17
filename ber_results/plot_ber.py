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
eb_n0=np.arange(-3,25,3)

ped_scal=np.load('updated_6bit_eval/pedestrian/0.00035/hp_ber.npy')
ped_mani=np.load('updated_6bit_eval/agrim_manifold_pedestrian/0.00035/hp_ber.npy')
ped_ideal=np.load('updated_6bit_eval/pedestrian/0.00035/ideal_ber.npy')

veh_scal=np.load('updated_6bit_eval/vehicular/0.00035/hp_ber.npy')
veh_mani=np.load('updated_6bit_eval/agrim_manifold_vehicular/0.00035/hp_ber.npy')
veh_ideal=np.load('updated_6bit_eval/agrim_manifold_vehicular/0.00035/ideal_ber.npy')
plt.yscale('log')
plt.ylabel('BER')
plt.xlabel(r'$E_{b}/N_{0}$ (dB)')

plt.plot(eb_n0,ped_scal[:10], 'b--o', lw=3, ms=10, label='Scalar')
plt.plot(eb_n0,ped_mani[:10],'g--o', lw=3, ms=10, label='Manifold')
plt.plot(eb_n0,ped_ideal[:10],'y--o', lw=3, ms=10, label='64 subcarriers - complete')

# plt.plot(eb_n0,veh_scal[:10], 'g--o', lw=3, ms=10, label='Scalar')
# plt.plot(eb_n0,veh_mani,'b--o', lw=3, ms=10, label='Manifold')
# plt.plot(eb_n0,veh_ideal,'y--o', lw=3, ms=10, label='64 subcarriers - complete')


plt.grid(which='major', linewidth='0.5', color='black');
plt.grid(which='minor', linewidth='0.5', color='grey');

plt.legend(fontsize=14.5)
plt.show()