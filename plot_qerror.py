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
eb_n0=np.arange(0,21,3)
stief_veh_scal=np.load('qerror_scal/veh_qerror_stief_scalar.npy')
stief_veh_mani=np.load('qerror_mani/veh_qerror_mani_stief.npy')
frob_veh_scal=np.load('qerror_scal/veh_frob_scala_qerror.npy')
frob_veh_mani=np.load('qerror_mani/veh_qerror_mani_frob.npy')

stief_ped_scal=np.load('qerror_scal/ped_qerror_scal_stief.npy')
stief_ped_mani=np.load('qerror_mani/ped_qerror_mani_stief.npy')
frob_ped_scal=np.load('qerror_scal/ped_qerror_scal_frob.npy')
frob_ped_mani=np.load('qerror_mani/ped_qerror_mani_frob.npy')

plt.ylabel('Quantization Error Frobenius avg')
plt.xlabel('Time')

# plt.plot(stief_veh_mani,label='Vehicular Manifold')
# plt.plot(stief_veh_scal, label='Vehicular Scalar')
# plt.plot(stief_ped_mani,label='Pedestrian Manifold')
# plt.plot(stief_ped_scal, label='Pedestrian Scalar')

# plt.plot(frob_veh_mani,label='Vehicular Manifold')
# plt.plot(frob_veh_scal, label='Vehicular Scalar')
plt.plot(frob_ped_mani,label='Pedestrian Manifold')
plt.plot(frob_ped_scal, label='Pedestrian Scalar')


plt.grid(which='major', linewidth='0.5', color='black');
plt.grid(which='minor', linewidth='0.5', color='grey');

plt.legend(fontsize=14.5)
plt.show()
