#Import Statements
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
import scipy.fftpack as sci
from scipy import stats
from scipy.interpolate import interp1d 
from scipy.interpolate import InterpolatedUnivariateSpline
import itpp
from mimo_tdl_channel import *
import sys
import signal
import pdb
import time
import copy
import code
from sumrate_BER import leakage_analysis, calculate_BER_performance_QPSK

np.random.seed(81)

def dpcm_encode(qtiz_x_0,x,alpha):
	qtiz_x=np.zeros(np.size(x))
	qtiz_x[0]=qtiz_x_0
	delta=abs(x[1]-x[0])
	beta=np.zeros(np.size(x))
	beta[0]=0
	for i in range(1,len(qtiz_x)-1):
		beta[i]=np.sign(x[i]-qtiz_x[i-1])
		qtiz_x[i] = qtiz_x[i-1] + beta[i]*delta
		if(np.sign(x[i+1]-qtiz_x[i])==beta[i]):
			delta=alpha*delta
		else:
			delta=delta/alpha
	qtiz_x[-1]=qtiz_x[-2]+beta[-1]*delta		
	# pdb.set_trace()		
	return beta,abs(x[1]-x[0])
def dpcm_decode(qtiz_x_0,alpha,beta,delta):
	delta= delta
	qtiz_x=np.zeros(np.size(beta))
	qtiz_x[0]=qtiz_x_0
	for i in range(1,np.size(beta)-1):
		qtiz_x[i]=qtiz_x[i-1]+beta[i]*delta
		if(beta[i+1]==beta[i]):
			delta=delta*alpha
		else:
			delta=delta/alpha
	qtiz_x[-1]=qtiz_x[-2]+beta[-1]*delta		
	return qtiz_x
	
def dpcm_encode_2d(qtiz_x_0, x, alpha1, alpha2):
	qtiz_x=np.zeros(np.shape(x))
	qtiz_x[0][0]=qtiz_x_0
	delta1=abs(x[0][1])	
#---------------------------------------------------------------------------
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)
    
#---------------------------------------------------------------------------
#Lambda Functions
frob_norm= lambda A:np.linalg.norm(A, 'fro')
diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
#---------------------------------------------------------------------------
#Channel Params
signal.signal(signal.SIGINT, sigint_handler)
itpp.RNG_randomize()
# c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Vehicular_A)
c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Pedestrian_A)
Ts=5e-8
num_subcarriers=64
Nt=4
Nr=2
B=6

#---------------------------------------------------------------------------
# Simulation Parameters
number_simulations=100 # equivalent of time
feedback_mats=8
freq_jump=(num_subcarriers-1)//(feedback_mats-1)
freq_quanta=9
time_quanta=1
# for cold start
ran_init=False
# Number of channels simulated for
num_chan_realisations=10
count=0
chan_offset=10
fdts=1e-4

class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
class_obj.set_norm_doppler(fdts)
tH_allH=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
g_theta=np.zeros((number_simulations,num_subcarriers,(2*Nt*Nr-Nr**2)))
un_theta=np.zeros((number_simulations,num_subcarriers,(2*Nt*Nr-Nr**2)))
gap=3
time_theta=np.zeros((number_simulations,num_subcarriers//gap+1))
time_interp_theta=np.zeros((number_simulations,num_subcarriers))
# un2_theta=np.zeros((number_simulations,num_subcarriers,(2*Nt*Nr-Nr**2)))
for simulation_index in range(number_simulations):
	# Print statement to indicate progress
    if(simulation_index%10==0):
        print ("Starting Gen sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
    # Generate Channel matrices with the above specs
    class_obj.generate()
    # Get the channel matrix for num_subcarriers
    H_list=class_obj.get_Hlist()
    tH_allH[simulation_index]=H_list
    V_list, U_list,sigma_list= find_precoder_list(H_list,False)
    allU[simulation_index]=U_list
    g_theta[simulation_index] = np.array([semiunitary_to_givens_vec(i) for i in U_list])
    
un_theta=np.unwrap(np.unwrap(g_theta, axis=1), axis=0)
x=np.arange(0,64,gap)
x1=[i for i in range(64)]
for simulation_index in range(number_simulations//gap+1):
	time_theta[simulation_index*3]=np.array([un_theta[simulation_index*3,carrier*gap,11] for carrier in range(num_subcarriers//gap+1)])
	time_interp_theta[simulation_index*3,:]=InterpolatedUnivariateSpline(x,time_theta[simulation_index*3,:], k=3)([x1]).flatten()
plt.plot(un_theta[0,:,11], label="original")
plt.plot(time_interp_theta[0,:], label="interpolated")
# InterpolatedUnivariateSpline([0,3],time_interp_theta[0,:])
# plt.plot(time_theta[0,:], label="without")
plt.legend()
plt.title("every 3rd cubcarrier")
plt.show()
pdb.set_trace()
xs=np.arange(64)
ys=np.arange(100)
x,y=np.meshgrid(xs,ys)
# z=un2_theta[:,:,6]
# fig=plt.figure()
# ax=plt.axes(projection='3d')

# diff_freq=np.zeros(63)

# diff_freq=np.array([[[un2_theta[k,i+1,j]-un2_theta[k,0,j] for i in range(63)] for j in range(12)] for k in range(100)])
# # plt.plot(un2_theta[0,:,10])

# # x= [sci.dct(diff_freq[i], norm='ortho') for i in range(12)] 
# x= np.array([[sci.dct(diff_freq[k,i], norm='ortho') for i in range(12)] for k in range(100)])
# y=x[:,10,32]
# xi=np.arange(100)
# # Generated linear fit
# slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
# line = slope*xi+intercept

# plt.plot(xi,y,'o', xi, line)
# plt.show()


# !import code; code.interact(local=vars())
z2=un_theta[:,:,11]
# z1=g_theta[:,:,4]
fig1=plt.figure()
ax=plt.axes(projection='3d')
ax.plot_wireframe(x,y,z2,color='black')
# ax.plot_wireframe(x,y,z1,color='black')
fig1.show()

pdb.set_trace()
