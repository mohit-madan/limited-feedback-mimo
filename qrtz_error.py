# Code to generate the generates precoders corresponding to 
# 100 channel evolutions for 10 independent channel realizations
# using both the time based and hopping based schemes 
# (i.e. fill 10 independent 100*`num_subcarriers` 
# time-frequency bins matrix with generated precoders), and
# save them as .npy files
#---------------------------------------------------------------------------
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.interpolate import griddata
import itpp
from mimo_tdl_channel import *
import sys
import signal
import pdb
import time
import copy
from sumrate_BER import leakage_analysis, calculate_BER_performance_QPSK,calculate_BER_performance_QAM256,waterfilling
from munch import munchify
import yaml
from config.arguments import parser
    
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)
def main():
    diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
    # channel_type="Vehicular"
    channel_type="Pedestrian"
    args = "config/"+ channel_type + ".yml"
    with open(args, 'r') as f:
        constants = munchify(yaml.load(f)) 

    #---------------------------------------------------------------------------
    #Channel Params
    signal.signal(signal.SIGINT, sigint_handler)
    #number of db values for which logsum rate will be calculated
    num_Cappar=7
    avg_hpcap=np.zeros(num_Cappar)
    avg_otcap=np.zeros(num_Cappar)
    avg_maxcap=np.zeros(num_Cappar)
    # Eb_N0_dB=np.arange(-6,6,2)

    Eb_N0_dB=np.arange(-3,25,3)
    # Eb_N0_dB=np.arange(-3,36,3)
    #Store BER here
    idealBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
    hpBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
    otBER_QPSK=np.zeros(Eb_N0_dB.shape[0])
    Ts=constants.Ts # 5e-8
    num_subcarriers=constants.num_subcarriers #64
    Nt=constants.Nt #4
    Nr=constants.Nr #2
    freq_quanta=constants.freq_quanta #4 - 5
    time_quanta=constants.time_quanta #1
    
    
    #---------------------------------------------------------------------------
    # Run Time variables
    # Stores the previously known estimates of the subcarriers
    past_vals=constants.past_vals #3
    fdts=constants.fdts
    count=0
    number_simulations=constants.number_simulations
    num_chan=constants.num_chan_realisations #10
    feedback_mats=constants.feedback_mats #8
    freq_jump=(num_subcarriers-1)//(feedback_mats-1)
    fb_indices=freq_jump*np.arange(feedback_mats)
    prev_indices=freq_jump*np.arange(feedback_mats-1)+freq_jump//2
    # prev_indices=freq_jump*np.arange(feedback_mats-1)
    
    net_fb_indices=np.sort(np.append(fb_indices,prev_indices))
    unsorted_net_fb_indices=np.append(fb_indices,prev_indices)

    total_fb_indices=[]
    total_fb_indices=np.array([unsorted_net_fb_indices for i in range(number_simulations//2)]).reshape([-1])
    temp=np.concatenate((np.zeros(np.size(fb_indices)),np.ones(np.size(prev_indices))))
    points=np.array([temp+2*i for i in range(number_simulations//2)]).reshape([-1]).astype(int)
    xs,ys=np.meshgrid(np.arange(num_subcarriers),np.arange(number_simulations))
    points=np.vstack((points,total_fb_indices))
    points=np.stack(points,axis=1)

    shifted_unsorted_net_fb_indices=np.append(prev_indices,fb_indices) 
    shifted_total_fb_indices=[]
    shifted_total_fb_indices=np.array([shifted_unsorted_net_fb_indices for i in range(number_simulations//2)]).reshape([-1])
    shifted_temp=np.concatenate((np.zeros(np.size(prev_indices)),np.ones(np.size(fb_indices))))
    shifted_points=np.array([shifted_temp+2*i for i in range(number_simulations//2)]).reshape([-1]).astype(int)
    shifted_points=np.vstack((shifted_points,shifted_total_fb_indices))
    shifted_points=np.stack(shifted_points,axis=1)
    # pdb.set_trace()
    # points_shifted=

    interp_nbr_vals=2
    steady_start_index=6
    sil_BER=0

    start_time=time.time()
    sigma_cb=np.load('./Codebooks/Independent_qt/sigma_cb_2bits_10000.npy') #do we need this codebook
    hp_Qerr=np.zeros(number_simulations)
    q_error=np.zeros((num_chan, number_simulations))
    chan_offset=0

    for chan_index in range(num_chan):
        tH_allH=np.load('Precoders_generated_new/6bit_'+channel_type+'/'+str(fdts)+'/th_allH_'+str(chan_index+chan_offset)+'.npy')
        tH_allU=np.load('Precoders_generated_new/6bit_'+channel_type+'/'+str(fdts)+'/th_allU_'+str(chan_index+chan_offset)+'.npy')
        allU=np.load('./Precoders_generated_new/6bit_'+channel_type+'/'+str(fdts)+'/allU_'+str(chan_index+chan_offset)+'.npy')


        for simulation_index in range(0,number_simulations):
            print("---------------------------------------------------------------------------")
            print ("Starting Eval sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
            print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
                +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
            
            curr_fb_indices=np.arange(feedback_mats)*2 if simulation_index%2==0 else np.arange(feedback_mats-1)*2+1
            hp_qterr=np.mean([stiefCD(tH_allU[simulation_index][net_fb_indices[i]],allU[simulation_index][net_fb_indices[i]])\
                for i in curr_fb_indices])
            hp_Qerr[simulation_index]=hp_qterr            
            gridz=np.zeros((3,num_subcarriers,2*Nt*Nr-Nr**2))
            # pdb.set_trace()
        q_error[chan_index] = hp_Qerr
    avg_error=np.mean(q_error,axis=0)
    pdb.set_trace()             
if __name__ == '__main__':      
    main()

np.save('ped_qerror_scal_frob.npy', avg_error)
plt.yscale("log")
plt.plot(Eb_N0_dB,hpBER_QPSK, marker='o', label="freq_hop")
plt.plot(Eb_N0_dB,idealBER_QPSK, marker='>', label="ideal")
plt.legend()
plt.ylabel("BER")
plt.xlabel("Eb/N0(dB)")
plt.show()