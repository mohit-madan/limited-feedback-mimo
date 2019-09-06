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
    args = parser.parse_args()
    with open(args.constants, 'r') as f:
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
    Eb_N0_dB=np.arange(-6,20,3)
    #Store BER here
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
    xs,ys=np.meshgrid(np.arange(64),np.arange(100))
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
    hp_Qerr=np.zeros(number_simulations-interp_nbr_vals-2)
    ot_Qerr=np.zeros(number_simulations-interp_nbr_vals-2)
    hp_Neterr=np.zeros(number_simulations-interp_nbr_vals-2)
    ot_Neterr=np.zeros(number_simulations-interp_nbr_vals-2)
    chan_offset=0
    gridz_prev=np.zeros((6,num_subcarriers,12))
    for chan_index in range(num_chan):

        tH_allH=np.load('Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/th_allH_'+str(chan_index+chan_offset)+'.npy')
        tH_allU=np.load('Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/th_allU_'+str(chan_index+chan_offset)+'.npy')
        tH_allU_vec=np.load('Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/tH_allU_vec'+str(chan_index+chan_offset)+'.npy')        
        tHS=np.load('Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/thS_'+str(chan_index+chan_offset)+'.npy')
        allU=np.load('./Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/allU_'+str(chan_index+chan_offset)+'.npy')
        allU_vec_copy=np.load('./Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/allU_vec'+str(chan_index+chan_offset)+'.npy')
        onlyt_allU=np.load('./Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/onlyt_allU_'+str(chan_index+chan_offset)+'.npy')
        allU_vec=np.load('./Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/allU_vec'+str(chan_index+chan_offset)+'.npy')
        onlyt_allU_vec=np.load('./Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/onlyt_allU_vec'+str(chan_index+chan_offset)+'.npy')
        # onlyt_allU=np.load('Precoders_generated/Pedestrian/'+str(fdts)+'/allU_'+str(chan_index+chan_offset)+'.npy')
        interpS=np.load('Precoders_generated/6bit_Pedestrian/'+str(fdts)+'/interpS_'+str(chan_index+chan_offset)+'.npy')
        for simulation_index in range(number_simulations-6):
        # for simulation_index in range(0,number_simulations-interp_nbr_vals-2):
            print("---------------------------------------------------------------------------")
            print ("Starting Eval sim: "+str(simulation_index)+" : of "+ str(number_simulations) + " # of total simulations")
            print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
                +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
            curr_fb_indices=np.arange(feedback_mats)*2 if simulation_index%2==0 else np.arange(feedback_mats-1)*2+1
            hp_qterr=np.mean([stiefCD(tH_allU[simulation_index][net_fb_indices[i]],allU[simulation_index][net_fb_indices[i]])\
                for i in curr_fb_indices])
            hp_Qerr[simulation_index]=(chan_index*hp_Qerr[simulation_index]+hp_qterr)/(chan_index+1)
            ot_qterr=np.mean([stiefCD(tH_allU[simulation_index][i],onlyt_allU[simulation_index][i])\
                for i in fb_indices])
            ot_Qerr[simulation_index]=(chan_index*ot_Qerr[simulation_index]+ot_qterr)/(chan_index+1)
            print("Qtisn Error: Hop -> "+str(hp_qterr) + " OnlyT -> "+str(ot_qterr))
            # print(" OnlyT -> "+str(ot_qterr))
            # print("Qtisn Error: Hop -> "+str(hp_qterr))
            # total_fb_indices=[]
            # total_fb_indices=np.array([net_fb_indices for i in range(number_simulations//2)]).reshape([-1])
            # temp=np.concatenate((np.zeros(np.size(fb_indices)),np.ones(np.size(prev_indices))))
            # points=np.array([temp+2*i for i in range(number_simulations//2)]).reshape([-1]).astype(int)
            # xs,ys=np.meshgrid(np.arange(64),np.arange(100))
            # points=np.vstack((points,total_fb_indices))
            # points=np.stack(points,axis=1)
            # pdb.set_trace()
            gridz=np.zeros((6,num_subcarriers,12))
            # pdb.set_trace()
            if(simulation_index%2==0):
                for i in range(12):
                    gridz[:,:,i]=griddata(points[0:45], allU_vec[points[(simulation_index*15+1)//2:(simulation_index*15+1)//2+45,0],\
                        points[(simulation_index*15+1)//2:(simulation_index*15+1)//2+45,1]][:,i], (ys[0:6],\
                            xs[0:6]), method='cubic')

                allU_vec_copy[simulation_index]=gridz[0]
                gridz_prev=np.array(gridz)
                # pdb.set_trace()
            else:
                allU_vec_copy[simulation_index]=gridz_prev[1]

            allU[simulation_index]=givens_frame_to_unitary(allU_vec_copy[simulation_index],Nt,Nr)
            
            for indice_index in range(fb_indices.shape[0]-1):
                # pdb.set_trace()
                curr_freq_index=fb_indices[indice_index]
                next_freq_index=fb_indices[indice_index+1]
                diff_freq=next_freq_index - curr_freq_index
                curr_U=onlyt_allU[simulation_index][curr_freq_index]
                next_U=onlyt_allU[simulation_index][next_freq_index]
                interpolatingT=sH_lift(curr_U,np.matrix(next_U))
                onlyt_allU[simulation_index][curr_freq_index+1:next_freq_index]=[sH_retract(curr_U,(t/diff_freq)*interpolatingT)\
                 for t in range(1,diff_freq)]
                curr_S=interpS[simulation_index][curr_freq_index]
                next_S=interpS[simulation_index][next_freq_index]
                qcurr_S=sigma_cb[np.argmin([la.norm(curr_S-codeword) for codeword in sigma_cb])]
                qnext_S=sigma_cb[np.argmin([la.norm(next_S-codeword) for codeword in sigma_cb])]
                interpS[simulation_index][curr_freq_index+1:next_freq_index]=[(1-(t/diff_freq))*qcurr_S+(t/diff_freq)*qnext_S\
                 for t in range(1,diff_freq)]
            
            
            
            ot_cap=[np.mean(leakage_analysis(tH_allH[simulation_index],tH_allU[simulation_index],\
                onlyt_allU[simulation_index],num_subcarriers,\
                waterfilling(tHS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
                waterfilling(interpS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
                Nt,Nr,ret_abs=True)) for p_dB in 5*np.arange(num_Cappar)]
            max_cap=[np.mean(leakage_analysis(tH_allH[simulation_index],tH_allU[simulation_index],\
                tH_allU[num_subcarriers],num_subcarriers,\
                waterfilling(tHS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
                waterfilling(interpS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
                Nt,Nr,ret_abs=True)) for p_dB in 5*np.arange(num_Cappar)]
            hp_cap=[np.mean(leakage_analysis(tH_allH[simulation_index],tH_allU[simulation_index],\
                allU[simulation_index],num_subcarriers,\
                waterfilling(tHS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
                waterfilling(interpS[simulation_index].flatten(),10**(0.1*p_dB)*num_subcarriers),\
                Nt,Nr,ret_abs=True)) for p_dB in 5*np.arange(num_Cappar)]
            avg_otcap=(count*avg_otcap+np.array(ot_cap))/(count+1)
            avg_hpcap=(count*avg_hpcap+np.array(hp_cap))/(count+1)
            avg_maxcap=(count*avg_maxcap+np.array(max_cap))/(count+1)
            print("Avg. Onlyt Capacity " +str(repr(avg_otcap)))
            print("Avg. Max Capacity "+str(repr(avg_maxcap)))
            print("Avg. Freq Hopping Capacity "+str(repr(avg_hpcap)))
            hp_neterr=np.mean([stiefCD(tH_allU[simulation_index][i],allU[simulation_index][i])\
                for i in range(num_subcarriers)])
            hp_Neterr[simulation_index]=(chan_index*hp_Qerr[simulation_index]+hp_qterr)/(chan_index+1)
            ot_neterr=np.mean([stiefCD(tH_allU[simulation_index][i],onlyt_allU[simulation_index][i])\
                for i in range(num_subcarriers)])
            ot_Neterr[simulation_index]=(chan_index*ot_Qerr[simulation_index]+ot_qterr)/(chan_index+1)
            
            #----------------------------------------------------------------------
            #BER tests
            if(sil_BER==0):
                BER_onlyt_QPSK=np.zeros(Eb_N0_dB.shape[0])
                BER_freqhop_QPSK=np.zeros(Eb_N0_dB.shape[0])
                for i in range(Eb_N0_dB.shape[0]):
                    BER_onlyt_QPSK[i]=calculate_BER_performance_QPSK(np.array(tH_allH[simulation_index]),onlyt_allU[simulation_index],Eb_N0_dB[i])
                    BER_freqhop_QPSK[i]=calculate_BER_performance_QPSK(np.array(tH_allH[simulation_index]),allU[simulation_index],Eb_N0_dB[i])
                    #print(BER_quasigeodesic_list_QAM[i])
                hpBER_QPSK=(count*hpBER_QPSK+BER_freqhop_QPSK)/(count+1)
                otBER_QPSK=(count*otBER_QPSK+BER_onlyt_QPSK)/(count+1)
                print("ot_pred = np."+str(repr(otBER_QPSK)))
                print("hp_pred = np."+str(repr(hpBER_QPSK)))
            
            count=count+1
            # pdb.set_trace()
    pdb.set_trace()             
if __name__ == '__main__':      
    main()
    # pdb.set_trace()
plt.yscale("log")
plt.plot(Eb_N0_dB,hpBER_QPSK, marker='o', label="freq_hop")
plt.plot(Eb_N0_dB,otBER_QPSK, marker='+', label="only_time")
plt.legend()
plt.ylabel("BER")
plt.xlabel("Eb/N0(dB)")
plt.show()
x=np.arange(0,35,5)
ot_achievable_rate=np.divide(avg_otcap,avg_maxcap)
hp_achievable_rate=np.divide(avg_hpcap,avg_maxcap)
plt.plot(x,ot_achievable_rate, marker='o', label="only_time")
plt.plot(x,hp_achievable_rate, marker='+', label="freq_hop")
plt.legend()
plt.show()