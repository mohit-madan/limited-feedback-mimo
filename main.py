 #Import Statements
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import itpp
from mimo_tdl_channel import *
import sys
import signal
import pdb
import time
from sumrate_BER import leakage_analysis, calculate_BER_performance_QPSK
from munch import munchify
import yaml
from config.arguments import parser

np.random.seed(81)
#---------------------------------------------------------------------------
#SIGINT handler
def sigint_handler(signal, frame):
    #Do something while breaking
    pdb.set_trace()
    sys.exit(0)

def main():
    # channel_type="Pedestrian"
    channel_type="Vehicular"
    args = "config/"+ channel_type + ".yml"
    with open(args, 'r') as f:
        constants = munchify(yaml.load(f)) 
    #---------------------------------------------------------------------------
    #Lambda Functions
    frob_norm= lambda A:np.linalg.norm(A, 'fro')
    diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
    #---------------------------------------------------------------------------
    #Channel Params
    signal.signal(signal.SIGINT, sigint_handler)
    # itpp.RNG_randomize()
    itpp.RNG_reset(81)

    c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Vehicular_A)
    # c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Pedestrian_A)
    Ts=constants.Ts
    num_subcarriers=constants.num_subcarriers
    Nt=constants.Nt
    Nr=constants.Nr
    dphi_size=(2*Nt*Nr-Nr**2+Nr)//2
    gtheta_size=(2*Nt*Nr-Nr**2-Nr)//2
    vec_size=2*Nt*Nr-Nr**2
    B=constants.B
    #---------------------------------------------------------------------------
    # Simulation Parameters
    number_simulations=constants.number_simulations # equivalent of time
    feedback_mats=constants.feedback_mats
    freq_jump=(num_subcarriers-1)//(feedback_mats-1)
    freq_quanta=constants.freq_quanta
    time_quanta=constants.time_quanta
    
    # Number of channels simulated for
    num_chan_realisations=constants.num_chan_realisations
    count=0
    chan_offset=constants.chan_offset
    fdts=constants.fdts
    # pdb.set_trace()
    #---------------------------------------------------------------------------
    # Run Time variables
    norm_fn=constants.norm_fn
    qtiz_rate=constants.qtiz_rate
    # qtiz_rate=1.3

    Q_error_even=np.zeros((number_simulations//2,feedback_mats),dtype=np.float64) #
    Q_error_odd=np.zeros((number_simulations//2,feedback_mats-1),dtype=np.float64) #
    
    onlyt_Q_error=np.zeros((number_simulations,feedback_mats),dtype=np.float64) #
    q_error=np.zeros((10,100))
    # Use Stiefel Chordal Distance as norm
    # norm_fn='stiefCD'
    start_time=time.time()
    save=False
    for chan_index in range(num_chan_realisations):
        print("-----------------------------------------------------------------------")
        print ("Starting Chan Realisation: "+str(chan_index)+" : of "+ str(num_chan_realisations) + " # of total channel realisations for "+str(fdts))
        print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
            +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
        # Variables to store quantised U values for number_simulations
        allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
        allU_vec=np.zeros((number_simulations,num_subcarriers,vec_size))
        onlyt_allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
        #half of the time full vector will be transmitted
        onlyt_allU_vec=np.zeros((number_simulations,num_subcarriers,vec_size))
        # onlyt_allU_vec=np.zeros((number_simulations//2,num_subcarriers,vec_size))
        #half of the time partial vectors will be transmitted
        # part_onlyt_allU_vec=np.zeros((number_simulations//2,num_subcarriers,dphi_size))
        # Variables to store unquantised U and H values for number_simulations
        tH_allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
        tH_allH=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
        # Variables to store quantised sigma values
        interpS=np.zeros((number_simulations,num_subcarriers,Nr),dtype=complex)
        # Variables to store unquantised sigma values
        tHS=np.zeros((number_simulations,num_subcarriers,Nr),dtype=complex)
        del_vec=np.zeros((number_simulations,feedback_mats,vec_size))
        beta_vec=np.zeros((number_simulations, feedback_mats, vec_size))
        onlyt_del_vec=np.zeros((number_simulations,feedback_mats,vec_size))
        onlyt_beta_vec=np.zeros((number_simulations, feedback_mats, vec_size))
        #half of the time partial knowledge will be transmitted
        part_onlyt_del_vec=np.zeros((number_simulations,feedback_mats,dphi_size))
        part_onlyt_beta_vec=np.zeros((number_simulations,feedback_mats,dphi_size))
        tH_allU_vec=np.zeros((number_simulations,num_subcarriers,vec_size))#may not be required in future
        # Generate Channels
        class_obj=MIMO_TDL_Channel(Nt,Nr,c_spec,Ts,num_subcarriers)
        class_obj.set_norm_doppler(fdts)

        # ------------------------------------------------------------------------------
        # main simulation loop for the algorithm
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
            tH_allU[simulation_index]=U_list
            # pdb.set_trace()
            if(simulation_index==0):
                U_vec=unitary_frame_to_givens(U_list, None, False)
            else:
                # U_vec=unitary_frame_to_givens(U_list, prev_Uvec, True)
                U_vec=unitary_frame_to_givens(U_list, tH_allU_vec[:simulation_index], True)
           # pdb.set_trace()            
            tH_allU_vec[simulation_index]=U_vec
            if(simulation_index>0):
                for i in range(feedback_mats):
                    onlyt_test_index=freq_jump*i
                    
                    if(simulation_index==1):
                        onlyt_del_vec[simulation_index][i]=abs(U_vec[onlyt_test_index]\
                        -onlyt_allU_vec[simulation_index-1][onlyt_test_index])/2 #checking if this initializing works
                    else:    
                        onlyt_del_vec[simulation_index][i]=calc_del(U_vec[onlyt_test_index],\
                            onlyt_allU_vec[simulation_index-1][onlyt_test_index], onlyt_beta_vec[simulation_index-1][i]\
                            , qtiz_rate, onlyt_del_vec[simulation_index-1][i])
                    # part_onlyt_del_vec[simulation_index][i]= onlyt_del_vec[simulation_index,i,gtheta_size:]

                    onlyt_oU_vec=U_vec[onlyt_test_index]
                    onlyt_prev_qU_vec=onlyt_allU_vec[simulation_index-1][onlyt_test_index]
                    onlyt_qU_vec, onlyt_beta=dpcm_pred(onlyt_oU_vec,onlyt_prev_qU_vec,onlyt_del_vec[simulation_index][i])
                    
                    # pdb.set_trace()   
                    if(simulation_index%2==0):
                        onlyt_allU_vec[simulation_index][onlyt_test_index]=onlyt_qU_vec
                        if(simulation_index%4==0):
                            onlyt_allU_vec[simulation_index][onlyt_test_index][:gtheta_size]=\
                            onlyt_qU_vec[:gtheta_size]
                            onlyt_allU_vec[simulation_index][onlyt_test_index][gtheta_size:]=\
                            onlyt_allU_vec[simulation_index-1][onlyt_test_index][gtheta_size:]
                    else:
                        # theta params(low var) remain same while phi parameters change for odd time instances
                        onlyt_allU_vec[simulation_index][onlyt_test_index][gtheta_size:]=\
                        onlyt_qU_vec[gtheta_size:]
                        onlyt_allU_vec[simulation_index][onlyt_test_index][:gtheta_size]=\
                        onlyt_allU_vec[simulation_index-1][onlyt_test_index][:gtheta_size]
                    onlyt_beta_vec[simulation_index][i]=onlyt_beta
                    
                    # part_onlyt_allU_vec[simulation_index][onlyt_test_index]=onlyt_qU_vec[gtheta_size:]

                    onlyt_allU[simulation_index][onlyt_test_index]=givens_vec_to_semiunitary(\
                        onlyt_allU_vec[simulation_index][onlyt_test_index],Nt,Nr)
                    onlyt_Q_error[simulation_index][i] = stiefCD(\
                        onlyt_allU[simulation_index][onlyt_test_index],tH_allU[simulation_index][onlyt_test_index])

                    test_index=simulation_index%2*(freq_jump//2) + freq_jump*i
                    interpS[simulation_index][onlyt_test_index]=sigma_list[onlyt_test_index]

                    # ---------------------------hop_pred vectors----------------------------------------
                    if(simulation_index%2==1 and i==feedback_mats-1):
                        continue
                    
                    if(simulation_index==1):
                        # pdb.set_trace()
                        oU_vec=U_vec[test_index]
                        qU_vec=qtiz_func(oU_vec,B,Nt,Nr)
                        allU_vec[simulation_index][test_index]=qU_vec
                        allU[simulation_index][test_index]=givens_vec_to_semiunitary(qU_vec, Nt, Nr)
                        Q_error_odd[simulation_index//2][i] = diff_frob_norm(allU[simulation_index][test_index]\
                        ,tH_allU[simulation_index][test_index]) 
                        continue
                    elif(simulation_index==2 or simulation_index==3):
                        del_vec[simulation_index][i]=abs(U_vec[test_index]\
                        -allU_vec[simulation_index-2][test_index])/2 #checking if this works
                    else:
                        del_vec[simulation_index][i]=calc_del(U_vec[test_index],\
                            allU_vec[simulation_index-2][test_index], beta_vec[simulation_index-2][i]\
                            , qtiz_rate, del_vec[simulation_index-2][i])

                    oU_vec= U_vec[test_index]
                    prev_qU_vec=allU_vec[simulation_index-2][test_index]
                    
                    # if(simulation_index>3):
                    #     if(simulation_index%4==1 or simulation_index%4 ==0):
                    #         del_vec[simulation_index][i][gtheta_size:]=del_vec[simulation_index-2][i][gtheta_size:]
                    #     else:
                    #         del_vec[simulation_index][i][:gtheta_size]=del_vec[simulation_index-2][i][:gtheta_size]

                    qU_vec, beta=dpcm_pred(oU_vec,prev_qU_vec,del_vec[simulation_index][i])
                    # pdb.set_trace()

                    if(simulation_index>3):
                        if(simulation_index%4==1 or simulation_index%4 ==0):
                            allU_vec[simulation_index][test_index][:gtheta_size]=allU_vec[simulation_index-2][test_index][:gtheta_size]
                            allU_vec[simulation_index][test_index][gtheta_size:]=qU_vec[gtheta_size:]
                        # elif(simulation_index%4==0):
                        else:
                            allU_vec[simulation_index][test_index][gtheta_size:]=allU_vec[simulation_index-2][test_index][gtheta_size:]
                            allU_vec[simulation_index][test_index][:gtheta_size]=qU_vec[:gtheta_size]
                    else:
                        # pdb.set_trace()
                        allU_vec[simulation_index][test_index]=qU_vec  
                    beta_vec[simulation_index][i]=beta
                    allU[simulation_index][test_index]=givens_vec_to_semiunitary(qU_vec,Nt,Nr)
                    if(simulation_index%2==0):
                        Q_error_even[simulation_index//2][i] = diff_frob_norm(allU[simulation_index][test_index]\
                             ,tH_allU[simulation_index][test_index]) 
                    else:
                        Q_error_odd[simulation_index//2][i] = diff_frob_norm(allU[simulation_index][test_index]\
                             ,tH_allU[simulation_index][test_index]) 

                if(simulation_index%10==0):
                    # pdb.set_trace()
                    print("---------------------------------------------------------------------------")
                    print("Simulation Index: " +str(simulation_index))
                    print("Hop QT Error: "+str(np.mean(Q_error_even[simulation_index//2]))+\
                    " Only T QT Error: "+str(np.mean(onlyt_Q_error[simulation_index])))
                    print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
                    +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
                
            else:
                for i in range(feedback_mats):
                    # initialization of the complete values
                    onlyt_test_index=freq_jump*i
                    onlyt_oU_vec=U_vec[onlyt_test_index]
                    onlyt_qU_vec=qtiz_func(onlyt_oU_vec,B,Nt,Nr)
                    onlyt_allU_vec[simulation_index][onlyt_test_index]=onlyt_qU_vec                 
                    onlyt_allU[simulation_index][onlyt_test_index]=givens_vec_to_semiunitary(onlyt_qU_vec, Nt, Nr)
                    # pdb.set_trace()
                    onlyt_Q_error[simulation_index][i]=stiefCD(onlyt_allU[simulation_index][onlyt_test_index]\
                        ,tH_allU[simulation_index][onlyt_test_index])   
                    test_index=simulation_index%2*(freq_jump//2) + freq_jump*i
                    interpS[simulation_index][onlyt_test_index]=sigma_list[onlyt_test_index]
                    if(simulation_index%2==1 and i==feedback_mats-1):
                        continue

                    oU_vec=U_vec[test_index]
                    qU_vec=qtiz_func(oU_vec,B,Nt,Nr)
                    
                    allU_vec[simulation_index][test_index]=qU_vec
                    allU[simulation_index][test_index]=givens_vec_to_semiunitary(qU_vec, Nt, Nr)                    
                    Q_error_even[simulation_index//2][i]=diff_frob_norm(allU[simulation_index][test_index]\
                        ,tH_allU[simulation_index][test_index])
                # pdb.set_trace()

            prev_Uvec=U_vec
            prev_Ulist=U_list
            tHS[simulation_index]=sigma_list
        q_error[chan_index][::2]=np.mean(Q_error_even, axis=1)
        q_error[chan_index][1::2]=np.mean(Q_error_odd, axis=1)
        pdb.set_trace()
            
        if(save==True):
            np.save('Precoders_generated_new/6bit_'+str(channel_type)+'/'+str(fdts)+'/th_allH_'+str(chan_index+chan_offset)+'.npy',tH_allH)
            np.save('Precoders_generated_new/6bit_'+str(channel_type)+'/'+str(fdts)+'/th_allU_'+str(chan_index+chan_offset)+'.npy',tH_allU)
            np.save('Precoders_generated_new/6bit_'+str(channel_type)+'/'+str(fdts)+'/th_allU_vec'+str(chan_index+chan_offset)+'.npy',tH_allU_vec)
            np.save('Precoders_generated_new/6bit_'+str(channel_type)+'/'+str(fdts)+'/thS_'+str(chan_index+chan_offset)+'.npy',tHS)
            np.save('Precoders_generated_new/6bit_'+str(channel_type)+'/'+str(fdts)+'/allU_'+str(chan_index+chan_offset)+'.npy',allU)
            np.save('Precoders_generated_new/6bit_'+str(channel_type)+'/'+str(fdts)+'/allU_vec'+str(chan_index+chan_offset)+'.npy',allU_vec)
            np.save('Precoders_generated_new/6bit_'+str(channel_type)+'/'+str(fdts)+'/interpS_'+str(chan_index+chan_offset)+'.npy',interpS)
    pdb.set_trace()

if __name__ == '__main__':      
    main()
    pdb.set_trace()
    x=np.mean(q_error,axis=0)
    plt.plot(x)
    plt.show()

# plt.plot(onlyt_allU_vec[:,63,6])
# plt.plot(tH_allU_vec[:,63,6])
# plt.show()

# plt.plot(allU_vec[0::2,0,4])
# plt.plot(tH_allU_vec[0::2,0,4])

# plt.show()

# plt.plot(np.mean(Q_error[1::4],axis=1))
# plt.show()