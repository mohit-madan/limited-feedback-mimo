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
	args = parser.parse_args()
	with open(args.constants, 'r') as f:
		constants = munchify(yaml.load(f)) 
	#---------------------------------------------------------------------------
	#Lambda Functions
	frob_norm= lambda A:np.linalg.norm(A, 'fro')
	diff_frob_norm = lambda A,B:np.linalg.norm(A-B, 'fro')
	#---------------------------------------------------------------------------
	#Channel Params
	signal.signal(signal.SIGINT, sigint_handler)
	itpp.RNG_randomize()
	#c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Vehicular_A)
	c_spec=itpp.comm.Channel_Specification(itpp.comm.CHANNEL_PROFILE.ITU_Pedestrian_A)
	Ts=constants.Ts
	num_subcarriers=constants.num_subcarriers
	Nt=constants.Nt
	Nr=constants.Nr
	B=constants.B

	#---------------------------------------------------------------------------
	# Simulation Parameters
	number_simulations=constants.number_simulations # equivalent of time
	feedback_mats=constants.feedback_mats
	freq_jump=(num_subcarriers-1)//(feedback_mats-1)
	freq_quanta=constants.freq_quanta
	time_quanta=constants.time_quanta
	# for cold start
	ran_init=False
	# Number of channels simulated for
	num_chan_realisations=constants.num_chan_realisations
	count=0
	chan_offset=constants.chan_offset
	fdts=constants.fdts
	past_vals=constants.past_vals
	# pdb.set_trace()
	#---------------------------------------------------------------------------
	# Run Time variables
	norm_fn=constants.norm_fn
	qtiz_rate=constants.qtiz_rate
	# Hopping feedback list
	# fb_list=np.zeros((2*feedback_mats-1,past_vals,Nt,Nr),dtype=complex) feedback list is not required
	# Time based feedback list
	onlyt_fb_list=np.zeros((feedback_mats,2*past_vals,Nt,Nr),dtype=complex)
	# Stores the time instant each subcarrier was communicated
	time_coef=np.zeros((2*feedback_mats-1,past_vals))
	# Store the quantized values here
	# Q_error=np.zeros((number_simulations//2,2*feedback_mats-1),dtype=np.float64) #
	onlyt_Q_error=np.zeros((number_simulations,feedback_mats),dtype=np.float64) #
	# Use Stiefel Chordal Distance as norm
	# norm_fn='stiefCD'
	start_time=time.time()
	save=True

	for chan_index in range(num_chan_realisations):
	    print("-----------------------------------------------------------------------")
	    print ("Starting Chan Realisation: "+str(chan_index)+" : of "+ str(num_chan_realisations) + " # of total channel realisations for "+str(fdts))
	    print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
	        +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))
	    # Variables to store quantised U values for number_simulations
	    allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
	    allU_vec=np.zeros((number_simulations,num_subcarriers,2*Nt*Nr-Nr**2))
	    # onlyt_allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
	    # Variables to store unquantised U and H values for number_simulations
	    tH_allU=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
	    tH_allH=np.zeros((number_simulations,num_subcarriers,Nt,Nr),dtype=complex)
	    # Variables to store quantised sigma values
	    interpS=np.zeros((number_simulations,num_subcarriers,Nr),dtype=complex)
	    # Variables to store unquantised sigma values
	    tHS=np.zeros((number_simulations,num_subcarriers,Nr),dtype=complex)
	    # Variable to store delta values for consecutive prediction of all the variables
	    # del_vec[simulation-1] is used to predict allU[simulation]
	    # del_vec=np.full((number_simulations,feedback_mats,2*Nt*Nr-Nr**2),1/2**B-1)
	    del_vec=np.zeros((number_simulations,feedback_mats,2*Nt*Nr-Nr**2))
	    beta_vec=np.zeros((number_simulations, feedback_mats, 2*Nt*Nr-Nr**2))
	    tH_allU_vec=np.zeros((number_simulations,num_subcarriers,2*Nt*Nr-Nr**2))#may not be required in future
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
	        if(simulation_index==0):
		        U_vec=unitary_frame_to_givens(U_list, None, False)
	        else:
	        	U_vec=unitary_frame_to_givens(U_list, prev_Uvec, True)
	        
	        tH_allU_vec[simulation_index]=U_vec
	        if(simulation_index>0):
	        	for i in range(feedback_mats):
		        	test_index=freq_jump*i	        		
		        	if(simulation_index==1):
		        		del_vec[simulation_index][i]=abs(U_vec[test_index]-allU_vec[simulation_index-1][test_index])/2#checking if this works
	        		else:
	        			del_vec[simulation_index][i]=calc_del(U_vec[test_index],\
	        				allU_vec[simulation_index-1][test_index], beta_vec[simulation_index-1][i], qtiz_rate\
	        				, del_vec[simulation_index-1][i])	

	        		oU_vec=U_vec[test_index]
	        		prev_qU_vec=allU_vec[simulation_index-1][test_index]
	        		qU_vec, beta=dpcm_pred(oU_vec,prev_qU_vec,del_vec[simulation_index][i])
	        		# pdb.set_trace()	
	        		allU_vec[simulation_index][test_index]=qU_vec
	        		beta_vec[simulation_index][i]=beta
	        		allU[simulation_index][test_index]=givens_vec_to_semiunitary(qU_vec,Nt,Nr)
	        		interpS[simulation_index][test_index]=sigma_list[test_index]
	        		onlyt_Q_error[simulation_index][i] = stiefCD(allU[simulation_index][test_index]\
	        			,tH_allU[simulation_index][test_index])
        		# pdb.set_trace()
	        else:
	        	for i in range(feedback_mats):
	        		test_index=freq_jump*i
	        		oU_vec=U_vec[test_index]
	        		qU_vec=qtiz_func(oU_vec,B,Nt,Nr)
	        		allU_vec[simulation_index][test_index]=qU_vec
	        		allU[simulation_index][test_index]=givens_vec_to_semiunitary(qU_vec, Nt, Nr)
	        		interpS[simulation_index][test_index]=sigma_list[test_index]
        		# pdb.set_trace()
        	prev_Uvec=U_vec
        	prev_Ulist=U_list
        	# if(simulation_index%2==1):
        	# 	print("---------------------------------------------------------------------------")
        	# 	print("Simulation Index: " +str(simulation_index))
        	# 	print(" Only T QT Error: "+str(np.mean(onlyt_Q_error[simulation_index])))
				# print(str(time.strftime("Elapsed Time %H:%M:%S",time.gmtime(time.time()-start_time)))\
				# +str(time.strftime(" Current Time %H:%M:%S",time.localtime(time.time()))))        
        	
	    if(save==True):
	        np.save('Precoders_generated/Pedestrian8s/'+str(fdts)+'/th_allH_'+str(chan_index+chan_offset)+'.npy',tH_allH)
	        np.save('Precoders_generated/Pedestrian8s/'+str(fdts)+'/th_allU_'+str(chan_index+chan_offset)+'.npy',tH_allU)
	        np.save('Precoders_generated/Pedestrian8s/'+str(fdts)+'/thS_'+str(chan_index+chan_offset)+'.npy',tHS)
	        np.save('Precoders_generated/Pedestrian8s/'+str(fdts)+'/allU_'+str(chan_index+chan_offset)+'.npy',allU)
	        # np.save('Precoders_generated/Pedestrian/'+str(fdts)+'/onlyt_allU_'+str(chan_index+chan_offset)+'.npy',onlyt_allU)
	        np.save('Precoders_generated/Pedestrian8s/'+str(fdts)+'/interpS_'+str(chan_index+chan_offset)+'.npy',interpS)

if __name__ == '__main__':		
	main()


# plt.plot(allU_vec[:,0,5])
# plt.plot(tH_allU_vec[:,0,5])
# plt.show()
