#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
P is MT, QR are both MT
use sampling to do the KL
using all pairwise maginals to update every parameter

Apply gussian noise to marginals of P
Q is built from trainning data

"""

from __future__ import print_function
import numpy as np
from Util import *
from opt_clt import *
import utilM
import util_opt
from CLT_class import CLT
from CNET_class import CNET
from MIXTURE_CLT import MIXTURE_CLT, load_mt
import time
import copy
import JT
#from BNET_CLASS import BNET


from scipy.optimize import minimize
from opt_clt import get_single_var_marginals

import sys



'''
Compute the cross entropy of PlogQ using samples from P
Q is mixtrue of trees
'''
def compute_cross_entropy_mt_sampling(P, Q, samples):
    LL_P = P.computeLL_each_datapoint(samples)
    LL_Q = Q.computeLL_each_datapoint(samples)
    #print ('P:', np.sum(LL_P))
    #print ('Q:', np.sum(LL_Q))
    #print (LL_P.shape)
    #approx_cross_entropy = np.sum(np.exp(LL_P)*LL_Q)
    #approx_cross_entropy = np.sum((LL_P - LL_Q))
    approx_cross_entropy = np.sum(LL_Q)
    return approx_cross_entropy 


def pertub_model(model, model_type='mt', percent=0.1):
    
    
    if model_type=='mt':
        
        updated_cpt_list = []
        
        for c in range (model.n_components):
      
            sub_tree =model.clt_list[c]
            topo_order = sub_tree.topo_order
            updated_cpt = np.copy(sub_tree.cond_cpt)
            peturb_no = int(np.round(topo_order.shape[0]* percent))
            #print (peturb_no)
            rand_number = np.random.choice(topo_order.shape[0], size=peturb_no, replace=False)
            
            
            #rand_number[0] = 0
            #print ('rand_number',rand_number)
            
            #print(np.random.choice(topo_order.shape[0], size=peturb_no, replace=False))
        
            rand_decimal = np.random.rand(peturb_no, 2, 2)
            
            #print (rand_decimal)
            #print (np.sum(rand_decimal, axis = 1))
    
            
            # make a valid cpt
            norm_const = np.sum(rand_decimal, axis = 1)
            
            rand_decimal[:,:,0] = rand_decimal[:,:,0]/norm_const[:,0, np.newaxis]
            rand_decimal[:,:,1] = rand_decimal[:,:,1]/norm_const[:,1, np.newaxis]
            
            #print (rand_decimal)
            #print (updated_cpt)
            root = topo_order[0]
            if root in rand_number:
                sum_val = rand_decimal[0,0,0]  + rand_decimal[0,1,1] 
                rand_decimal[0,0,0]  = rand_decimal[0,0,1] = rand_decimal[0,0,0]/sum_val
                rand_decimal[0,1,0]  = rand_decimal[0,1,1] = rand_decimal[0,1,1]/sum_val
                
            
            #print (rand_decimal)
    
            #updated_cpt[rand_number,:,:] = 0.5
            updated_cpt[rand_number,:,:] = rand_decimal
            #print (updated_cpt[rand_number])
            
            updated_cpt_list.append(updated_cpt)
            
        return updated_cpt_list


'''
Sample from tree distribution
'''
def sample_from_tree(clt, n_samples):

    
    #t_vars = tree[0]
    topo_order = clt.topo_order
    parents = clt.parents
    
    #print ('topo_order: ', topo_order)
    
    cpt = np.copy(clt.cond_cpt)
    
    
    tree_samples = np.zeros((n_samples, topo_order.shape[0]), dtype = int)
    
    # tree root

    nums_0_r = int(np.rint(cpt[0,0,0] * n_samples))
    #print nums_0_r
    tree_samples [:nums_0_r, topo_order[0]] = 0
    tree_samples [nums_0_r:, topo_order[0]] = 1
    #print tree_samples
    
    
    
    #all_vars = np.arange(vars.shape[0])
    for j in range (1, topo_order.shape[0]):
        
        
        
        #print '---------------- ', j, ' ---------------'
        t_child = topo_order[j]
        t_parent = parents[t_child]
        
        #print 'c:', t_child
       # print 'p:', t_parent
        
        # find where parent = 0 and parent = 1
        par_0 = np.where(tree_samples[:,t_parent]==0)[0]
        par_1 = np.where(tree_samples[:,t_parent]==1)[0]
        
        #print ('par_0: ', par_0)
        #print ('par_1: ', par_1)
        
 
        num_10 = int(np.round(cpt[j,1,0] * par_0.shape[0], decimals =0))
        num_11 = int(np.round(cpt[j,1,1] * par_1.shape[0], decimals =0))
    
        #num_pa0 = np.round(cpt[j,:,0] * par_0.shape[0], decimals =0)
        #num_pa1 = np.round(cpt[j,:,1] * par_1.shape[0], decimals =0)
        
        
        
        #print num_10, num_11

        
        arr_pa0 = np.zeros(par_0.shape[0],dtype = int)
        arr_pa0[:num_10] = 1
        #print arr_pa0
        np.random.shuffle(arr_pa0)
        #print arr_pa0
        tree_samples[par_0, t_child] = arr_pa0
        #print tree_samples
        
        
        arr_pa1 = np.zeros(par_1.shape[0],dtype = int)
        arr_pa1[:num_11] = 1
        #print arr_pa1
        np.random.shuffle(arr_pa1)
        #print arr_pa1
        tree_samples[par_1, t_child] = arr_pa1
       # print tree_samples
    
    #print ('tree_samples')
    #print tree_samples
    #print np.sum(tree_samples, axis=0)
    
    
    return tree_samples
        
    


'''
Sample from mixture of trees
'''    

def sample_from_mt (mt, n_samples):
    
    samples = []
    
    ''' for each component '''
    for i in range (mt.n_components):
        '''make sure num of samples >= n_samples'''  
        sub_n_samples = int(mt.mixture_weight[i]*n_samples)+1
        #print ('sub_n_samples: ', sub_n_samples)
        sub_samples = sample_from_tree(mt.clt_list[i],sub_n_samples)
        samples.append(sub_samples)
        

    samples = np.vstack(samples)
    #samples = np.asanyarray(samples)
    np.random.shuffle(samples)
    
    #print samples
    #print samples.shape
    
    
    # correct the number of samples
    diff = samples.shape[0] - n_samples 
    
    # too many samples
    if diff > 0:
        rand_ind = np.random.randint(samples.shape[0], size=diff)
        
        #print rand_ind
        samples = np.delete(samples, rand_ind, 0)
        #print samples
        
        
    
    #print samples.shape
    return samples






# the objective function
def objective(x, mt_R, mt_Q,  marginal_P, n_variables):
    #print (marginal_P)
    #print (pair_marginal_P)
    #n_samples = samples_P.shape[0]
    n_variables = marginal_P.shape[0]
    n_components = mt_Q.n_components
    
    #print ('weights: ', mt_R.mixture_weight)
    

    lamda = x[0]
    
    marginal_R = np.zeros_like(marginal_P)
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
              
        # get marginals:
        marginal_R +=mt_R.mixture_weight[c] * get_single_var_marginals(mt_R.clt_list[c].topo_order, mt_R.clt_list[c].parents, mt_R.clt_list[c].cond_cpt)
    
    # first part:
    first_part = lamda*(np.sum(marginal_P*np.log(marginal_R)))
    
    #print ('marginal_R: ', marginal_R)
    
    #print ('first part: ', first_part)
    
    # second part:
    second_part = 0
    for c in range (n_components):
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
    #print ('aaaa:', second_part)
    sec_part = (1.0-lamda)*second_part
    
    #print ('second part: ', sec_part)
    
    # maximize is the negation of minimize
    #print ('obj value: ', first_part+sec_part)
    
    return -(first_part+sec_part)
    
    
    
# the derivative function
def derivative(x, mt_R, mt_Q,  marginal_P, n_variables):

    #n_variables = marginal_P.shape[0]
    n_components = mt_Q.n_components
    der = np.zeros_like(x)
    
    #print ('x:',x)

    lamda = x[0]
    
    
    ''' pre calculation '''
    marginal_R = np.zeros_like(marginal_P)
    sub_marginal_R = []
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
        
        
        # get marginals:
        sub_marginal_R.append( get_single_var_marginals(mt_R.clt_list[c].topo_order, mt_R.clt_list[c].parents, mt_R.clt_list[c].cond_cpt))
        marginal_R +=mt_R.mixture_weight[c] * sub_marginal_R[c]
    
    
    marginal_P_divide_R = marginal_P/ marginal_R
    #print (marginal_P_divide_R)


    # first part:
    first_part = np.sum(marginal_P*np.log(marginal_R))
    
    # second part:
    second_part = 0
    for c in range (n_components):
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
    
    '''deravertive of lamda'''
    der_lam = 0
    der_lam = first_part-second_part   # test, not update lam
    der[0] = der_lam
    
    der_h_arr = np.zeros(n_components)
    '''deravertive of theta, h, For each subtree'''
    for c in range (n_components):
        sub_tree = mt_R.clt_list[c]
        h_weight = mt_R.mixture_weight[c]
        theta = sub_tree.cond_cpt
        jt = mt_R.jt_list[c]
        # dervative of hidden variable H
        der_h = 0        
        
        #der_h=lamda*np.sum(marginal_P_divide_R*sub_marginal_R[i]) 
        der_h=lamda*np.sum(marginal_P_divide_R*sub_marginal_R[c]) +(1-lamda)*np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
        der_h_arr[c] = der_h
        
        # derivativ of thetas
        der_theta = np.zeros_like(theta)
        
        
        jt.clique_potential = np.copy(theta)
        jt.clique_potential[0,0,1] = jt.clique_potential[0,1,0] = 0
        # add 1 varialbe in JT
        jt_var = copy.deepcopy(jt)
            
        for var in range(n_variables):
    
            new_potential = jt_var.add_query_var(var)
     
            jt_var.propagation(new_potential)
                   
            # normalize
            norm_const=np.einsum('ijkl->i',new_potential)
            new_potential /= norm_const[:,np.newaxis,np.newaxis,np.newaxis]
    
            der_theta[:,:,:] += (marginal_P[var,0]/marginal_R[var,0])*(new_potential[:,:,:,0]/theta[:,:,:]) + \
                    (marginal_P[var,1]/marginal_R[var,1])*(new_potential[:,:,:,1]/theta[:,:,:])               
            

        der_theta[:,:,:] = h_weight * (lamda*der_theta[:,:,:]+(1.0-lamda)*(mt_Q.clt_list[c].cond_cpt[:,:,:]/theta[:,:,:]))
        
        
        '''Apply theta_{\bar{b}|a} = 1-theta_{b|a}'''
        # root: special case
        der_theta[0,0,0] -= der_theta[0,1,1]
        der_theta[0,1,1] = -der_theta[0,0,0]
        der_theta[0,0,1] = der_theta[0,0,0]    
        der_theta[0,1,0] = der_theta[0,1,1]
    
        der_theta[1:,0,:] -= der_theta[1:,1,:]
        der_theta[1:,1,:] = -der_theta[1:,0,:]
    
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        der[start] = der_h
        der[start+1: end+1] = der_theta.flatten()
    

    #print (der_theta)    
    '''make h to be sum to 1'''
    der_h_adj = np.sum(der_h_arr)/n_components
    #print ('arr: ', der_h_arr)
    #print ('adj: ', der_h_adj)
    
    for i in range (n_components):
        start = i*(4*n_variables+1)+1
        #print ('before:', der[start])
        der[start] -= der_h_adj
        #print (der[start])
    
    #print (der)
    return der *(-1.0)

'''   
# the objective function use pairwise marginal of P
'''
def objective_pair(x, mt_R, mt_Q,  pair_marginal_P, n_variables):
    #print (marginal_P)
    #print (pair_marginal_P)
    #n_samples = samples_P.shape[0]
    #n_variables = pair_marginal_P.shape[0]
    n_components = mt_Q.n_components
    
    #print ('weights: ', mt_R.mixture_weight)
    

    lamda = x[0]
    
    pair_marginal_R = np.zeros_like(pair_marginal_P)
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
        
        mt_R.jt_list[c].clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        mt_R.jt_list[c].clique_potential[0,0,1] = mt_R.jt_list[c].clique_potential[0,1,0] = 0
              
    # get marginals of R:
    pair_marginal_R, temp_marginal_R =mt_R.inference_jt([],np.arange(n_variables))
    
    
    # first part:
    first_part = lamda*(np.sum(pair_marginal_P*np.log(pair_marginal_R)))
    
    #print ('pair_marginal_R: ', pair_marginal_R)
    
    #print ('first part: ', first_part)
    #print ('aa:', np.sum(pair_marginal_P*np.log(pair_marginal_R)))

    
    # second part:
    second_part = 0
    for c in range (n_components):
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
    #print ('aaaa:', second_part)
    sec_part = (1.0-lamda)*second_part
    
    #print ('second part: ', sec_part)

    
    # maximize is the negation of minimize
    #print ('obj value: ', first_part+sec_part)
    return -(first_part+sec_part)
    
    
'''   
# the derivative function
'''
def derivative_pair(x, mt_R, mt_Q,  pair_marginal_P, n_variables):

    #n_variables = pair_marginal_P.shape[0]
    n_components = mt_Q.n_components
    ids = np.arange(n_variables)
    der = np.zeros_like(x)
    
    #print ('x:',x)

    lamda = x[0]
    
    
    ''' pre calculation '''
    pair_marginal_R = np.zeros_like(pair_marginal_P)
    sub_marginal_R = []
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
        
        
        sub_jt = mt_R.jt_list[c]
        sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0

        p_xy =  JT.get_marginal_JT(sub_jt, [], np.arange(n_variables))
        
        # get marginals:
        
        p_x = np.zeros((n_variables, 2))
        #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
        
        p_x[:,0] = p_xy[0,:,0,0] + p_xy[0,:,1,0]
        p_x[:,1] = p_xy[0,:,0,1] + p_xy[0,:,1,1]        
        p_x[0,0] = p_xy[1,0,0,0] + p_xy[1,0,1,0]
        p_x[0,1] = p_xy[1,0,0,1] + p_xy[1,0,1,1]
        
        #print ('diff: ', np.max(pair_marginal - p_xy))
        # Normalize        
        p_x = Util.normalize1d(p_x)
        
        for j in xrange (ids.shape[0]):
            p_xy[j,j,0,0] = p_x[j,0] - 1e-8
            p_xy[j,j,1,1] = p_x[j,1] - 1e-8
            p_xy[j,j,0,1] = 1e-8
            p_xy[j,j,1,0] = 1e-8
        
        
        sub_marginal_R.append(p_xy)
        pair_marginal_R += p_xy * mt_R.mixture_weight[c]
        #p_xy_all = Util.normalize2d(p_xy_all)
        
    
    
    pair_marginal_P_divide_R = pair_marginal_P/ pair_marginal_R
    #print (pair_marginal_P_divide_R)
    


    # first part:
    first_part = np.sum(pair_marginal_P*np.log(pair_marginal_R))
    #print ('first: ', first_part)

    # second part:
    second_part = 0
    for c in range (n_components):
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
    
    '''deravertive of lamda'''
    der_lam = 0
    der_lam = first_part-second_part   
    der[0] = der_lam
    
    der_h_arr = np.zeros(n_components)
    '''deravertive of theta, h, For each subtree'''
    for c in range (n_components):
        sub_tree = mt_R.clt_list[c]
        h_weight = mt_R.mixture_weight[c]
        theta = sub_tree.cond_cpt
        #jt = mt_R.jt_list[i]
        # dervative of hidden variable H
        
        
        sub_jt = copy.deepcopy(mt_R.jt_list[c])
        sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0
#        sub_single_marginal = util_opt.get_single_var_marginals(sub_tree.topo_order, sub_tree.parents, sub_tree.cond_cpt)
#        sub_edge_marginal = util_opt.get_edge_marginals(sub_tree.topo_order, sub_tree.parents, sub_tree.cond_cpt, sub_single_marginal)

        der_h = 0        
        
        der_h=lamda*np.sum(pair_marginal_P_divide_R*sub_marginal_R[c]) + (1-lamda)*np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
        der_h_arr[c] = der_h
        
        # derivativ of thetas
        der_theta = np.zeros_like(theta)
        
        
#        for i in range (n_variables):
#            t = sub_tree.topo_order[i]
#            u = sub_tree.parents[t]
#            
#            if u !=-9999:
#                der_theta[t,:,:] += pair_marginal_P_divide_R[t,u,:,:] * sub_edge_marginal[t,:,:] / theta[t,:,:]
#            else:
#                der_theta[t,0,0] += pair_marginal_P_divide_R[t,t,0,0] * sub_edge_marginal[t,0,0] / theta[t,0,0]
#                der_theta[t,1,1] += pair_marginal_P_divide_R[t,t,1,1] * sub_edge_marginal[t,1,1] / theta[t,1,1]
#        
        #jt.clique_potential = np.copy(theta)
        #jt.clique_potential[0,0,1] = jt.clique_potential[0,1,0] = 0
        # add 1 varialbe in JT
        #jt_var = copy.deepcopy(jt)
        
        #temp_tree = copy.deepcopy(sub_tree)
        binary_arr = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        #sub_pxy_list =[]
        for j in range (n_variables): 
            #temp_tree.cond_cpt = np.copy(sub_tree.cond_cpt) # reset
            t = sub_tree.topo_order[j]
            u = sub_tree.parents[t]
            #print (t,u)
            
            
            '''
            size = 4*nVar*nVar*2*2, 4 represent the 4 values of theta_c|u
            '''
            pxy_regarding_theta = []
            for k in range (binary_arr.shape[0]):
                val_t = binary_arr[k,0]
                val_u = binary_arr[k,1]
                
                evid_theta = []
                evid_theta.append([t,val_t])
                if u != -9999:
                    evid_theta.append([u,val_u])
                #print (evid_theta)
#                start = time.time()
#                inst_cpt = sub_tree.instantiation(evid_theta)
#                #print ('inst cpt')
#                #print (inst_cpt)                
#                sub_pxy1 = sub_tree.inference(inst_cpt, ids)
#                print ('1:', time.time()-start)
                #print (sub_pxy)
                #print ('-----')
                #print (sub_pxy[c,p])
                
                #start2 = time.time()
                sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
                sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0
                #sub_jt.set_evidence(evid_theta)
                
                sub_pxy = JT.get_marginal_JT(sub_jt, evid_theta, np.arange(n_variables))
                #print ('2:', time.time()-start2)
                #print (np.max(abs(sub_pxy1 - sub_pxy)))
                
                
                pxy_regarding_theta.append(sub_pxy)
            
            pxy_regarding_theta_arr = np.asarray(pxy_regarding_theta)
            for y in range(n_variables):
                for z in range (y+1, n_variables):
                    
                    #  val_c=0, val_u=0
                    der_theta[t,0,0] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[0,y,z,0,0] + \
                        pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[0,y,z,0,1]+ \
                        pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[0,y,z,1,0] + \
                        pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[0,y,z,1,1])/theta[t,0,0]
                    
                    
                    
                    der_theta[t,1,1] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[3,y,z,0,0] + \
                        pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[3,y,z,0,1]+ \
                        pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[3,y,z,1,0] + \
                        pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[3,y,z,1,1])/theta[t,1,1]
                        
                    
                    if u != 9999:
                        der_theta[t,0,1] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[1,y,z,0,0] + \
                            pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[1,y,z,0,1]+ \
                            pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[1,y,z,1,0] + \
                            pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[1,y,z,1,1])/theta[t,0,1]
                    
                        der_theta[t,1,0] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[2,y,z,0,0] + \
                            pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[2,y,z,0,1]+ \
                            pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[2,y,z,1,0] + \
                            pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[2,y,z,1,1])/theta[t,1,0]
                        
                    
       
 


        der_theta[:,:,:] = h_weight * (lamda*der_theta[:,:,:]+(1.0-lamda)*(mt_Q.clt_list[c].cond_cpt[:,:,:]/theta[:,:,:]))
        
        #print (der_theta)

        
        
        '''Apply theta_{\bar{b}|a} = 1-theta_{b|a}'''
        # root: special case
        der_theta[0,0,0] -= der_theta[0,1,1]
        der_theta[0,1,1] = -der_theta[0,0,0]
        der_theta[0,0,1] = der_theta[0,0,0]    
        der_theta[0,1,0] = der_theta[0,1,1]
    
        der_theta[1:,0,:] -= der_theta[1:,1,:]
        der_theta[1:,1,:] = -der_theta[1:,0,:]
    
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        der[start] = der_h
        der[start+1: end+1] = der_theta.flatten()
    

    #print (der_theta)    
    '''make h to be sum to 1'''
    der_h_adj = np.sum(der_h_arr)/n_components
    #print (der_h_arr)
    #print (der_h_adj)
    
    for i in range (n_components):
        start = i*(4*n_variables+1)+1
        #print ('before:', der[start])
        der[start] -= der_h_adj
        #print (der[start])
    
    #print (der)
    return der *(-1.0)
'''

'''

def main_opt_mt():
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    #n_components_P = int(sys.argv[6])
    mt_dir = sys.argv[6]
    perturb_rate = float(sys.argv[8])
    n_components = int(sys.argv[10])
    
    
    max_iter = 100
    max_iter_opt = 100
    epsilon = 1e-4
    n_samples = 100000 # number of samples used to do the optimization
    tum_module = data_name
    '''No Noise is required, since sampling have variance, already has noise'''

    P_type = 'mt'
    pair = True  # using pairwise marginals
    
    
    #train_filename = sys.argv[1]
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    
    
    # test purpsoe
    #train_dataset = train_dataset[:20,:7] # have none value
    #train_dataset = train_dataset[:,:7] # have none value
    
    n_variables = train_dataset.shape[1]
    
    
    if P_type == 'mt':
        '''
        ### Load the trained mixture of clt, consider as P
        '''
        print ('Start reloading MT...')
        #mt_dir =  'mt_output/'
        reload_mix_clt = load_mt(mt_dir, tum_module)
        
        # Set information for MT
        for t in reload_mix_clt.clt_list:
            t.nvariables = n_variables
            # learn the junction tree for each clt
            jt = JT.JunctionTree()
            jt.learn_structure(t.topo_order, t.parents, t.cond_cpt)
            reload_mix_clt.jt_list.append(jt)
        
        # using mixture of trees as P
        model_P = reload_mix_clt
        
        p_xy_all = np.zeros((n_variables, n_variables, 2, 2))
        p_x_all = np.zeros((n_variables, 2))
        for i, jt in enumerate(model_P.jt_list):
            p_xy = JT.get_marginal_JT(jt, [], np.arange(n_variables))
            p_xy_all += p_xy * model_P.mixture_weight[i]


        p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
        p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]
        
        p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
        p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]
        
        
        # Normalize        
        marginal_P = Util.normalize1d(p_x_all)
        
        
        for i in xrange (n_variables):
            p_xy_all[i,i,0,0] = p_x_all[i,0] - 1e-8
            p_xy_all[i,i,1,1] = p_x_all[i,1] - 1e-8
            p_xy_all[i,i,0,1] = 1e-8
            p_xy_all[i,i,1,0] = 1e-8
        
        pair_marginal_P = Util.normalize2d(p_xy_all)
        
        
        '''
        Sampling from P
        '''
        samples_P = sample_from_mt(model_P, n_samples)
    
    elif P_type == 'bn':
        '''
        # Learn BNET as P
        '''
#        order = np.arange(train_dataset.shape[1])
#        np.random.shuffle(order)
#        print("Learning Bayesian Network.....")
#        bnet = BNET()
#        bnet.learnStructure_PE(train_dataset, order, option=1)
#        #print("done")
#        samples_P = bnet.getSamples(n_samples)
#        model_P = bnet

    
    # 10% to generate Q
    #half_data = np.minimum(samples_P[:int(samples_P.shape[0]/10),:],1000)
    half_data = train_dataset[:int(train_dataset.shape[0]/10),:]
    # another 10% used for evaluation
    #eval_data = samples_P[half_data.shape[0]+1:,:]
    eval_data = samples_P

#    '''statistics of P'''
#    xycounts_P = Util.compute_xycounts(samples_P) + 1 # laplace correction
#    xcounts_P = Util.compute_xcounts(samples_P) + 2 # laplace correction
#    pair_marginal_P = Util.normalize2d(xycounts_P)
#    marginal_P = Util.normalize1d(xcounts_P)
#    

    '''
    Get the noise
    '''
    noise_mu = 0
    noise_std = 0.01
    noise_percent = 1
    
#    print ('dataset: ', data_name)
#    print ('mu: ', noise_mu)
#    print ('std: ', noise_std)
#    print ('percent: ', noise_percent)
#    
    pair_marginal_P_blur = util_opt.add_noise (pair_marginal_P, n_variables, noise_mu, noise_std, percent_noise=noise_percent)
    marginal_P_blur = marginal_P
    
    '''
    Q Learn from dataset
    '''
    print ('-------------- Mixture of trees Learn from partial data: (Q) ----------')
    mt_Q = MIXTURE_CLT()
    mt_Q.learnStructure(half_data, n_components)
    mt_Q.EM(half_data, max_iter, epsilon)
    
    
    
    
    if perturb_rate > 0:
        
        #print ('here')
        
        perturbed_list = pertub_model(mt_Q, 'mt', perturb_rate)
        

        for c in range (n_components):
            mt_Q.clt_list[c].cond_cpt= perturbed_list[c]
            
    
    
    cross_PP = compute_cross_entropy_mt_sampling (model_P, model_P, eval_data)
    
    cross_PQ = compute_cross_entropy_mt_sampling (model_P, mt_Q, eval_data)
    
    

    
    print ('-------------- Mixture of trees Learn Learn from P and Q using samples: (R) ----------')
    mt_R = copy.deepcopy(mt_Q)
    
    '''construct junction tree list for R'''
    for i in range (n_components):
        jt = JT.JunctionTree()
        sub_tree = mt_R.clt_list[i]
        jt.learn_structure(sub_tree.topo_order, sub_tree.parents, sub_tree.cond_cpt)
        mt_R.jt_list.append(jt)
    
#    if pair == True:
#        args = (mt_R, mt_Q,  pair_marginal_P)
#    else:   
#        args = (mt_R, mt_Q,  marginal_P)
    
    
    # set the bound for all variables
    bnd = (0.001,0.999)
    n_parm = (4*n_variables+1)*n_components+1 # number of parameters that needs to update
    bounds = [bnd,]*n_parm
    
    x0 = np.zeros(n_parm)
    x0[0] = 0.5  # initial value for lamda
    #print (mt_R.mixture_weight)
    
    for i in range (n_components):
        start = i*(4*n_variables+1)+1
        end = start+4*n_variables
        x0[start] = mt_R.mixture_weight[i]   #mixture weight H
        x0[start+1: end+1] = mt_R.clt_list[i].cond_cpt.flatten()
    
    
    if pair == True:
        
        #pair_marginal_P_no_dup = np.copy(pair_marginal_P)
        pair_marginal_P_no_dup = np.copy(pair_marginal_P_blur)
        # eliminate the duplication
        for i in range (n_variables):
            for j in range (i+1):
                pair_marginal_P_no_dup[i,j] = 0
                
        args = (mt_R, mt_Q,  pair_marginal_P_no_dup, n_variables)
    
        res = minimize(objective_pair, x0, method='SLSQP', jac=derivative_pair, # without normalization constraint
               options={'ftol': 1e-4, 'disp': True, 'maxiter': max_iter_opt},
               bounds=bounds, args = args)
    else:
        
        args = (mt_R, mt_Q,  marginal_P, n_variables)
        #res = minimize(objective, x0, method='SLSQP', jac=derivative, constraints=normalize_cons,  # with normalization constriant
        res = minimize(objective, x0, method='SLSQP', jac=derivative, # without normalization constraint
               options={'ftol': 1e-4, 'disp': True, 'maxiter': max_iter_opt},
               bounds=bounds, args = args)
    #clt_R.cond_cpt = res.x[1:].reshape(nvariables,2,2)
    

    x = res.x
    for i in range (n_components):
        start = i*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[i] = x[start] 
        mt_R.clt_list[i].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
    
    
    print ('P||P:', cross_PP/n_samples)
    print ()
    
    print ('P||Q:', cross_PQ/n_samples)
    print ()

    
    cross_PR = compute_cross_entropy_mt_sampling (model_P, mt_R, eval_data)
    print ('P||R:', cross_PR/n_samples)
    
    
    output_rec = np.array([cross_PQ/n_samples, cross_PR/n_samples])
    #print (output_rec.shape)
    output_file = '../output_results/'+data_name+'/mt_'+str(perturb_rate)
    with open(output_file, 'a') as f_handle:
        #print ("hahah")
        np.savetxt(f_handle, output_rec.reshape(1,2), fmt='%f', delimiter=',')
    
    #return cross_PQ/n_samples, cross_PR/n_samples

if __name__=="__main__":
    #main_cutset()
    #main_clt()
#    start = time.time()
#    main_opt_mt()
#    print ('Total running time: ', time.time() - start) 
    
#    start = time.time()
#    n_times = 5
#    
#    Q_arr =  np.zeros(n_times)
#    R_arr =  np.zeros(n_times)
#    for i in range (n_times):
#        Q_arr[i], R_arr[i] = main_opt_mt('nltcs', 0.4)
#        
#    print ('avg P||Q:', np.sum(Q_arr)/n_times)
#    print ('avg P||R:', np.sum(R_arr)/n_times)
#    print ('Total running time: ', time.time() - start) 
    
    start = time.time()
    main_opt_mt()
    print ('Total running time: ', time.time() - start)