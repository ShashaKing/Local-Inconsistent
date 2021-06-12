#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
P is MT, QR are both MT
use sampling to do the KL
This version has evidence

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

import sys


from scipy.optimize import minimize
#from opt_clt import get_single_var_marginals



'''
Compute the cross entropy of PlogQ using samples from P
Q is mixtrue of trees
Assume we can always get the Pr(e) from P and Q
Pr(x|e) = Pr(x,e)|Pr(e)
'''
#def compute_cross_entropy_mt_sampling_evid(P, Q, samples, evid_var):
#    evid_LL_P = P.compute_cond_LL_each_datapoint(samples, evid_var)
#    evid_LL_Q = Q.computeLL_each_datapoint(samples)
#    #LL_P_evid = 
#    #print ('P:', np.sum(LL_P))
#    #print ('Q:', np.sum(LL_Q))
#    #print (LL_P.shape)
#    approx_cross_entropy = np.sum(np.exp(evid_LL_P)*evid_LL_Q)
#    return approx_cross_entropy 


def compute_cross_entropy_mt_sampling_evid(P, Q, samples, evid_list):
    LL_P = P.compute_cond_LL_each_datapoint(samples, evid_list)
    LL_Q = Q.compute_cond_LL_each_datapoint(samples, evid_list)
    #print ('P:', np.sum(LL_P))
    #print ('Q:', np.sum(LL_Q))
    #print (LL_P.shape)
    #approx_cross_entropy = np.sum(np.exp(LL_P)*LL_Q)
    #approx_cross_entropy = np.sum((LL_P - LL_Q))
    approx_cross_entropy = np.sum(LL_Q)
    return approx_cross_entropy 




'''
Sample from tree distribution
'''
def sample_from_tree_evid(clt, n_samples, evid, non_evid_var):

    
    #t_vars = tree[0]
    topo_order = clt.topo_order
    parents = clt.parents
    
    #print ('topo_order: ', topo_order)
    n_variables = topo_order.shape[0]
    evid_var =  evid[:,0]
    
    
    
    tree_samples = np.zeros((n_samples, topo_order.shape[0]), dtype = int)
    
    # set the evidence
    for i in range (evid.shape[0]):
        tree_samples[:, evid[i,0]] = evid[i,1]
    
   
    '''
    Compute the posterior distribtuion Pr(xi|par, e) by reconstruct the posterior
    CPT
    P(x|pa,e) = P(x,pa,e)|P(pa,e) = P(x,pa|e)/P(pa|e)
    If pa is in e, then P(x|pa,e) = P(x,e)|P(e) = P(x|e)
    '''
    
    #cpt = clt.instantiation(list(evid))
    jt = JT.JunctionTree()
    jt.learn_structure(clt.topo_order, clt.parents, clt.cond_cpt)
    P_xy_evid =  JT.get_marginal_JT(jt, list(evid), non_evid_var)
        
        
    P_x_evid = np.zeros((non_evid_var.shape[0], 2))
    #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
    
    P_x_evid[:,0] = P_xy_evid[0,:,0,0] + P_xy_evid[0,:,1,0]
    P_x_evid[:,1] = P_xy_evid[0,:,0,1] + P_xy_evid[0,:,1,1]        
    P_x_evid[0,0] = P_xy_evid[1,0,0,0] + P_xy_evid[1,0,1,0]
    P_x_evid[0,1] = P_xy_evid[1,0,0,1] + P_xy_evid[1,0,1,1]
    

    # normalize
    P_xy_given_evid = Util.normalize2d(P_xy_evid)
    P_x_given_evid = Util.normalize1d(P_x_evid)
    P_e = np.sum(P_x_evid[0,:])
    
    P_xy_given_evid_full = np.zeros((n_variables, n_variables, 2,2))
    #print (P_xy_evid_full[non_evid_var[:,None],non_evid_var].shape)
    P_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_given_evid
    P_x_given_evid_full = np.zeros((n_variables, 2))
    P_x_given_evid_full[non_evid_var,:] = P_x_given_evid
    
    cpt = np.zeros_like(clt.cond_cpt)
    
    for i in range (1,n_variables):
        cld = topo_order[i]
        par = parents[cld]
        
        if cld in evid_var:
            continue
        
        if par in evid_var:
            cpt[i,0,:] = P_x_given_evid_full[cld,0]
            cpt[i,1,:] = P_x_given_evid_full[cld,1]
            continue
        
        cpt[i,0,0] = P_xy_given_evid_full[cld,par,0,0]/P_x_given_evid_full[par,0]
        cpt[i,0,1] = P_xy_given_evid_full[cld,par,0,1]/P_x_given_evid_full[par,1]
        cpt[i,1,0] = P_xy_given_evid_full[cld,par,1,0]/P_x_given_evid_full[par,0]
        cpt[i,1,1] = P_xy_given_evid_full[cld,par,1,1]/P_x_given_evid_full[par,1]
    
    # root
    root = topo_order[0]
    if root not in evid_var:
        cpt[0,0,:] = P_x_given_evid_full[root,0]
        cpt[0,1,:] = P_x_given_evid_full[root,1]
        
    
    #print ('cpt')
    #print (cpt)

    # tree root
    if topo_order[0] not in evid[:,0]:
   
        nums_0_r = int(np.rint(cpt[0,0,0] * n_samples))
        #print nums_0_r
        tree_samples [:nums_0_r, topo_order[0]] = 0
        tree_samples [nums_0_r:, topo_order[0]] = 1
    #print (tree_samples)
    
    #all_vars = np.arange(vars.shape[0])
    for j in range (1, topo_order.shape[0]):
        
        #evidence, do not sample
        if topo_order[j] in evid[:,0]:
            continue
        
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
    #print (tree_samples)
    #print np.sum(tree_samples, axis=0)
    
    
    return tree_samples
        
    


'''
Sample from mixture of trees with evidence
Reject sampling
'''    

def sample_from_mt_evid (mt, n_samples, evids):
    
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
    
    for i in range (evids.shape[0]):
        var = evids[i,0]
        val = evids[i,1]
        ind = np.where(samples[:,var]==val)[0]
        samples = samples[ind]

    # correct the number of samples
    diff = samples.shape[0] - n_samples 
    
    # too many samples
    if diff > 0:
        rand_ind = np.random.randint(samples.shape[0], size=diff)
        
        #print rand_ind
        samples = np.delete(samples, rand_ind, 0)
        #print samples
        
        
    #print (samples.shape)

    #print samples.shape
    return samples



'''
Sample from mixture of trees with evidence
Direct sample from posterior distribution
Check slides in Sampling Algorithmsfor Probablistic Graphical models: Gibbs sampling
''' 
def sample_from_mt_evid_posterior (mt, n_samples, evids, non_evid_var):
    
    samples = []
    

    '''
    Compute Pr(H|e)
    '''
    P_he = np.zeros(mt.n_components)
    for i in range (mt.n_components):
        sub_tree = mt.clt_list[i]
        inst_cpt = sub_tree.instantiation(list(evids))
        P_he[i] = utilM.ve_tree_bin(sub_tree.topo_order, sub_tree.parents, inst_cpt)* mt.mixture_weight[i]
        
    p_h_given_e = P_he/np.sum(P_he)
    
    #print ('before:', mt.mixture_weight)
    #print ('after: ', p_h_given_e)
    
    ''' for each component '''
    for i in range (mt.n_components):
        '''make sure num of samples >= n_samples'''  
        sub_n_samples = int(p_h_given_e[i]*n_samples)+1
        #print ('sub_n_samples: ', sub_n_samples)
        sub_samples = sample_from_tree_evid(mt.clt_list[i],sub_n_samples, evids, non_evid_var)
        samples.append(sub_samples)
        

    samples = np.vstack(samples)
    #samples = np.asanyarray(samples)
    np.random.shuffle(samples)
    
    #print samples
    #print samples.shape
    
#    for i in range (evids.shape[0]):
#        var = evids[i,0]
#        val = evids[i,1]
#        ind = np.where(samples[:,var]==val)[0]
#        samples = samples[ind]

    # correct the number of samples
    diff = samples.shape[0] - n_samples 
    
    # too many samples
    if diff > 0:
        rand_ind = np.random.randint(samples.shape[0], size=diff)
        
        #print rand_ind
        samples = np.delete(samples, rand_ind, 0)
        #print samples
        
        
    #print (samples.shape)

    #print samples.shape
    return samples



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
        print ('sub_n_samples: ', sub_n_samples)
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


'''
marginal P only contains marginals for non-evid variables
'''
# the objective function
def objective(x, mt_R, mt_Q,  marginal_P, evid_list, non_evid_var, n_variables):
    #print (marginal_P)
    #print (pair_marginal_P)
    #n_samples = samples_P.shape[0]
    #n_variables = marginal_P.shape[0]
    #print (marginal_P.shape[0])
    #n_variables = evids.shape[0] + non_evid_var.shape[0]
    n_components = mt_Q.n_components
    
    #print (n_variables)

    
    #print ('weights: ', mt_R.mixture_weight)
    

    lamda = x[0]
    
    marginal_R = np.zeros_like(marginal_P)
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
        
        sub_jt = copy.deepcopy(mt_R.jt_list[c])
        sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0
        sub_jt.set_evidence(evid_list)
        
        
        # the single marginal from jt is using topo order, therefore, need to be re-ordered
        rev_order = np.argsort(mt_R.clt_list[c].topo_order)
        sub_marginal_R = sub_jt.get_single_marginal()[rev_order][non_evid_var]
        #print (sub_marginal_R)
        
        # double check the correctness
#        print (sub_marginal_R)
#        
#        pair_marginal =  JT.get_marginal_JT(sub_jt, evid_list, np.arange(n_variables))
#        
#        temp = mt_R.clt_list[i].instantiation(evid_list)
#        p_xy = mt_R.clt_list[i].inference(temp, np.arange(n_variables))
#        p_x = np.zeros((n_variables, 2))
#        p_x[:,0] = p_xy[0,:,0,0] + p_xy[0,:,1,0]
#        p_x[:,1] = p_xy[0,:,0,1] + p_xy[0,:,1,1]
#        
#        p_x[0,0] = p_xy[1,0,0,0] + p_xy[1,0,1,0]
#        p_x[0,1] = p_xy[1,0,0,1] + p_xy[1,0,1,1]
#        print (p_x)
#        print (sub_marginal_R - p_x)
#        print (pair_marginal- p_xy)
#        sss
              
        # get marginals:
        #marginal_R +=mt_R.mixture_weight[i] * get_single_var_marginals(mt_R.clt_list[i].topo_order, mt_R.clt_list[i].parents, mt_R.clt_list[i].cond_cpt)
        marginal_R += mt_R.mixture_weight[c] * sub_marginal_R
    
    #print (marginal_R)
    # Normalize
    marginal_R = Util.normalize1d(marginal_R)
    #print (marginal_R)

    # first part:
    first_part = lamda*(np.sum(marginal_P*np.log(marginal_R)))
    
    #print ('marginal_R: ', marginal_R)
    
    #print ('first part: ', first_part)
    
    # second part:
    second_part = 0
    for c in range (n_components):
        #rev_order = np.argsort(mt_R.clt_list[i].topo_order)
        #second_part += mt_R.mixture_weight[i]* np.sum(mt_Q.clt_list[i].cond_cpt[rev_order][non_evid_var] *np.log(mt_R.clt_list[i].cond_cpt[rev_order][non_evid_var]))
        second_part += mt_R.mixture_weight[c]* np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
    #print ('aaaa:', second_part)
    sec_part = (1.0-lamda)*second_part
    
    #print ('second part: ', sec_part)
    
    # maximize is the negation of minimize
    #print ('obj value: ', first_part+sec_part)
    
    return -(first_part+sec_part)
    
    
'''    
# the derivative function
marginal P only contains marginals for non-evid variables
'''
def derivative(x, mt_R, mt_Q,  marginal_P, evid_list, non_evid_var, n_variables):

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
        
        sub_jt = copy.deepcopy(mt_R.jt_list[c])
        sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0
        sub_jt.set_evidence(evid_list)
        
        # the single marginal from jt is using topo order, therefore, need to be re-ordered
        rev_order = np.argsort(mt_R.clt_list[c].topo_order)
        
        # get marginals:
        temp_sub_marginal_R = sub_jt.get_single_marginal()[rev_order][non_evid_var]
        sub_marginal_R.append( Util.normalize1d(temp_sub_marginal_R))
        
        marginal_R +=mt_R.mixture_weight[c] * temp_sub_marginal_R
    
    
    marginal_R = Util.normalize1d(marginal_R)
    marginal_P_divide_R = marginal_P/ marginal_R
    #print (marginal_P_divide_R)


    # first part:
    first_part = np.sum(marginal_P*np.log(marginal_R))
    
    # second part:
    second_part = 0
    for c in range (n_components):
        #rev_order = np.argsort(mt_R.clt_list[i].topo_order)
        #second_part += mt_R.mixture_weight[i]* np.sum(mt_Q.clt_list[i].cond_cpt[rev_order][non_evid_var] *np.log(mt_R.clt_list[i].cond_cpt[rev_order][non_evid_var]))
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
        #rev_order = np.argsort(mt_R.clt_list[i].topo_order)
        #der_h=lamda*np.sum(marginal_P_divide_R*sub_marginal_R[i]) +(1-lamda)*np.sum(mt_Q.clt_list[i].cond_cpt[rev_order][non_evid_var] *np.log(mt_R.clt_list[i].cond_cpt[rev_order][non_evid_var]))
        der_h=lamda*np.sum(marginal_P_divide_R*sub_marginal_R[c]) +(1-lamda)*np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
        der_h_arr[c] = der_h
        
        # derivativ of thetas
        der_theta = np.zeros_like(theta)
        
        
        jt.clique_potential = np.copy(theta)
        jt.clique_potential[0,0,1] = jt.clique_potential[0,1,0] = 0
        jt.set_evidence(evid_list)
        # add 1 varialbe in JT
        jt_var = copy.deepcopy(jt)
            
        for ind, var in enumerate (non_evid_var):
    
            new_potential = jt_var.add_query_var(var)
     
            jt_var.propagation(new_potential)
                   
            # normalize
            norm_const=np.einsum('ijkl->i',new_potential)
            new_potential /= norm_const[:,np.newaxis,np.newaxis,np.newaxis]
    
            der_theta[:,:,:] += (marginal_P[ind,0]/marginal_R[ind,0])*(new_potential[:,:,:,0]/theta[:,:,:]) + \
                    (marginal_P[ind,1]/marginal_R[ind,1])*(new_potential[:,:,:,1]/theta[:,:,:])               
            

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
def objective_pair(x, mt_R, mt_Q,  pair_marginal_P, evid_list, non_evid_var, n_variables):
    #print (marginal_P)
    #print (pair_marginal_P)
    #n_samples = samples_P.shape[0]
    #n_variables = pair_marginal_P.shape[0]
    n_components = mt_Q.n_components
    
    #print ('weights: ', mt_R.mixture_weight)
    #print (non_evid_var)
    

    lamda = x[0]
    
    pair_marginal_R = np.zeros_like(pair_marginal_P)
    for c in range (n_components):
        start = c*(4*n_variables+1)+1
        end = start+4*n_variables
        mt_R.mixture_weight[c] = x[start] 
        mt_R.clt_list[c].cond_cpt = x[start+1: end+1].reshape(n_variables,2,2)
        
        
        #ub_jt = mt_R.jt_list[c]
        mt_R.jt_list[c].clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
        mt_R.jt_list[c].clique_potential[0,0,1] = mt_R.jt_list[c].clique_potential[0,1,0] = 0
        #sub_jt.set_evidence(evid_list)
        
#        p_xy =  JT.get_marginal_JT(sub_jt, evid_list, non_evid_var)
#        
#        # get marginals:
#        
#        p_x = np.zeros((non_evid_var.shape[0], 2))
#        #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
#        
#        p_x[:,0] = p_xy[0,:,0,0] + p_xy[0,:,1,0]
#        p_x[:,1] = p_xy[0,:,0,1] + p_xy[0,:,1,1]        
#        p_x[0,0] = p_xy[1,0,0,0] + p_xy[1,0,1,0]
#        p_x[0,1] = p_xy[1,0,0,1] + p_xy[1,0,1,1]
#        
#        #print ('diff: ', np.max(pair_marginal - p_xy))
#        # Normalize        
#        p_x = Util.normalize1d(p_x)
#        
#        for j in xrange (non_evid_var.shape[0]):
#            p_xy[j,j,0,0] = p_x[j,0] - 1e-8
#            p_xy[j,j,1,1] = p_x[j,1] - 1e-8
#            p_xy[j,j,0,1] = 1e-8
#            p_xy[j,j,1,0] = 1e-8
#        
#        #print (np.sum(np.sum(p_xy[0],axis=1),axis =1))
#        #sub_marginal_R.append(p_xy)
#        pair_marginal_R += p_xy * mt_R.mixture_weight[c]
#              
    # get marginals of R:
    pair_marginal_R, temp_marginal_R =mt_R.inference_jt(evid_list,non_evid_var)
    
    #print (pair_marginal_R)

    
    
    # first part:
    first_part = lamda*(np.sum(pair_marginal_P*np.log(pair_marginal_R)))
    
    #print ('marginal_R: ', marginal_R)
    
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
def derivative_pair(x, mt_R, mt_Q,  pair_marginal_P, evid_list, non_evid_var, n_variables):
    
    
    #n_variables = pair_marginal_P.shape[0]
    n_components = mt_Q.n_components
    #ids = np.arange(n_variables)
    der = np.zeros_like(x)
    
    
    #print ('x:',x)

    lamda = x[0]
    
    # The JT potential list that instantiated with evidence
    # save a copy
    #sub_jt_potential_evid = []
    # the probablity of evidence of each sub tree
    #evid_prob_list = []
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
        
        
        #sub_jt.set_evidence(evid_list)
        #sub_jt_potential_evid.append(np.copy(sub_jt.clique_potential))
        #p_xy =  JT.get_marginal_JT(sub_jt, [], non_evid_var)
        p_xy =  JT.get_marginal_JT(sub_jt, evid_list, non_evid_var)
        
        
        #evid_prob_list.append(np.sum(p_xy[0,1]))
        #print (np.sum(p_xy[0,1]))

        #print (np.sum(np.sum(p_xy[0],axis =1), axis =1))

        
        
        p_xy_norm = Util.normalize2d(p_xy)
        p_x = np.zeros((non_evid_var.shape[0], 2))
        #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
        
        p_x[:,0] = p_xy_norm[0,:,0,0] + p_xy_norm[0,:,1,0]
        p_x[:,1] = p_xy_norm[0,:,0,1] + p_xy_norm[0,:,1,1]        
        p_x[0,0] = p_xy_norm[1,0,0,0] + p_xy_norm[1,0,1,0]
        p_x[0,1] = p_xy_norm[1,0,0,1] + p_xy_norm[1,0,1,1]
        
        #print ('diff: ', np.max(pair_marginal - p_xy))
        # Normalize        
        p_x = Util.normalize1d(p_x)
        
        for j in xrange (non_evid_var.shape[0]):
            p_xy_norm[j,j,0,0] = p_x[j,0] - 1e-8
            p_xy_norm[j,j,1,1] = p_x[j,1] - 1e-8
            p_xy_norm[j,j,0,1] = 1e-8
            p_xy_norm[j,j,1,0] = 1e-8
        
#        for j in xrange (non_evid_var.shape[0]):
#            p_xy_norm[j,j,0,0] = 1
#            p_xy_norm[j,j,1,1] = 1
#            p_xy_norm[j,j,0,1] = 1
#            p_xy_norm[j,j,1,0] = 1
#
        sub_marginal_R.append(p_xy_norm)
        pair_marginal_R += p_xy * mt_R.mixture_weight[c]
        #p_xy_all = Util.normalize2d(p_xy_all)
        
    
    pair_marginal_R = Util.normalize2d(pair_marginal_R)
    p_x = np.zeros((non_evid_var.shape[0], 2))
    #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
    
    p_x[:,0] = pair_marginal_R[0,:,0,0] + pair_marginal_R[0,:,1,0]
    p_x[:,1] = pair_marginal_R[0,:,0,1] + pair_marginal_R[0,:,1,1]        
    p_x[0,0] = pair_marginal_R[1,0,0,0] + pair_marginal_R[1,0,1,0]
    p_x[0,1] = pair_marginal_R[1,0,0,1] + pair_marginal_R[1,0,1,1]
    
    #print ('diff: ', np.max(pair_marginal - p_xy))
    # Normalize        
    p_x = Util.normalize1d(p_x)
    
    for j in xrange (non_evid_var.shape[0]):
        pair_marginal_R[j,j,0,0] = p_x[j,0] - 1e-8
        pair_marginal_R[j,j,1,1] = p_x[j,1] - 1e-8
        pair_marginal_R[j,j,0,1] = 1e-8
        pair_marginal_R[j,j,1,0] = 1e-8
#    
    
    
#    for j in xrange (non_evid_var.shape[0]):
#        pair_marginal_R[j,j,0,0] = 1
#        pair_marginal_R[j,j,1,1] = 1
#        pair_marginal_R[j,j,0,1] = 1
#        pair_marginal_R[j,j,1,0] = 1
    
    
    pair_marginal_P_divide_R = pair_marginal_P/ pair_marginal_R
    #print (pair_marginal_P_divide_R.shape)

    


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
        
        
        sub_marginal_R_elem = np.zeros((n_variables, n_variables, 2,2))
        #print (P_xy_evid_full[non_evid_var[:,None],non_evid_var].shape)
        sub_marginal_R_elem[non_evid_var[:,None],non_evid_var] = sub_marginal_R[c]
        
        sub_jt = mt_R.jt_list[c]
        #sub_jt.clique_potential = np.copy(sub_jt_potential_evid[c])
        #sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0

        der_h = 0        
        
        der_h=lamda*np.sum(pair_marginal_P_divide_R*sub_marginal_R[c]) + (1-lamda)*np.sum(mt_Q.clt_list[c].cond_cpt *np.log(mt_R.clt_list[c].cond_cpt))
        der_h_arr[c] = der_h
        
        # derivativ of thetas
        der_theta = np.zeros_like(theta)
        
        
        #jt.clique_potential = np.copy(theta)
        #jt.clique_potential[0,0,1] = jt.clique_potential[0,1,0] = 0
        # add 1 varialbe in JT
        #jt_var = copy.deepcopy(jt)
        
        #temp_tree = copy.deepcopy(sub_tree)
        binary_arr = np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        #print ('evid prob: ',evid_prob_list[c])
        #sub_pxy_list =[]
        for t in non_evid_var: 
            #temp_tree.cond_cpt = np.copy(sub_tree.cond_cpt) # reset
            #t = sub_tree.topo_order[j]
            u = sub_tree.parents[t]
            #print (t,u)
            
            if u == -9999:  # root, skip
                continue
            
            
            '''
            size = 4*nVar*nVar*2*2, 4 represent the 4 values of theta_c|u
            '''
            pxy_regarding_theta = []
            for k in range (binary_arr.shape[0]):
                val_t = binary_arr[k,0]
                val_u = binary_arr[k,1]
                
                evid_theta = copy.copy(evid_list)
                evid_theta.append([t,val_t])
                evid_theta.append([u,val_u])
                
                
#                evid_list_copy = copy.copy(evid_list)
#                evid_list_copy.append([t,val_t])
#                evid_list_copy.append([u,val_u])
#                #print (evid_theta)
#                start = time.time()
#                inst_cpt = sub_tree.instantiation(evid_list_copy)
#                #print ('inst cpt')
#                #print (inst_cpt)                
#                sub_pxy_2 = sub_tree.inference(inst_cpt, non_evid_var)
                #print ('1:', time.time()-start)
                #print (sub_pxy)
                #print ('-----')
                #print (sub_pxy[c,p])
                
                #start2 = time.time()
                #sub_jt.clique_potential = np.copy(sub_jt_potential_evid[c])
                
                #print (sub_jt_potential_evid[c])
                
                sub_jt.clique_potential = np.copy(mt_R.clt_list[c].cond_cpt)
                sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0
                sub_jt.clique_potential[0,0,1] = sub_jt.clique_potential[0,1,0] = 0
#                sub_jt.set_evidence(evid_theta)
                
                sub_pxy = JT.get_marginal_JT(sub_jt, evid_theta, non_evid_var)
                #print ('2:', time.time()-start2)
#                print (np.max(abs(sub_pxy - sub_pxy_2)))
                

                
                #print ('aaa:', np.sum(np.sum(np.sum(sub_pxy, axis = 1), axis=1), axis =1))
                
                #sub_pxy = sub_pxy/evid_prob_list[c]
                
                #print ('bbb:', np.sum(sub_pxy, axis =1))
                #sss
                
                
                #print (sub_pxy.shape)
                
                pxy_regarding_theta.append(sub_pxy)
            
            pxy_regarding_theta_arr = np.asarray(pxy_regarding_theta)
            
            #print (np.sum(pxy_regarding_theta_arr[1],axis=1))
            #print (pxy_regarding_theta_arr.shape)
            
            
            #print ('evid prob: ',evid_prob_list[c])
            #print (np.sum(pxy_regarding_theta_arr))
            
            # Normalize
            norm_const = np.einsum('ijlmn->jl', pxy_regarding_theta_arr)
            
            #print (norm_const[0,1])
            
            #pxy_regarding_theta_arr /= norm_const[np.newaxis,:,:,np.newaxis,np.newaxis]
            pxy_regarding_theta_arr /= norm_const[0,1]
            
            #print (pxy_regarding_theta_arr)
            
       
            
            for y in range(non_evid_var.shape[0]):
                for z in range (y+1, non_evid_var.shape[0]):
                    
                    #  val_c=0, val_u=0
#                    der_theta[t,0,0] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[0,y,z,0,0] + \
#                        pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[0,y,z,0,1]+ \
#                        pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[0,y,z,1,0] + \
#                        pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[0,y,z,1,1])/theta[t,0,0]
#                    
#                    der_theta[t,0,1] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[1,y,z,0,0] + \
#                        pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[1,y,z,0,1]+ \
#                        pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[1,y,z,1,0] + \
#                        pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[1,y,z,1,1])/theta[t,0,1]
#                    
#                    der_theta[t,1,0] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[2,y,z,0,0] + \
#                        pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[2,y,z,0,1]+ \
#                        pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[2,y,z,1,0] + \
#                        pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[2,y,z,1,1])/theta[t,1,0]
#                    
#                    der_theta[t,1,1] += (pair_marginal_P_divide_R[y,z,0,0] * pxy_regarding_theta_arr[3,y,z,0,0] + \
#                        pair_marginal_P_divide_R[y,z,0,1] * pxy_regarding_theta_arr[3,y,z,0,1]+ \
#                        pair_marginal_P_divide_R[y,z,1,0] * pxy_regarding_theta_arr[3,y,z,1,0] + \
#                        pair_marginal_P_divide_R[y,z,1,1] * pxy_regarding_theta_arr[3,y,z,1,1])/theta[t,1,1]
                    
                    
                    
                    der_theta[t,0,0] += (pair_marginal_P_divide_R[y,z,0,0] * (pxy_regarding_theta_arr[0,y,z,0,0]-sub_marginal_R_elem[y,z,0,0]*sub_marginal_R_elem[t,u,0,0]) + \
                        pair_marginal_P_divide_R[y,z,0,1] * (pxy_regarding_theta_arr[0,y,z,0,1]-sub_marginal_R_elem[y,z,0,1]*sub_marginal_R_elem[t,u,0,0])+ \
                        pair_marginal_P_divide_R[y,z,1,0] * (pxy_regarding_theta_arr[0,y,z,1,0]-sub_marginal_R_elem[y,z,1,0]*sub_marginal_R_elem[t,u,0,0]) + \
                        pair_marginal_P_divide_R[y,z,1,1] * (pxy_regarding_theta_arr[0,y,z,1,1]-sub_marginal_R_elem[y,z,1,1]*sub_marginal_R_elem[t,u,0,0]))/theta[t,0,0]
                    
                    der_theta[t,0,1] += (pair_marginal_P_divide_R[y,z,0,0] * (pxy_regarding_theta_arr[1,y,z,0,0]-sub_marginal_R_elem[y,z,0,0]*sub_marginal_R_elem[t,u,0,1]) + \
                        pair_marginal_P_divide_R[y,z,0,1] * (pxy_regarding_theta_arr[1,y,z,0,1]-sub_marginal_R_elem[y,z,0,1]*sub_marginal_R_elem[t,u,0,1])+ \
                        pair_marginal_P_divide_R[y,z,1,0] * (pxy_regarding_theta_arr[1,y,z,1,0]-sub_marginal_R_elem[y,z,1,0]*sub_marginal_R_elem[t,u,0,1]) + \
                        pair_marginal_P_divide_R[y,z,1,1] * (pxy_regarding_theta_arr[1,y,z,1,1]-sub_marginal_R_elem[y,z,1,1]*sub_marginal_R_elem[t,u,0,1]))/theta[t,0,1]
                    
                    der_theta[t,1,0] += (pair_marginal_P_divide_R[y,z,0,0] * (pxy_regarding_theta_arr[2,y,z,0,0]-sub_marginal_R_elem[y,z,0,0]*sub_marginal_R_elem[t,u,1,0]) + \
                        pair_marginal_P_divide_R[y,z,0,1] * (pxy_regarding_theta_arr[2,y,z,0,1]-sub_marginal_R_elem[y,z,0,1]*sub_marginal_R_elem[t,u,1,0])+ \
                        pair_marginal_P_divide_R[y,z,1,0] * (pxy_regarding_theta_arr[2,y,z,1,0]-sub_marginal_R_elem[y,z,1,0]*sub_marginal_R_elem[t,u,1,0]) + \
                        pair_marginal_P_divide_R[y,z,1,1] * (pxy_regarding_theta_arr[2,y,z,1,1]-sub_marginal_R_elem[y,z,1,1]*sub_marginal_R_elem[t,u,1,0]))/theta[t,1,0]
                    
                    der_theta[t,1,1] += (pair_marginal_P_divide_R[y,z,0,0] * (pxy_regarding_theta_arr[3,y,z,0,0]-sub_marginal_R_elem[y,z,0,0]*sub_marginal_R_elem[t,u,1,1]) + \
                        pair_marginal_P_divide_R[y,z,0,1] * (pxy_regarding_theta_arr[3,y,z,0,1]-sub_marginal_R_elem[y,z,0,1]*sub_marginal_R_elem[t,u,1,1])+ \
                        pair_marginal_P_divide_R[y,z,1,0] * (pxy_regarding_theta_arr[3,y,z,1,0]-sub_marginal_R_elem[y,z,1,0]*sub_marginal_R_elem[t,u,1,1]) + \
                        pair_marginal_P_divide_R[y,z,1,1] * (pxy_regarding_theta_arr[3,y,z,1,1]-sub_marginal_R_elem[y,z,1,1]*sub_marginal_R_elem[t,u,1,1]))/theta[t,1,1]
                        
                        
                    
       
 


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
#    dataset_dir = '../dataset/'
#    data_name = 'nltcs'
#    n_components_P = 3
#    n_components = 4
    
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    #n_components_P = int(sys.argv[6])
    mt_dir = sys.argv[6]
    n_components = int(sys.argv[8])
    perturb_rate = float(sys.argv[10])
    e_percent = float(sys.argv[12])
    
    max_iter = 10000
    epsilon = 1e-4
    n_samples = 100000 # number of samples used to do the optimization
    tum_module = data_name
    '''No Noise is required, since sampling have variance, already has noise'''
    

    P_type = 'mt'
    pair = True  # using pairwise marginals
    # test
    
    

    
    
    
    
    
    
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
    
#    evids = np.zeros((7,2), dtype = int)
#    evids[:,0] = np.array([1,2,5,7,9,11,13])
#    evids[:,1] = np.array([1,0,1,0,1,1,1])
#    evid_var =  evids[:,0]
#    non_evid_var = np.setdiff1d(np.arange(n_variables), evid_var)
#    evid_list = list(evids)
#    
#    print ('evid:')
#    print (evids)
#    print ('non evids:')
#    print (non_evid_var)
#    
    
    evids = util_opt.read_evidence_file('../evidence/', e_percent, 'nltcs')
    
    evid_var =  evids[:,0]
    non_evid_var = np.setdiff1d(np.arange(n_variables), evid_var)
    evid_list = list(evids)
    
    evid_flag = np.full(n_variables,-1) #-1 means non evidence
    #evid_arr = np.asarray(evid_list)
    evid_flag[evids[:,0]] = evids[:,1]    
    
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
        
        pair_marginal_P, marginal_P =  model_P.inference(evid_list, non_evid_var)
        
        #print (pair_marginal_P.shape)
        #print (marginal_P)
        

        '''
        Sampling from P
        '''
        #samples_P = sample_from_mt_evid(model_P, n_samples, evids)
        samples_P = sample_from_mt_evid_posterior(model_P, n_samples, evids, non_evid_var)
    
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
    
    
    
    # 10% to generate Q
    #half_data = np.minimum(samples_P[:int(samples_P.shape[0]/10),:],100)
    #half_data = sample_from_mt(model_P, 1000)
    # another 10% used for evaluation
    #print (half_data[:10])
    half_data = train_dataset[:int(train_dataset.shape[0]/10),:]
    
    #eval_data = samples_P[samples_P.shape[0]-int(samples_P.shape[0]/10):,:]
    #eval_data = samples_P[half_data.shape[0]+1:,:]
    eval_data = samples_P
    
    
#    sample_P_non_evid = samples_P[:,non_evid_var]
#    '''statistics of P only consider the non-evid variables'''
#    xycounts_P = Util.compute_xycounts(sample_P_non_evid) + 1 # laplace correction
#    xcounts_P = Util.compute_xcounts(sample_P_non_evid) + 2 # laplace correction
#    pair_marginal_P = Util.normalize2d(xycounts_P)
#    marginal_P = Util.normalize1d(xcounts_P)
    
#    print (marginal_P)
#    
#    print ('---')
#    for var in (non_evid_var):
#        print (model_P.get_node_marginal(list(evids), var))
#
#    print (samples_P.shape[0])
#    sss
#    '''
#    test
#    '''
#    clt = CLT()
#    clt.learnStructure(train_dataset)
#    jt = JT.JunctionTree()
#    jt.learn_structure(clt.topo_order, clt.parents, clt.cond_cpt)
#    P_xy_evid =  JT.get_marginal_JT(jt, list(evids), non_evid_var)
#        
#        
#    P_x_evid = np.zeros((non_evid_var.shape[0], 2))
#    #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
#    
#    P_x_evid[:,0] = P_xy_evid[0,:,0,0] + P_xy_evid[0,:,1,0]
#    P_x_evid[:,1] = P_xy_evid[0,:,0,1] + P_xy_evid[0,:,1,1]        
#    P_x_evid[0,0] = P_xy_evid[1,0,0,0] + P_xy_evid[1,0,1,0]
#    P_x_evid[0,1] = P_xy_evid[1,0,0,1] + P_xy_evid[1,0,1,1]
#    
#
#    # normalize
#    P_xy_given_evid = Util.normalize2d(P_xy_evid)
#    P_x_given_evid = Util.normalize1d(P_x_evid)
#    
#    print ('-----aaaaa------')
#    print (P_x_given_evid)
#    test_samples = sample_from_tree_evid(clt, 10000, evids, non_evid_var)
#    xcounts_test = Util.compute_xcounts(test_samples) + 2 
#    print ('---')
#    print (Util.normalize1d(xcounts_test)[non_evid_var])
#    
#
#    '''
#    test end
#    '''
    
    
    
    '''
    Q Learn from dataset, should be learned using samples without evidence
    '''
    print ('-------------- Mixture of trees Learn from partial data: (Q) ----------')
    mt_Q = MIXTURE_CLT()
    mt_Q.learnStructure(half_data, n_components)
    mt_Q.EM(half_data, max_iter, epsilon)
    
    
    

    
    
    cross_PP = compute_cross_entropy_mt_sampling_evid (model_P, model_P, eval_data, list(evids))
    
    cross_PQ = compute_cross_entropy_mt_sampling_evid (model_P, mt_Q, eval_data,list(evids))
    
#    print ('P||P:', cross_PP/n_samples)
#    print ()
#    
#    print ('P||Q:', cross_PQ/n_samples)
#    print ()

    
    
    print ('-------------- Mixture of trees Learn Learn from P and Q using samples: (R) ----------')
    mt_R = copy.deepcopy(mt_Q)
    
    
#    '''
#    Find evidence location of each sub tree in Q
#    Q and R share the same structure
#    key: sub tree; value: sub {}
#        key: evid var, value, as c or as p
#    '''
#    evid_var_index_c = {}
#    evid_var_index_p = {}
#    
#    
#    for c in range(mt_R.n_components):
#        temp_tree = mt_R.clt_list[c]
#        #print (temp_tree.parents)
#        #print (temp_tree.topo_order)
#        # initialization
#        evid_var_index_c[c] = {}
#        evid_var_index_p[c] = {}
#    
#        for e in evid_var:  
#            # initialization
#            #evid_var_index_c[c][e] = []
#            #evid_var_index_p[c][e] = []
#
#            #variable as child
#            index_c = np.where(temp_tree.topo_order==e)[0]
#            #print ('c:', index_c)
#            # variable as parent
#            varible_child = np.where(temp_tree.parents ==e)[0]
#            ix = np.isin(temp_tree.topo_order, varible_child)
#            index_p = np.where(ix)
#            #print ('p:', index_p)
#            evid_var_index_c[c][e]=index_c
#            evid_var_index_p[c][e]=index_p
#    
#    #print (evid_var_index_c[0])
#    #print (evid_var_index_p[0])
    
    
    '''construct junction tree list for R'''
    for i in range (n_components):
        jt = JT.JunctionTree()
        sub_tree = mt_R.clt_list[i]
        jt.learn_structure(sub_tree.topo_order, sub_tree.parents, sub_tree.cond_cpt)
        mt_R.jt_list.append(jt)
    
    evid_list = list(evids)
    #print (evid_list)

    if pair == True:
        #args = (mt_R, mt_Q,  pair_marginal_P, evids, non_evid_var, evid_var_index_c, evid_var_index_p)
        args = (mt_R, mt_Q,  pair_marginal_P, evid_list, non_evid_var, n_variables)
    else:   
        #args = (mt_R, mt_Q,  marginal_P, evids, non_evid_var, evid_var_index_c, evid_var_index_p)
        args = (mt_R, mt_Q,  marginal_P, evid_list, non_evid_var, n_variables)
    
    
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
        res = minimize(objective_pair, x0, method='SLSQP', jac=derivative_pair, # without normalization constraint
               options={'ftol': 1e-4, 'disp': True, 'maxiter': 100},
               bounds=bounds, args = args)
    else:
            
        #res = minimize(objective, x0, method='SLSQP', jac=derivative, constraints=normalize_cons,  # with normalization constriant
        res = minimize(objective, x0, method='SLSQP', jac=derivative, # without normalization constraint
               options={'ftol': 1e-4, 'disp': True, 'maxiter': 100},
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

    
    cross_PR = compute_cross_entropy_mt_sampling_evid (model_P, mt_R, eval_data, list(evids))
    print ('P||R:', cross_PR/n_samples)
    
    
    output_rec = np.array([cross_PQ/n_samples, cross_PR/n_samples])
    #print (output_rec.shape)
    output_file = '../output_results/'+data_name+'/mt_e_'+str(e_percent) +'_'+str(perturb_rate)
    with open(output_file, 'a') as f_handle:
        #print ("hahah")
        np.savetxt(f_handle, output_rec.reshape(1,2), fmt='%f', delimiter=',')
    
    #return cross_PQ/n_samples, cross_PR/n_samples

if __name__=="__main__":
    #main_cutset()
    #main_clt()
    start = time.time()
    main_opt_mt()
    print ('Total running time: ', time.time() - start) 
    
    
#    start = time.time()
#    n_times = 5

#    Q_arr =  np.zeros(n_times)
#    R_arr =  np.zeros(n_times)
#    for i in range (n_times):
#        Q_arr[i], R_arr[i] = main_opt_mt('nltcs', 0.05)
#        
#    print ('avg P||Q:', np.sum(Q_arr)/n_times)
#    print ('avg P||R:', np.sum(R_arr)/n_times)
#    print ('Total running time: ', time.time() - start) 