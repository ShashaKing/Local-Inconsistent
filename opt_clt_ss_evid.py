#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:26:43 2020
instead of marginal, we get the sufficiet statistics from P
This version we have evidences

use the common format to update derivative

"""

# optimization problem with chow-liu tree
import numpy as np
from scipy.optimize import minimize
#from scipy.optimize import Bounds  # define the bound
#from scipy import optimize

from CLT_class import CLT
from Util import *
import JT
from MIXTURE_CLT import MIXTURE_CLT, load_mt

import sys
import time
import copy

import util_opt
import utilM





def compute_cross_entropy_mt_sampling_evid(Q, samples, evid_list):
    LL_Q = Q.getWeights(samples)
    #print ('P:', np.sum(LL_P))
    #print ('Q:', np.sum(LL_Q))
    #print (LL_P.shape)
    #approx_cross_entropy = np.sum(np.exp(LL_P)*LL_Q)
    #approx_cross_entropy = np.sum((LL_P - LL_Q))
    
    cond_cpt_evid = Q.instantiation(evid_list)
    evid_prob = utilM.ve_tree_bin(Q.topo_order, Q.parents, cond_cpt_evid) 
    
    #print (np.sum(LL_Q)/samples.shape[0])
    #print (evid_prob)

    approx_cross_entropy = np.sum(LL_Q)/samples.shape[0] - np.log(evid_prob)
    return approx_cross_entropy 



'''
Replace cpt with random numbers
'''
def pertub_model(model, model_type='clt', percent=0.1):
    
    
    if model_type=='clt':
        topo_order = model.topo_order
        #parents = model.parents
        updated_cpt = np.copy(model.cond_cpt)
        peturb_no = int(np.round(topo_order.shape[0]* percent))
        #print (peturb_no)
        #rand_number = np.random.randint(topo_order.shape[0], size=peturb_no)
        rand_number = np.random.choice(topo_order.shape[0], size=peturb_no, replace=False)
        
        
        #rand_number[0] = 0
        #print ('rand_number',rand_number)
    
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
        
        return updated_cpt

#"""
#using theta_{\bar{b}|a} = 1-theta_{b|a}
#Cleaned all the commented code based on version 0704
#"""
#
#def get_single_var_marginals(topo_order, parents, cond_cpt):
#    # get marginals:
#    marginals= np.zeros((topo_order.shape[0],2))
#    #marginal_R[topo_order[0]] = theta[0,:,0]
#    marginals[topo_order[0]] = cond_cpt[0,:,0]
#    for k in range (1,topo_order.shape[0]):
#        c = topo_order[k]
#        p = parents[c]
#        marginals[c] = np.einsum('ij,j->i',cond_cpt[k], marginals[p])
#    
#    return marginals
#
#
## ordered by topo order
#def get_edge_marginals(topo_order, parents, cond_cpt, single_marginal):
#        
#    # edge_marginals ordered by topo order
#    edge_marginals = np.zeros_like(cond_cpt)
#    edge_marginals[0,0,0] = cond_cpt[0,0,0]
#    edge_marginals[0,1,1] = cond_cpt[0,1,1]
#        
#    parents_order = parents[topo_order]
#    topo_marginals = single_marginal[parents_order[1:]]   # the parent marignals, ordered by topo_order 
#        
#    edge_marginals[1:] = np.einsum('ijk,ik->ijk',cond_cpt[1:], topo_marginals)
#
#    return edge_marginals
'''
# compute P(x|e)log(R(x|e))
#
'''
def cross_entropy_evid(P, R, evid_list, non_evid_var):
    cross_entropy = 0
    
    total_n_var = len(evid_list)+non_evid_var.shape[0] #evid+non-evid
    
    # assume P is a tree    
    jt_P = JT.JunctionTree()
    jt_P.learn_structure(P.topo_order, P.parents, P.cond_cpt)

        
    P_xy_evid =  JT.get_marginal_JT(jt_P, evid_list, non_evid_var)
    #print (P_xy_evid.shape)
    #p_xy_norm = Util.normalize2d(p_xy)
    P_x_evid = np.zeros((non_evid_var.shape[0], 2))
    #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
    
    P_x_evid[:,0] = P_xy_evid[0,:,0,0] + P_xy_evid[0,:,1,0]
    P_x_evid[:,1] = P_xy_evid[0,:,0,1] + P_xy_evid[0,:,1,1]        
    P_x_evid[0,0] = P_xy_evid[1,0,0,0] + P_xy_evid[1,0,1,0]
    P_x_evid[0,1] = P_xy_evid[1,0,0,1] + P_xy_evid[1,0,1,1]
    
    # Probablity of evidence according to P
    P_e = np.sum(P_x_evid[0,:])
    #print (P_xy_evid)
    #print (P_x_evid)
    #print (np.sum(P_x_evid, axis = 1))
    #print (P_e)
    
    
    # Probablity of evidence according to R
    cond_cpt_e = R.instantiation(evid_list)
    R_e = utilM.ve_tree_bin(R.topo_order, R.parents, cond_cpt_e)
    #print(R_e)
    
    
    # mark which variable is evidence
    evid_flag = np.full(total_n_var,-1) #-1 means non evidence
    evid_arr = np.asarray(evid_list)
    evid_flag[evid_arr[:,0]] = evid_arr[:,1]    
    #print (evid_flag)
    
    
    P_xy_evid_full = np.zeros((total_n_var, total_n_var, 2,2))
    #print (P_xy_evid_full[non_evid_var[:,None],non_evid_var].shape)
    P_xy_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_evid
    P_x_evid_full = np.zeros((total_n_var, 2))
    P_x_evid_full[non_evid_var,:] = P_x_evid
    
    #print (P_xy_evid_full)
    #print (P_xy_evid)
        
    # root is the special case
#    for i in range (1, R.topo_order.shape[0]):
#        cld = R.topo_order[i]
#        par = R.parents[cld]
#        
#        ind_c = np.where(non_evid_var==cld)[0]
#        ind_p = np.where(non_evid_var==par)[0]
#        val_c =  evid_flag[cld]
#        val_p =  evid_flag[par]
#        print (ind_c, ind_p)
#        # both cld and par are not evid
#        if  val_c ==-1 and val_p ==-1:
#            cross_entropy += np.sum(P_xy_evid[ind_c, ind_p] * np.log(cond_cpt_e[i]))
#        # cld is evidence   
#        elif val_c !=-1 and val_p ==-1:
#            cross_entropy += np.sum(P_x_evid[ind_c] * np.log(cond_cpt_e[i,val_c,:]))
#        # par is evidence   
#        elif val_c ==-1 and val_p !=-1:
#            cross_entropy += np.sum(P_x_evid[ind_p] * np.log(cond_cpt_e[i,:,val_p]))
#        # else both cld and par are evidence
#        else:
#            cross_entropy += P_e * np.log(cond_cpt_e[i,val_c,val_p])
#        
#     
#    print ('cross entropy: ', cross_entropy)   
    
#    cross_entropy = 0
    # root is the special case
    for i in range (1, R.topo_order.shape[0]):
        cld = R.topo_order[i]
        par = R.parents[cld]
        
        val_c =  evid_flag[cld]
        val_p =  evid_flag[par]
        # both cld and par are not evid
        if  val_c ==-1 and val_p ==-1:
            cross_entropy += np.sum(P_xy_evid_full[cld, par] * np.log(cond_cpt_e[i]))
        # cld is evidence   
        elif val_c !=-1 and val_p ==-1:
            cross_entropy += np.sum(P_x_evid_full[cld] * np.log(cond_cpt_e[i,val_c,:]))
        # par is evidence   
        elif val_c ==-1 and val_p !=-1:
            cross_entropy += np.sum(P_x_evid_full[par] * np.log(cond_cpt_e[i,:,val_p]))
        # else both cld and par are evidence
        else:
            cross_entropy += P_e * np.log(cond_cpt_e[i,val_c,val_p])
    
    # root
    R_root = R.topo_order[0]
    val_root = evid_flag[R_root]
    # not evid
    if val_root == -1:
        R_root_marginal = np.array([cond_cpt_e[0,0,0], cond_cpt_e[0,1,1]])
        cross_entropy += np.sum(P_x_evid_full[R_root]* np.log(R_root_marginal))
    else:
        cross_entropy += P_e * np.log(cond_cpt_e[i,val_root,val_root])
        
            
    print ('cross entropy: ', cross_entropy)  
           
    cross_entropy -= P_e * np.log(R_e)
    
    print ('cross entropy: ', cross_entropy)

    
    return cross_entropy

#'''
## compute P(x|e)log(R(x|e)), when marginal from P is given
##
#'''
#def cross_entropy_evid_marginal(P_xy_evid_full, P_x_evid_full, R, evid_list, non_evid_var):
#    cross_entropy = 0
#    
##    total_n_var = len(evid_list)+non_evid_var.shape[0] #evid+non-evid
##    
##    # assume P is a tree    
##    jt_P = JT.JunctionTree()
##    jt_P.learn_structure(P.topo_order, P.parents, P.cond_cpt)
##
##        
##    P_xy_evid =  JT.get_marginal_JT(jt_P, evid_list, non_evid_var)
##    #print (P_xy_evid.shape)
##    #p_xy_norm = Util.normalize2d(p_xy)
##    P_x_evid = np.zeros((non_evid_var.shape[0], 2))
##    #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
##    
##    P_x_evid[:,0] = P_xy_evid[0,:,0,0] + P_xy_evid[0,:,1,0]
##    P_x_evid[:,1] = P_xy_evid[0,:,0,1] + P_xy_evid[0,:,1,1]        
##    P_x_evid[0,0] = P_xy_evid[1,0,0,0] + P_xy_evid[1,0,1,0]
##    P_x_evid[0,1] = P_xy_evid[1,0,0,1] + P_xy_evid[1,0,1,1]
##    
##    # Probablity of evidence according to P
##    P_e = np.sum(P_x_evid[0,:])
##    #print (P_xy_evid)
##    #print (P_x_evid)
##    #print (np.sum(P_x_evid, axis = 1))
##    #print (P_e)
##    
##    
#    # Probablity of evidence according to R
#    cond_cpt_e = R.instantiation(evid_list)
#    R_e = utilM.ve_tree_bin(R.topo_order, R.parents, cond_cpt_e)
#    #print(R_e)
##    
##    
##    # mark which variable is evidence
##    evid_flag = np.full(total_n_var,-1) #-1 means non evidence
##    evid_arr = np.asarray(evid_list)
##    evid_flag[evid_arr[:,0]] = evid_arr[:,1]    
##    #print (evid_flag)
##    
##    
##    P_xy_evid_full = np.zeros((total_n_var, total_n_var, 2,2))
##    #print (P_xy_evid_full[non_evid_var[:,None],non_evid_var].shape)
##    P_xy_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_evid
##    P_x_evid_full = np.zeros((total_n_var, 2))
##    P_x_evid_full[non_evid_var,:] = P_x_evid
##    
##    #print (P_xy_evid_full)
##    #print (P_xy_evid)
##        
##    # root is the special case
###    for i in range (1, R.topo_order.shape[0]):
###        cld = R.topo_order[i]
###        par = R.parents[cld]
###        
###        ind_c = np.where(non_evid_var==cld)[0]
###        ind_p = np.where(non_evid_var==par)[0]
###        val_c =  evid_flag[cld]
###        val_p =  evid_flag[par]
###        print (ind_c, ind_p)
###        # both cld and par are not evid
###        if  val_c ==-1 and val_p ==-1:
###            cross_entropy += np.sum(P_xy_evid[ind_c, ind_p] * np.log(cond_cpt_e[i]))
###        # cld is evidence   
###        elif val_c !=-1 and val_p ==-1:
###            cross_entropy += np.sum(P_x_evid[ind_c] * np.log(cond_cpt_e[i,val_c,:]))
###        # par is evidence   
###        elif val_c ==-1 and val_p !=-1:
###            cross_entropy += np.sum(P_x_evid[ind_p] * np.log(cond_cpt_e[i,:,val_p]))
###        # else both cld and par are evidence
###        else:
###            cross_entropy += P_e * np.log(cond_cpt_e[i,val_c,val_p])
###        
###     
###    print ('cross entropy: ', cross_entropy)   
##    
##    cross_entropy = 0
#    # root is the special case
#    for i in range (1, R.topo_order.shape[0]):
#        cld = R.topo_order[i]
#        par = R.parents[cld]
#        
#        val_c =  evid_flag[cld]
#        val_p =  evid_flag[par]
#        # both cld and par are not evid
#        if  val_c ==-1 and val_p ==-1:
#            cross_entropy += np.sum(P_xy_evid_full[cld, par] * np.log(cond_cpt_e[i]))
#        # cld is evidence   
#        elif val_c !=-1 and val_p ==-1:
#            cross_entropy += np.sum(P_x_evid_full[cld] * np.log(cond_cpt_e[i,val_c,:]))
#        # par is evidence   
#        elif val_c ==-1 and val_p !=-1:
#            cross_entropy += np.sum(P_x_evid_full[par] * np.log(cond_cpt_e[i,:,val_p]))
#        # else both cld and par are evidence
#        else:
#            cross_entropy += P_e * np.log(cond_cpt_e[i,val_c,val_p])
#    
#    # root
#    R_root = R.topo_order[0]
#    val_root = evid_flag[R_root]
#    # not evid
#    if val_root == -1:
#        R_root_marginal = np.array([cond_cpt_e[0,0,0], cond_cpt_e[0,1,1]])
#        cross_entropy += np.sum(P_x_evid_full[R_root]* np.log(R_root_marginal))
#    else:
#        cross_entropy += P_e * np.log(cond_cpt_e[i,val_root,val_root])
#        
#            
#    #print ('cross entropy: ', cross_entropy)  
#           
#    cross_entropy -= P_e * np.log(R_e)
#    
#    #print ('cross entropy: ', cross_entropy)
#
#    
#    return cross_entropy


def cross_entropy_evid_parm(R, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag):
    
    cross_entropy = 0

    
    cond_cpt_e = R.instantiation(evid_list)
    #cond_cpt_e =  np.nan_to_num(cond_cpt_e)
    #R_e = utilM.ve_tree_bin(R.topo_order, R.parents, cond_cpt_e)
    
    #cross_PlogR =  util_opt.compute_cross_entropy_parm(pair_marginal_P,marginal_P, parents, topo_order, theta)
    for i in range (1, R.topo_order.shape[0]):
        cld = R.topo_order[i]
        par = R.parents[cld]
        
        val_c =  evid_flag[cld]
        val_p =  evid_flag[par]
        # both cld and par are not evid
        if  val_c ==-1 and val_p ==-1:
            cross_entropy += np.sum(pair_marginal_P[cld, par] * np.log(cond_cpt_e[i]))
        # cld is evidence   
        elif val_c !=-1 and val_p ==-1:
            cross_entropy += np.sum(marginal_P[cld] * np.log(cond_cpt_e[i,val_c,:]))
        # par is evidence   
        elif val_c ==-1 and val_p !=-1:
            cross_entropy += np.sum(marginal_P[par] * np.log(cond_cpt_e[i,:,val_p]))
        # else both cld and par are evidence
        else:
            cross_entropy += np.log(cond_cpt_e[i,val_c,val_p])
    
    # root
    R_root = R.topo_order[0]
    val_root = evid_flag[R_root]
    # not evid
    if val_root == -1:
        R_root_marginal = np.array([cond_cpt_e[0,0,0], cond_cpt_e[0,1,1]])
        cross_entropy += np.sum(marginal_P[R_root]* np.log(R_root_marginal))
    else:
        cross_entropy += np.log(cond_cpt_e[0,val_root,val_root])
        #cross_entropy += 0
        #print (cond_cpt_e)
    
    #cross_entropy -= np.log(R_e)
    
    return cross_entropy
    

# the objective function
def objective(x, clt_R, cpt_Q, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag):
    #print (marginal_P)
    #print (pair_marginal_P)
    
    n_variable = evid_flag.shape[0]
    lamda = x[0]
    theta = x[1:].reshape(n_variable,2,2)
    
    clt_R.cond_cpt = theta
           
    # get marginals:
    #marginal_R = get_single_var_marginals(topo_order, parents, theta)
    
    # first part
    cross_PlogR = cross_entropy_evid_parm(clt_R, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag)
    first_part = lamda*(cross_PlogR)
    
    #print ('first part: ', first_part)
    
    # second part:
    sec_part = (1.0-lamda)*(np.sum(cpt_Q *np.log(theta)))
    
    # maximize is the negation of minimize
    #print ('obj value: ', first_part+sec_part)
    return -(first_part+sec_part)
    
    
    
# the derivative function
def derivative(x, clt_R, cpt_Q, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag):

    lamda = x[0]
    theta = x[1:].reshape(marginal_P.shape[0],2,2)
    n_variable = evid_flag.shape[0]
    
    
    clt_R.cond_cpt = theta
    #print (pair_marginal_P.shape)

    
    # derivative of lambda
    cross_PlogR =  cross_entropy_evid_parm(clt_R, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag)
    der_lam = cross_PlogR - np.sum(cpt_Q *np.log(theta))
    
    # derivativ of thetas
    der_theta = np.zeros_like(theta)
                
    #cond_cpt_e = clt_R.instantiation(evid_list)    
    jt_R = JT.JunctionTree()
    jt_R.learn_structure(clt_R.topo_order, clt_R.parents, clt_R.cond_cpt)

        
    #R_xy_evid =  JT.get_marginal_JT(jt_R, evid_list, non_evid_var)
    R_xy_evid =  JT.get_marginal_JT(jt_R, evid_list, np.arange(n_variable))
#    print (R_xy_evid)
    #p_xy_norm = Util.normalize2d(p_xy)
    #R_x_evid = np.zeros((non_evid_var.shape[0], 2))
    R_x_evid = np.zeros((n_variable, 2))
    #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
    
    
    R_x_evid[:,0] = R_xy_evid[0,:,0,0] + R_xy_evid[0,:,1,0]
    R_x_evid[:,1] = R_xy_evid[0,:,0,1] + R_xy_evid[0,:,1,1]        
    R_x_evid[0,0] = R_xy_evid[1,0,0,0] + R_xy_evid[1,0,1,0]
    R_x_evid[0,1] = R_xy_evid[1,0,0,1] + R_xy_evid[1,0,1,1]
    
#    print (R_x_evid)
#    sss
    
    R_xy_given_evid = Util.normalize2d(R_xy_evid)
    R_x_given_evid = Util.normalize1d(R_x_evid)
    
#    R_xy_given_evid_full = np.zeros((n_variable, n_variable, 2,2))
#    R_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = R_xy_given_evid
#    R_x_given_evid_full = np.zeros((n_variable, 2))
#    R_x_given_evid_full[non_evid_var,:] = R_x_given_evid
    
    
    R_xy_given_evid_full = R_xy_given_evid
    R_x_given_evid_full = R_x_given_evid
    
    '''P(x,u|e)-R(x,u|e), where (x,u) is one edge in R, ordered in topo_order of R'''
    edge_marginal_diff = np.zeros_like(cpt_Q)
    for i in range (1,n_variable):
        cld = clt_R.topo_order[i]
        par = clt_R.parents[cld]
        
        val_c =  evid_flag[cld]
        val_p =  evid_flag[par]
        # both cld and par are not evid
        if  val_c ==-1 and val_p ==-1:
            edge_marginal_diff[i] = pair_marginal_P[cld, par] *(1- R_xy_given_evid_full[cld, par])
        # cld is evidence   
        elif val_c !=-1 and val_p ==-1:
            #print (R_x_given_evid_full[cld])
            edge_marginal_diff[i,val_c,:] = marginal_P[cld] *(1- R_x_given_evid_full[cld])
        # par is evidence   
        elif val_c ==-1 and val_p !=-1:
            edge_marginal_diff[i,:, val_p] = marginal_P[par] *(1- R_x_given_evid_full[par])
        # else both cld and par are evidence
        else:
            edge_marginal_diff[i, val_c,val_p] = 0
        
        #edge_marginal_P[i+1] = pair_marginal_P[cld, pa]
    
    root = clt_R.topo_order[0]    
    val_root = evid_flag[root]
    # not evid
    if val_root == -1:
        edge_marginal_diff[0,0,:] = marginal_P[root,0]* (1 - R_x_given_evid_full[root,0])
        edge_marginal_diff[0,1,:] = marginal_P[root,1]* (1 - R_x_given_evid_full[root,1])
    else:
        edge_marginal_diff = 0
    
    #edge_marginal_P[0,0,:] = marginal_P[root,0]
    #edge_marginal_P[0,1,:] = marginal_P[root,1]
    #print (edge_marginal_P)
        
    der_theta[:,:,:] = lamda*edge_marginal_diff/theta+(1.0-lamda)*(cpt_Q[:,:,:]/theta[:,:,:])
    
    #print (der_theta)

    '''Apply theta_{\bar{b}|a} = 1-theta_{b|a}'''
    # root: special case
    der_theta[0,0,0] -= der_theta[0,1,1]
    der_theta[0,1,1] = -der_theta[0,0,0]
    der_theta[0,0,1] = der_theta[0,0,0]    
    der_theta[0,1,0] = der_theta[0,1,1]

    der_theta[1:,0,:] -= der_theta[1:,1,:]
    der_theta[1:,1,:] = -der_theta[1:,0,:]
    
    #print ('---')
    #print (der_theta)

        

    der = np.zeros_like(x)
    der[0] = der_lam
    der[1:] = der_theta.flatten() 
    
    return der *(-1.0)


#'''
#Update the parameters of R directly from P
#'''
#def update_S_use_P(P_pair,P_single, S):
#    return Util.compute_conditional_CPT(P_pair, P_single, S.topo_order, S.parents)


#'''
#Add noise to distribution pairwise marignals of P, single variable marginal of P
#Which will cause sum_i Pair(i,j)!=Single(j)
#But it is guarateed that sum Pair(i,j) = 1, Pair(i,j) and Pair(j,i) is related
#'''
#def add_noise (P_pair, P_single, noise_mu, noise_std, percent_noise=0.1):
#    
#    n_var = P_single.shape[0]
#    #percent_noise = 0.1
#    # how many potential function that has noise
#    
#    num_noise = int(n_var*percent_noise)
#    #noise_pair = np.random.choice(n_var*n_var*2*2, size=num_var_noise)
#    
#    #percent_noise = 1
#    #num_var_noise = int(n_var*n_var*2*2* percent_noise)
#    #noise_var = np.random.choice(n_var*n_var*2*2, size=num_var_noise)
#    
#    
#    pair_noise= np.random.normal(loc=noise_mu, scale=noise_std, size=(num_noise,2,2))
#    single_noise = np.random.normal(loc=noise_mu, scale=noise_std, size=(num_noise,2))
#
#    #print (noise.shape)
#    
#    P_pair_noise = np.zeros_like(P_pair)
#    P_single_noise = np.zeros_like(P_single)
#    
#    
#    
#    
#    noise_seq = np.random.choice(num_edges, size=num_pair_noise)
#    #print ('noise_seq:', noise_seq)
#    
#    edges = []
#    for i in range (n_var):
#        for j in range(i+1, n_var):
#            edges.append([i,j])
#    
#    
#    Q_noise = np.copy(P)
#    
#    for k,s in enumerate(noise_seq):
#        
#        [i,j] = edges[s]
#        #print (i,j)       
#        #print (P[i,j])
#        
#        '''apply noise'''
#        Q_noise[i,j] += noise[k]
#        
#        '''Set all value between [0.01 ~ 0.99]'''
#        Q_noise[i,j][Q_noise[i,j] < 0.01] = 0.01
#        Q_noise[i,j][Q_noise[i,j] > 0.99] = 0.99
#        
#        '''normalize'''
#        Q_noise[i,j] /= np.sum(Q_noise[i,j])
#        #print (Q_noise[i,j])
#    
#        '''symetric'''
#
#        Q_noise[j,i,0,0] = Q_noise[i,j,0,0]
#        Q_noise[j,i,0,1] = Q_noise[i,j,1,0]
#        Q_noise[j,i,1,0] = Q_noise[i,j,0,1]
#        Q_noise[j,i,1,1] = Q_noise[i,j,1,1]
#                
#        #print (Q_noise[j,i])
#            
#
##
##    
#    return Q_noise


def main_opt_clt():
#    
    #dataset_dir = sys.argv[2]
    #data_name = sys.argv[4]
    
    #dataset_dir = '../../dataset/'
    #data_name = 'nltcs'
    blur_flag = True
    #n_components_P = 3
    #decimals = 2 # how many decimals left for distribution P
    #tum_module = data_name+'_'+str(n_components_P)
    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    #n_components_P = int(sys.argv[6])
    mt_dir = sys.argv[6]
    perturb_rate = float(sys.argv[8])
    e_percent = float(sys.argv[10])
    
    tum_module = data_name
    n_samples = 100000
    #e_percent = 0.2
    blur_flag = True
    
    print('------------------------------------------------------------------')
    print('Construct CLT using optimization methods')
    print('------------------------------------------------------------------')
    
    
    #train_filename = sys.argv[1]
    train_filename = dataset_dir + data_name + '.ts.data'
    test_filename = dataset_dir + data_name +'.test.data'
    valid_filename = dataset_dir + data_name + '.valid.data'
    
    #out_file = '../module/' + data_name + '.npz'
    train_dataset = np.loadtxt(train_filename, dtype=int, delimiter=',')
    valid_dataset = np.loadtxt(valid_filename, dtype=int, delimiter=',')
    test_dataset = np.loadtxt(test_filename, dtype=int, delimiter=',')
    print ("********* Using Validation / Test Dataset in distribtuion 'P' ************")
    full_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    full_dataset = np.concatenate((full_dataset, test_dataset), axis=0)
    
    n_variables = train_dataset.shape[1]
    
    #n_evids = int(np.round(n_variables * e_percent, decimals = 0))
    
    #rand_var = np.random.choice(n_variables, size=n_evids, replace=False)
    #rand_rec = np.random.randint(train_dataset.shape[0], size = 1)
    #print (rand_var)
    #print (rand_rec)

#    evids = np.zeros((n_evids,2), dtype = int)
#    evids[:,0] = np.array(rand_var)
#    #evids[:,0] = np.array([0,1,2])
#    evids[:,1] = np.array(train_dataset[rand_rec, rand_var])
    
    evids = util_opt.read_evidence_file('../evidence/', e_percent, 'nltcs')
    
    evid_var =  evids[:,0]
    non_evid_var = np.setdiff1d(np.arange(n_variables), evid_var)
    evid_list = list(evids)
    
    evid_flag = np.full(n_variables,-1) #-1 means non evidence
    #evid_arr = np.asarray(evid_list)
    evid_flag[evids[:,0]] = evids[:,1]    
#    print ('evid_flag:', evid_flag)
#    
#    print ('evid:')
#    print (evids)
#    print ('non evids:')
#    print (non_evid_var)
#    print ('evid_list', evid_list)
    
    #print (train_dataset[rand_rec])

    
    P_type = 'mt'
    
    if P_type == 'mt':
        '''
        ### Load the trained mixture of clt, consider as P
        '''
        print ('Start reloading MT...')
        #mt_dir =  '../mt_output/'
        reload_mix_clt = load_mt(mt_dir, tum_module)
        non_evid_size = non_evid_var.shape[0]
        
        # Set information for MT
        for t in reload_mix_clt.clt_list:
            t.nvariables = non_evid_size
            # learn the junction tree for each clt
            jt = JT.JunctionTree()
            jt.learn_structure(t.topo_order, t.parents, t.cond_cpt)
            reload_mix_clt.jt_list.append(jt)
        
        # using mixture of trees as P
        model_P = reload_mix_clt
                
        #p_xy_all = np.zeros((non_evid_size, non_evid_size, 2, 2))
        #p_x_all = np.zeros((non_evid_size, 2))
        p_xy_all = np.zeros((n_variables, n_variables, 2, 2))
        p_x_all = np.zeros((n_variables, 2))
        for i, jt in enumerate(model_P.jt_list):
            p_xy = JT.get_marginal_JT(jt, evid_list, np.arange(n_variables))
            p_xy_all += p_xy * model_P.mixture_weight[i]

        #print (p_xy_all.shape)
        #print (p_xy_all)
    

        p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
        p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]
        
        p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
        p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]
        
        #print (p_x_all)

        # Normalize        
        P_x_given_evid = Util.normalize1d(p_x_all)
        
        P_xy_given_evid = Util.normalize2d(p_xy_all)
        for i in xrange (non_evid_size):
            P_xy_given_evid[i,i,0,0] = p_x_all[i,0] - 1e-8
            P_xy_given_evid[i,i,1,1] = p_x_all[i,1] - 1e-8
            P_xy_given_evid[i,i,0,1] = 1e-8
            P_xy_given_evid[i,i,1,0] = 1e-8
        
        #P_xy_given_evid = Util.normalize2d(p_xy_all)
    else:
    
        print("Learning Chow-Liu Trees on full data ......")
        clt_P = CLT()
        clt_P.learnStructure(full_dataset)
        
        #parents = clt_Q.parents # the structure of the tree is defined by the parents of each variables
        #topo_order = clt_Q.topo_order # the DFS order of the tree
        #cpt_Q =  clt_Q.cond_cpt # The cpts of distribution Q
     
        
    #    marginal_P = clt_P.xprob # the single marginals of P
    #    # get the pairwise marginals of P
    #    jt_P = JT.JunctionTree()
    #    jt_P.learn_structure(clt_P.topo_order, clt_P.parents, clt_P.cond_cpt)
    #    pair_marginal_P = JT.get_marginal_JT(jt_P, [], np.arange(n_variables))
        
        
        jt_P = JT.JunctionTree()
        jt_P.learn_structure(clt_P.topo_order, clt_P.parents, clt_P.cond_cpt)
    
            
        #P_xy_evid =  JT.get_marginal_JT(jt_P, evid_list, non_evid_var)
        P_xy_evid =  JT.get_marginal_JT(jt_P, evid_list, non_evid_var)
        
        #print (P_xy_evid.shape)
        #p_xy_norm = Util.normalize2d(p_xy)
        P_x_evid = np.zeros((non_evid_var.shape[0], 2))
        #p_xy = mt_R.clt_list[c].inference(mt_R.clt_list[c].cond_cpt, ids)
        
        P_x_evid[:,0] = P_xy_evid[0,:,0,0] + P_xy_evid[0,:,1,0]
        P_x_evid[:,1] = P_xy_evid[0,:,0,1] + P_xy_evid[0,:,1,1]        
        P_x_evid[0,0] = P_xy_evid[1,0,0,0] + P_xy_evid[1,0,1,0]
        P_x_evid[0,1] = P_xy_evid[1,0,0,1] + P_xy_evid[1,0,1,1]
        
    
        # normalize
        P_xy_given_evid = Util.normalize2d(P_xy_evid)
        P_x_given_evid = Util.normalize1d(P_x_evid)
    
    #print (P_x_given_evid)
    
    # Use half of the training data to bulid Q
    #half_data = train_dataset[:int(train_dataset.shape[0]/10),:]
    half_data = np.minimum(train_dataset[:int(train_dataset.shape[0]/10),:],1000)
    clt_Q = CLT()
    clt_Q.learnStructure(half_data)
    
    clt_Q.cond_cpt = pertub_model(clt_Q, model_type='clt', percent=perturb_rate)
    
    # Initialize R as P
    clt_R = copy.deepcopy(clt_Q)
    
    
#    '''test'''
#    #P_P = cross_entropy_evid(clt_P, clt_P, list(evids), non_evid_var)
#    P_R = cross_entropy_evid(clt_P, clt_R, list(evids), non_evid_var)
#    #print ('P||P:', P_P)
#    print ('P||R:', P_R)
#    
#    P_xy_given_evid_full = np.zeros((n_variables, n_variables, 2,2))
#    #print (P_xy_evid_full[non_evid_var[:,None],non_evid_var].shape)
#    P_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_given_evid
#    P_x_given_evid_full = np.zeros((n_variables, 2))
#    P_x_given_evid_full[non_evid_var,:] = P_x_given_evid
#    P_e = np.sum(P_x_evid[0,:])
#    P_R2 = cross_entropy_evid_parm(clt_R, P_x_given_evid_full, P_xy_given_evid_full, evid_list, non_evid_var, evid_flag)
#    print ('P||R:', P_R2*P_e)
#    ss
    

    #print( np.sum(clt_P.getWeights(test_dataset)) / test_dataset.shape[0])
    #print( np.sum(clt_Q.getWeights(test_dataset)) / test_dataset.shape[0])
    #print( np.sum(clt_R.getWeights(test_dataset)) / test_dataset.shape[0])
    



#    '''test'''
#    # using extremetly simple example (a chain) to test
#    nvariables = 3
#    topo_order = np.array([0,1,2])
#    parents = np.array([-9999,0,1])    
#    cpt_Q = np.zeros((3,2,2))
#    cpt_Q[0,0,0] = 0.3
#    cpt_Q[0,0,1] = 0.3
#    cpt_Q[0,1,0] = 0.7
#    cpt_Q[0,1,1] = 0.7
#    cpt_Q[1,0,0] = 0.2
#    cpt_Q[1,0,1] = 0.4
#    cpt_Q[1,1,0] = 0.8
#    cpt_Q[1,1,1] = 0.6
#    cpt_Q[2,0,0] = 0.3
#    cpt_Q[2,0,1] = 0.1
#    cpt_Q[2,1,0] = 0.7
#    cpt_Q[2,1,1] = 0.9
#    
#    
#    marginal_P = np.zeros((3,2))
#    marginal_P[0,0]=0.3
#    marginal_P[0,1]=0.7
#    marginal_P[1,0]=0.34
#    marginal_P[1,1]=0.66
#    marginal_P[2,0]=0.168
#    marginal_P[2,1]=0.832
#    
#    
#    cpt_R = np.copy(cpt_Q)
#    cpt_R[0,0,0] = 0.6
#    cpt_R[0,0,1] = 0.6
#    cpt_R[0,1,0] = 0.4
#    cpt_R[0,1,1] = 0.4
#    cpt_R[1,0,0] = 0.8
#    cpt_R[1,0,1] = 0.7
#    cpt_R[1,1,0] = 0.2
#    cpt_R[1,1,1] = 0.3
#    cpt_R[2,0,0] = 0.55
#    cpt_R[2,0,1] = 0.9
#    cpt_R[2,1,0] = 0.45
#    cpt_R[2,1,1] = 0.1
#    '''test end'''
#    

    
    
    # bulid the junction tree
    #jt = JT.JunctionTree()
    #jt.learn_structure(topo_order, parents, cpt_R)
    
    cpt_Q = clt_Q.cond_cpt
    cpt_R = clt_R.cond_cpt
    
    #args = (jt, topo_order, parents, cpt_Q, marginal_P)
    #marginal_P_blur = np.round(marginal_P, decimals = decimals)
    #pair_marginal_P_blur = np.round(pair_marginal_P, decimals = decimals)
    
    
    if blur_flag == True:
        '''apply noise to P'''
        
        '''
        Get the noise
        '''
        noise_mu = 0
        noise_std = 0.01
        noise_percent = 1
        
#        print ('dataset: ', data_name)
#        print ('mu: ', noise_mu)
#        print ('std: ', noise_std)
#        print ('percent: ', noise_percent)
        
        P_xy_given_evid_blur = util_opt.add_noise (P_xy_given_evid, n_variables, noise_mu, noise_std, percent_noise=noise_percent)
        
        #print (P_xy_given_evid.shape)
        #print (P_xy_given_evid_blur.shape)

        #marginal_P_blur = marginal_P
        #print (P_xy_given_evid_blur[0])
        P_xy_given_evid_full = P_xy_given_evid_blur
        P_x_given_evid_full = P_x_given_evid
        
        
        
#        P_xy_given_evid_full = np.zeros((n_variables, n_variables, 2,2))
#        #print (P_xy_evid_full[non_evid_var[:,None],non_evid_var].shape)
#        P_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_given_evid_blur
#        P_x_given_evid_full = np.zeros((n_variables, 2))
#        P_x_given_evid_full[non_evid_var,:] = P_x_given_evid
        
        
        
        #print (P_xy_given_evid_full[0])
        #print (P_x_given_evid_full)
        
        #args = (clt_R.topo_order, clt_R.parents, cpt_Q, P_x_given_evid_full, P_xy_given_evid_full, evid_list, non_evid_var)
    else:
        
#        P_xy_given_evid_full = np.zeros((n_variables, n_variables, 2,2))
#        #print (P_xy_evid_full[non_evid_var[:,None],non_evid_var].shape)
#        P_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_given_evid
#        P_x_given_evid_full = np.zeros((n_variables, 2))
#        P_x_given_evid_full[non_evid_var,:] = P_x_given_evid
               
        
        #args = (clt_R.topo_order, clt_R.parents, cpt_Q, marginal_P, pair_marginal_P)
        
        P_xy_given_evid_full = P_xy_given_evid
        P_x_given_evid_full = P_x_given_evid
    
    
    
    
    args = (clt_R, cpt_Q, P_x_given_evid_full, P_xy_given_evid_full, evid_list, non_evid_var, evid_flag)
    
    # set the bound for all variables
    bnd = (0.001,0.999)
    bounds = [bnd,]*(4*n_variables+1)
    
    x0 = np.zeros(4*n_variables+1)
    x0[0] = 0.5  # initial value for lamda
    x0[1:] = cpt_R.flatten()
    
    # constraint: valid prob
    normalize_cons = []
    for i in range (n_variables):
        
#        print (x0[i*4+1]+ x0[i*4+3])
#        print (x0[i*4+2]+ x0[i*4+4])
        
        normalize_cons.append({'type': 'eq',
           'fun' : lambda x: np.array([x[i*4+1] + x[i*4+3] - 1, 
                                       x[i*4+2] + x[i*4+4] - 1])})
   
    
    #print (x0)
    
    #res = minimize(objective, x0, method='SLSQP', jac=derivative, constraints=normalize_cons,  # with normalization constriant
    res = minimize(objective, x0, method='SLSQP', jac=derivative, # without normalization constraint
               options={'ftol': 1e-4, 'disp': True, 'maxiter': 1000},
               bounds=bounds, args = args)
    clt_R.cond_cpt = res.x[1:].reshape(n_variables,2,2)
    clt_R.log_cond_cpt = np.log(clt_R.cond_cpt)
    
    #print (res.x[1:])
#    print ('P:')
#    print (clt_P.cond_cpt.flatten())
#    print ('Q:')
#    print(cpt_Q.flatten())
#    print ('R:')    
#    print (clt_R.cond_cpt.flatten())

#    bnd = (0,1)
#    bounds = [bnd,]*2
#    x0 = np.array([0.5, 0])
#    res = minimize(objective, x0, method='SLSQP', jac=derivative,
#                   options={'ftol': 1e-6, 'disp': True, 'maxiter': 1000},
#              bounds=bounds)
    
    
    
    print ('------Cross Entropy-------')
    
    if P_type == 'mt':
#        P_xy_given_evid_full = np.zeros((n_variables, n_variables, 2,2))
#        P_xy_given_evid_full[non_evid_var[:,None],non_evid_var] = P_xy_given_evid
#        P_x_given_evid_full = np.zeros((n_variables, 2))
#        P_x_given_evid_full[non_evid_var,:] = P_x_given_evid
#        
#        P_Q = cross_entropy_evid_parm(clt_Q, P_x_given_evid_full, P_xy_given_evid_full, evid_list, non_evid_var, evid_flag)
#        P_R = cross_entropy_evid_parm(clt_R, P_x_given_evid_full, P_xy_given_evid_full, evid_list, non_evid_var, evid_flag)
        
        samples = util_opt.sample_from_mt_evid_posterior(model_P, n_samples, evids, non_evid_var)
        #print (clt_Q.getWeights(samples))
        #print (clt_R.getWeights(samples))
        
        P_Q = compute_cross_entropy_mt_sampling_evid(clt_Q, samples, evid_list)
        P_R = compute_cross_entropy_mt_sampling_evid(clt_R, samples, evid_list)
    else: # P is tree
        P_P = cross_entropy_evid(clt_P, clt_P, evid_list, non_evid_var)
        P_Q = cross_entropy_evid(clt_P, clt_Q, evid_list, non_evid_var)
        P_R = cross_entropy_evid(clt_P, clt_R, evid_list, non_evid_var)
        print ('P||P:', P_P)
    print ('P||Q:', P_Q)
    print ('P||R:', P_R)
    
    #return P_Q, P_R
    
    output_rec = np.array([P_Q, P_R])
    #print (output_rec.shape)
    output_file = '../output_results/'+data_name+'/clt_e_'+str(e_percent) +'_'+str(perturb_rate)
    with open(output_file, 'a') as f_handle:
        #print ("hahah")
        np.savetxt(f_handle, output_rec.reshape(1,2), fmt='%f', delimiter=',')
#    print ('P||P:', util_opt.compute_KL(pair_marginal_P, marginal_P, clt_P))
#    print ('P||Q:', util_opt.compute_KL(pair_marginal_P, marginal_P, clt_Q))
#    print ('P||R:', util_opt.compute_KL(pair_marginal_P, marginal_P, clt_R))
    
#    clt_S = copy.deepcopy(clt_Q)
#    if blur_flag ==True:
#        '''apply noise to P'''
#        clt_S.cond_cpt = update_S_use_P(pair_marginal_P_blur,marginal_P_blur, clt_S)
#    else:
#        clt_S.cond_cpt = update_S_use_P(pair_marginal_P,marginal_P, clt_S)
#    print ('P||S:', util_opt.compute_KL(pair_marginal_P, marginal_P, clt_S))


if __name__=="__main__":

#    start = time.time()
#    main_opt_clt()
#    print ('Total running time: ', time.time() - start)

#    start = time.time()
#    n_times = 5
#    Q_arr =  np.zeros(n_times)
#    R_arr =  np.zeros(n_times)
#    for i in range (n_times):
#        Q_arr[i], R_arr[i] = main_opt_clt('nltcs', 0.0)
#    
#    print ('avg P||Q:', np.sum(Q_arr)/n_times)
#    print ('avg P||R:', np.sum(R_arr)/n_times)
#    print ('Total running time: ', time.time() - start)
    start = time.time()
    main_opt_clt()
    print ('Total running time: ', time.time() - start)