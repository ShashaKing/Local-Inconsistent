#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:53:24 2020
With evidence.
Cnode update is not included here

"""

from __future__ import print_function
import numpy as np
import sys
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

from cnet_extend import CNET_ARR  # The array version of cutset network
from cnet_extend import save_cnet
#import control_study

from scipy.optimize import minimize
from opt_clt_ss_evid import objective, derivative




def pertub_model(model, model_type='cnet', percent=0.1):
    
    
    if model_type=='cnet':
        
        updated_cpt_list = []
        
        for j in range (len(model.path)):
      
        
            topo_order = model.leaf_info_list[j][2]
            #parents = model.leaf_info_list[j][1]
            updated_cpt = np.copy(model.leaf_cpt_list[j])
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
            
            updated_cpt_list.append(updated_cpt)
            
        return updated_cpt_list



def compute_cross_entropy_mt_sampling_evid(Q, samples, evid_list):
    
    Q.compute_path_value()
    evids = np.array(evid_list)
    n_variables = samples.shape[1]
    
    if len(evid_list) ==0:
        evid_var =  np.array([])
        evid_flag = np.full(n_variables,-1) #-1 means non evidence
    else:
        evid_var =  evids[:,0]
        
        evid_flag = np.full(n_variables,-1) #-1 means non evidence
        #evid_arr = np.asarray(evid_list)
        evid_flag[evids[:,0]] = evids[:,1]   
    
    '''find the valid path that meet the evidence'''
    valid_path =[]
    
    #print ('evid flag:', evid_flag)
    
    if len(evid_list) == 0:
        valid_path =list(np.arange(len(Q.path)))
    else:
           
        for j in range (len(Q.path)):
            
            '''first check if path violate the set of evidence'''
            
            violate_flag = False
            path = Q.path[j]
            #print ('j:', j)
            #print ('path:', path)
            
            for var_sign in path:
                var = int(var_sign[:-1])
                sign = var_sign[-1] 
                #ind = np.where(evid_var==var)[0]
                #if ind.shape[0] == 0
                if evid_flag[var] == -1: # not evid
                    continue
                if evid_flag[var]==0 and sign == '+':
                    violate_flag = True
                    break
                if evid_flag[var]==1 and sign == '-':
                    violate_flag = True
                    break
            
            if violate_flag == True:
                continue
            else:
                valid_path.append(j)
    log_path_val = np.log(Q.path_value)
    #print ('valid_path:', valid_path)
    
    '''compute P(X,e)'''
    probs = np.zeros(samples.shape[0])
    P_e = 0
    for j in (valid_path):
        path = Q.path[j]
        inds = np.arange(samples.shape[0])
        
        evid_list =[] 
        evid_sub_tree = []
        for var_sign in path:
            var = int(var_sign[:-1])
            sign = var_sign[-1]
            
            if sign == '-': 
                sub_inds = np.where(samples[:,var]==0)[0]
                evid_list.append([var,0])
            
            if sign == '+': 
                sub_inds = np.where(samples[:,var]==1)[0]
                evid_list.append([var,1])
            
            inds = np.intersect1d(inds, sub_inds)
        
        evid_var_in_path = np.asarray(evid_list)[:,0]        
        evid_var_in_leaf = np.setdiff1d(evid_var, evid_var_in_path)
        
        ids = np.delete(np.arange(n_variables), evid_var_in_path)
        
        #print ('inds:', inds)
        #print ('ids:', ids)
        sub_samples = samples[inds,:][:, ids]
        #print (sub_samples.shape)
        
        sub_tree = CLT()
        sub_tree.topo_order =Q.leaf_info_list[j][2]
        sub_tree.parents = Q.leaf_info_list[j][1]
        sub_tree.cond_cpt = Q.leaf_cpt_list[j]
        sub_tree.log_cond_cpt = np.log(Q.leaf_cpt_list[j])
        
        #print (sub_tree.topo_order)
        #print (sub_tree.parents)
        #print (sub_tree.cond_cpt)
        

        
        probs[inds] = sub_tree.getWeights(sub_samples)+log_path_val[j]
        
        
        for ev in evid_var_in_leaf:
            ev_var = np.where(ids == ev)[0][0]
            evid_sub_tree.append([ev_var, evid_flag[ev]])
        
        #print ('bbb:', evid_var_in_leaf)
        sub_cond_cpt_evid = sub_tree.instantiation(evid_sub_tree)
        evid_prob = utilM.ve_tree_bin(sub_tree.topo_order, sub_tree.parents, sub_cond_cpt_evid) 
        P_e += evid_prob * Q.path_value[j]
        
        
        
    #print (P_e)
    #print (probs)
    
    #print (np.sum(probs)/samples.shape[0])
    #sss

    approx_cross_entropy = np.sum(probs)/samples.shape[0] - np.log(P_e)
    return approx_cross_entropy 

    

def convert_cnet_to_arr(cnet):
    main_dict = {}
    main_dict['depth'] = cnet.depth
    main_dict['n_variables'] =cnet.nvariables
    main_dict['structure'] = {}
    
    # save the cnet to the structure that can be stored later
    save_cnet(main_dict['structure'], cnet.tree, np.arange(cnet.nvariables))
    
    #print main_dict['structure']['x']
    
    cnet_a = CNET_ARR(main_dict['n_variables'], main_dict['depth'])
    cnet_a.convert_to_arr_ccnet(main_dict['structure'])
    #print (cnet_a.cnode_info)
    
    cnet_a.path = cnet_a.print_all_paths_to_leaf()
    #print (cnet_a.path)
    return cnet_a





    



def check(node):
    '''find the leaf node''' 
    # internal nodes
    if isinstance(node,list):
        #print ('*** in internal nodes ***')
        id,x,p0,p1,node0,node1=node
        print ('id, x: ', id, x)
        
        check(node0)
        check(node1)
        
    else:
        print ('parents: ', node.parents)
        

def check_arr(cnet_Q_arr):
    print ('path:', cnet_Q_arr.path)
    for i in range (len(cnet_Q_arr.path)):
        print ('sub tree: ', cnet_Q_arr.leaf_info_list[i][1])
    

'''version without evidence'''
def objective_cnode(x, P_cnode, Q_cnode):
    lamda = x[0]
    theta = x[1:]
    
    first_part = lamda * np.sum(P_cnode*np.log(theta))
    second_part = (1-lamda) * np.sum(Q_cnode*np.log(theta))
    
    #print ('obj:',first_part+second_part)
    return -(first_part+second_part)


'''version without evidence'''
def derivative_cnode(x, P_cnode, Q_cnode):
    lamda = x[0]
    theta = x[1:]
    
    n_cnodes = theta.shape[0]/2
    der_lam = np.sum(P_cnode*np.log(theta)) - np.sum(Q_cnode*np.log(theta))
    
    der_theta = np.zeros_like(theta)
    
    #der_theta = (lamda*(P_cnode)- (1-lamda)*Q_cnode)/theta
    der_theta[:n_cnodes] =  (lamda*(P_cnode[:n_cnodes])- (1-lamda)*Q_cnode[:n_cnodes])/theta[:n_cnodes]
    der_theta[n_cnodes:] = der_theta[:n_cnodes]*(-1)
    
    
    #print (der_theta)

    der = np.zeros_like(x)
    der[0] = der_lam
    der[1:] = der_theta
    
    return der *(-1.0)
    
## the objective function
#def objective(x, topo_order, parents, cpt_Q, marginal_P, pair_marginal_P):
#    #print (marginal_P)
#    #print (pair_marginal_P)
#
#    lamda = x[0]
#    theta = x[1:].reshape(marginal_P.shape[0],2,2)
#           
#    # get marginals:
#    #marginal_R =  (topo_order, parents, theta)
#
#    # first part:
#    cross_PlogR =  util_opt.compute_cross_entropy_parm(pair_marginal_P,marginal_P, parents, topo_order, theta)
#    first_part = lamda*cross_PlogR
#    
#    #print ('first part: ', first_part)
#    
#    # second part:
#    sec_part = (1.0-lamda)*(np.sum(cpt_Q *np.log(theta)))
#    
#    # maximize is the negation of minimize
#    #print ('obj value: ', first_part+sec_part)
#    return -(first_part+sec_part)
#    
#    
#    
## the derivative function
#def derivative(x, topo_order, parents, cpt_Q, marginal_P, pair_marginal_P):
#
#    lamda = x[0]
#    theta = x[1:].reshape(marginal_P.shape[0],2,2)
#    nvariable = topo_order.shape[0]
#    
#    '''
#    marginal_R = get_single_var_marginals(topo_order, parents, theta)
#
#    # derivative of lamda
#    der_lam = np.sum(marginal_P*np.log(marginal_R)） - np.sum(cpt_Q *np.log(theta))
#    
#    # derivativ of thetas
#    der_theta = np.zeros_like(theta)
#    
#
#    jt.clique_potential = np.copy(theta)
#    jt.clique_potential[0,0,1] = jt.clique_potential[0,1,0] = 0
#    # add 1 varialbe in JT
#    jt_var = copy.deepcopy(jt)
#        
#    for var in range(nvariable):
#
#        new_potential = jt_var.add_query_var(var)
# 
#        jt_var.propagation(new_potential)
#               
#        # normalize
#        norm_const=np.einsum('ijkl->i',new_potential)
#        new_potential /= norm_const[:,np.newaxis,np.newaxis,np.newaxis]
#
#        der_theta[:,:,:] += (marginal_P[var,0]/marginal_R[var,0])*(new_potential[:,:,:,0]/theta[:,:,:]) + \
#                (marginal_P[var,1]/marginal_R[var,1])*(new_potential[:,:,:,1]/theta[:,:,:])               
#            
#    '''
#    
#    # derivative of lambda
#    cross_PlogR =  util_opt.compute_cross_entropy_parm(pair_marginal_P,marginal_P, parents, topo_order, theta)
#    der_lam = cross_PlogR - np.sum(cpt_Q *np.log(theta))
#    
#    # derivativ of thetas
#    der_theta = np.zeros_like(theta)
#        
#    '''marginals of (x,u) where (x,u) is one edge in R, ordered in topo_order of R'''
#    edge_marginal_P = np.zeros_like(cpt_Q)
#    for i in range (nvariable-1):
#        cld = topo_order[i+1]
#        pa = parents[cld]
#        edge_marginal_P[i+1] = pair_marginal_P[cld, pa]
#    
#    root = topo_order[0]
#    edge_marginal_P[0,0,:] = marginal_P[root,0]
#    edge_marginal_P[0,1,:] = marginal_P[root,1]
#    #print (edge_marginal_P)
#        
#    der_theta[:,:,:] = lamda*edge_marginal_P/theta+(1.0-lamda)*(cpt_Q[:,:,:]/theta[:,:,:])
#    
#    #print (der_theta)
#
#    '''Apply theta_{\bar{b}|a} = 1-theta_{b|a}'''
#    # root: special case
#    der_theta[0,0,0] -= der_theta[0,1,1]
#    der_theta[0,1,1] = -der_theta[0,0,0]
#    der_theta[0,0,1] = der_theta[0,0,0]    
#    der_theta[0,1,0] = der_theta[0,1,1]
#
#    der_theta[1:,0,:] -= der_theta[1:,1,:]
#    der_theta[1:,1,:] = -der_theta[1:,0,:]
#    
#    #print ('---')
#    #print (der_theta)
#
#        
#
#    der = np.zeros_like(x)
#    der[0] = der_lam
#    der[1:] = der_theta.flatten() 
#    
#    return der *(-1.0)




def main_opt_cnet():

    dataset_dir = sys.argv[2]
    data_name = sys.argv[4]
    #n_components_P = int(sys.argv[6])
    mt_dir = sys.argv[6]
    depth = int(sys.argv[8])
    perturb_rate = float(sys.argv[10])
    e_percent = float(sys.argv[12])
   
    tum_module = data_name
    noise_mu = 0
    noise_std = 0.1
    noise_percent = 1
    #noise_parm = np.array([noise_mu, noise_std, noise_percent])
    noise_flag = True # Assume get noise distribtuion from P
    n_samples = 100000
    
    #print ('noise:', noise_parm)
    
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
    
#    evids = np.zeros((2,2), dtype = int)
#    evids[:,0] = np.array([1,2])
#    evids[:,1] = np.array([0,1])
#    evid_var =  evids[:,0]
#    non_evid_var = np.setdiff1d(np.arange(n_variables), evid_var)
#    evid_list = list(evids)
#    
#    evid_flag = np.full(n_variables,-1) #-1 means non evidence
#    #evid_arr = np.asarray(evid_list)
#    evid_flag[evids[:,0]] = evids[:,1]    
#    
#    print ('evid:')
#    print (evids)
#    print ('non evids:')
#    print (non_evid_var)
    
    evids = util_opt.read_evidence_file('../evidence/', e_percent, data_name)
    
    evid_var =  evids[:,0]
    non_evid_var = np.setdiff1d(np.arange(n_variables), evid_var)
    evid_list = list(evids)
    
    evid_flag = np.full(n_variables,-1) #-1 means non evidence
    #evid_arr = np.asarray(evid_list)
    evid_flag[evids[:,0]] = evids[:,1]   
    
    
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
    
    
    model_P = reload_mix_clt
    
    
#    p_xy_all = np.zeros((non_evid_size, non_evid_size, 2, 2))
#    p_x_all = np.zeros((non_evid_size, 2))
#    for i, jt in enumerate(model_P.jt_list):
#        p_xy = JT.get_marginal_JT(jt, evid_list, np.arange(non_evid_size))
#        p_xy_all += p_xy * model_P.mixture_weight[i]
#
#    #print (p_xy_all.shape)
#    #print (p_xy_all)
#
#    p_x_all[:,0] = p_xy_all[0,:,0,0] + p_xy_all[0,:,1,0]
#    p_x_all[:,1] = p_xy_all[0,:,0,1] + p_xy_all[0,:,1,1]
#    
#    p_x_all[0,0] = p_xy_all[1,0,0,0] + p_xy_all[1,0,1,0]
#    p_x_all[0,1] = p_xy_all[1,0,0,1] + p_xy_all[1,0,1,1]
#    
#
#    # Normalize        
#    P_x_given_evid = Util.normalize1d(p_x_all)
#    
#    P_xy_given_evid = Util.normalize2d(p_xy_all)
#    for i in xrange (non_evid_size):
#        P_xy_given_evid[i,i,0,0] = p_x_all[i,0] - 1e-8
#        P_xy_given_evid[i,i,1,1] = p_x_all[i,1] - 1e-8
#        P_xy_given_evid[i,i,0,1] = 1e-8
#        P_xy_given_evid[i,i,1,0] = 1e-8

    
    
    """
    construct the cnet
    """
    #cnets = []
    print("Learning Cutset Networks from partial training data.....")
    half_data = train_dataset[:int(train_dataset.shape[0]/10),:]
    
    '''
    Cutset Network Learn from dataset
    '''
    print ('-------------- Cutset Network Learn from Data: (Q) ----------')
    cnet_Q = CNET(depth=depth)
    cnet_Q.learnStructure(half_data)
    cnet_Q_arr = convert_cnet_to_arr(cnet_Q)
    #print (cnet_Q_arr.path)
    #print (cnet_Q_arr.cnode_info)
    
    perturb_leaf_cpt_list = pertub_model(cnet_Q_arr, model_type='cnet', percent=perturb_rate)
    
    for j in range (len(cnet_Q_arr.path)):
        cnet_Q_arr.leaf_cpt_list[j] = perturb_leaf_cpt_list[j]
    
    #cross_PQ = util_opt.compute_cross_entropy_cnet(reload_mix_clt, cnet_Q_arr)
    #print ('P||Q:', cross_PQ)
    #print ()
    
    #check(cnet_Q.tree)
    #check_arr(cnet_Q_arr)

    
    
    print ('-------------- Cutset Network Learn from P and Q: (R) ----------')
    cnet_R = copy.deepcopy(cnet_Q)
    cnet_R_arr = copy.deepcopy(cnet_Q_arr)

    
    '''first get the valid path, path that not violate the evidence'''
    valid_path =[]    
    for j in range (len(cnet_Q_arr.path)):
        
        '''first check if path violate the set of evidence'''
        
        violate_flag = False
        path = cnet_Q_arr.path[j]
        #print ('j:', j)
        #print ('path:', path)
        
        for var_sign in path:
            var = int(var_sign[:-1])
            sign = var_sign[-1] 
            #ind = np.where(evid_var==var)[0]
            #if ind.shape[0] == 0
            if evid_flag[var] == -1: # not evid
                continue
            if evid_flag[var]==0 and sign == '+':
                violate_flag = True
                break
            if evid_flag[var]==1 and sign == '-':
                violate_flag = True
                break
        
        if violate_flag == True:
            continue
        else:
            valid_path.append(j)
        
    #print ('valid path a: ', valid_path)

    
    '''
    Inference P to get list of marginals and pairwise marginals for
    each leaf tree in Q
    '''
    pair_marginal_P = []
    marginal_P = []
    
    #for i in range (len(cnet_Q_arr.path)):
    for i in valid_path:
        #print (i)
        path = cnet_Q_arr.path[i]
        
        #print ('path: ', path)

        evid_list =[]  # evidence list
        for var_sign in path:
            var = int(var_sign[:-1])
            sign = var_sign[-1]
            
            #print (var, sign)
            
            if sign == '-': # going to left
                
                '''add evidence to P'''
                evid_list.append([var,0])
                                
            
            elif sign == '+': # going to right
                #print (Q.cnode_info[2,var_ind])
                
                '''add evidence to P'''
                evid_list.append([var,1])
              
        evid_var_in_path = np.asarray(evid_list)[:,0]
        evid_var_in_leaf = np.setdiff1d(evid_var, evid_var_in_path)
        
        #print (evid_var_in_path)
        #print (evid_var_in_leaf)
        
        #print (np.sort(evid_arr[:,0]))
        ids = np.delete(np.arange(n_variables), evid_var_in_path)
        #print (np.concatenate((evid_var_in_path, evid_var_in_leaf), axis=None))
        #print ('ids: ', ids)
        #pxy, px = model_P.inference_jt([],ids)
        
        for ev in evid_var_in_leaf:
            evid_list.append([ev, evid_flag[ev]])
        
        #print (evid_list)
        pxy, px = model_P.inference_jt(evid_list,ids) # already nomalized
        pair_marginal_P.append(pxy)
        marginal_P.append(px)
        
        #print (px)

    
    '''
    # for weights assigned to cnode
    Let's first not update the cnode weight. And code here is for the version
    That not has evidence
    '''
    cnode_ind = np.where(cnet_R_arr.cnode_info[0] >= 0)[0]
    n_cnodes = cnode_ind.shape[0] # number of cnods
    #cnode_weigths = np.zeros(2*n_cnodes)
    marginal_P_cnodes = np.zeros((2,n_cnodes))
    
    # for each branch in cutset network
    #for i in range (len(cnet_R_arr.path)):
    for i in valid_path:
        path = cnet_R_arr.path[i]
        #print ('path:', path)
        P_temp = copy.deepcopy(model_P)
        #print (P.topo_order)
        #print (P.parents)
        
        var_ind = 0
        
        evid_list =[]  # evidence regarding to distribution P
        for var_sign in path:
            var = int(var_sign[:-1])
            sign = var_sign[-1]
            
            #print (var, sign)
            incremental_evid_list =[]  # evidence that increased in every depth
            if sign == '-': # going to left
                
                '''add evidence to P'''
                evid_list.append([var,0])
                incremental_evid_list.append([var,0])
                #P_temp.cond_cpt = P_temp.instantiation(evid_list)
                for k, t in enumerate(P_temp.clt_list):
                    t.cond_cpt = t.instantiation(incremental_evid_list)
                    
                
                                
                if marginal_P_cnodes[0, var_ind] == 0: # not calculated
                    #print ('var:', var, 'sign:', sign)
                    #print (evid_list)
                    
                    '''P_marginal should be calculated based on different distribution P'''
                    
                    P_marginal = 0
                    for k, t in enumerate(P_temp.clt_list):                    
                        P_marginal += utilM.ve_tree_bin(t.topo_order, t.parents, t.cond_cpt) * P_temp.mixture_weight[k]
                    #print ('P m: ', P_marginal)
                    marginal_P_cnodes[0, var_ind] = P_marginal
                    #print (cnode_entropy[0,var_ind])
                
                var_ind = 2*var_ind+1
                
                
            
            if sign == '+': # going to right
                #print (Q.cnode_info[2,var_ind])
                
                '''add evidence to P'''
                evid_list.append([var,1])
                incremental_evid_list.append([var,1])
                #P_temp.cond_cpt = P_temp.instantiation(evid_list)
                for k, t in enumerate(P_temp.clt_list):
                    t.cond_cpt = t.instantiation(incremental_evid_list)
                
                if marginal_P_cnodes[1, var_ind] == 0: # not calculated
                    
                    #print ('var:', var, 'sign:', sign)
                    #print (evid_list)
                    
                    '''P_marginal should be calculated based on different distribution P'''
                    P_marginal = 0
                    for k, t in enumerate(P_temp.clt_list):                    
                        P_marginal += utilM.ve_tree_bin(t.topo_order, t.parents, t.cond_cpt) * P_temp.mixture_weight[k]
                    #P_marginal = utilM.ve_tree_bin(P_temp.topo_order, P_temp.parents, P_temp.cond_cpt)
                    #print ('P m: ', P_marginal)
                    marginal_P_cnodes[1, var_ind] = P_marginal
                    #print (cnode_entropy[1,var_ind])
                
                var_ind = 2*var_ind+2
            
    
    #print (marginal_P_cnodes)
    
#    sum_val = np.sum(marginal_P_cnodes, axis =0)
#    #print (sum_val)
#    marginal_P_cnodes /= sum_val
#    #print (marginal_P_cnodes)


        
    '''
    Add noise to pairwise marginals
    '''
    pair_marginal_P_noise = []
    for i in range (len(pair_marginal_P)):
        path_ind = valid_path[i]
        pair_marginal_P_noise.append( util_opt.add_noise (pair_marginal_P[i], n_variables-len(cnet_Q_arr.path[path_ind]), noise_mu, noise_std, percent_noise=noise_percent))
        
    marginal_P_noise = marginal_P
    
    
    '''
    Update cnet R leaf parameters
    '''
    #for j in range (len(cnet_Q_arr.path)):
    
    for ind, j in enumerate(valid_path):
        
                
        '''clt_R, cpt_Q, marginal_P, pair_marginal_P, evid_list, non_evid_var, evid_flag'''
        
        #for i in range (len(cnet_Q_arr.path)):
   
        path = cnet_Q_arr.path[j]
        
        #print ('path pp: ', path)

        
        evid_sub_tree = []
        non_evid_var_sub_tree = []
        evid_flag_sub_tree =[]
        
        evid_list =[]  # evidence list
        for var_sign in path:
            var = int(var_sign[:-1])
            sign = var_sign[-1]
            
            #print (var, sign)
            
            if sign == '-': # going to left
                
                '''add evidence to P'''
                evid_list.append([var,0])
                                
            
            elif sign == '+': # going to right
                #print (Q.cnode_info[2,var_ind])
                
                '''add evidence to P'''
                evid_list.append([var,1])
              
        evid_var_in_path = np.asarray(evid_list)[:,0]
        evid_var_in_leaf = np.setdiff1d(evid_var, evid_var_in_path)
        
        #print (evid_var_in_path)
        #print (evid_var_in_leaf)
        
        #print (np.sort(evid_arr[:,0]))
        ids = np.delete(np.arange(n_variables), evid_var_in_path)
        #print (np.concatenate((evid_var_in_path, evid_var_in_leaf), axis=None))
        #print ('ids: ', ids)
        #pxy, px = model_P.inference_jt([],ids)
        
        for ev in evid_var_in_leaf:
            ev_var = np.where(ids == ev)[0][0]
            evid_sub_tree.append([ev_var, evid_flag[ev]])

        evid_sub_tree_arr = np.array(evid_sub_tree)
        
        non_evid_var_sub_tree = np.setdiff1d(np.arange(ids.shape[0]), evid_sub_tree_arr[:,0])
       
        evid_flag_sub_tree = np.full(ids.shape[0],-1) #-1 means non evidence
        #evid_arr = np.asarray(evid_list)
        evid_flag_sub_tree[evid_sub_tree_arr[:,0]] = evid_sub_tree_arr[:,1]   
        
        #print ('aaa:',evid_sub_tree)
        #print (non_evid_var_sub_tree)
        #print (evid_flag_sub_tree)

        
        if noise_flag == True:
            '''apply noise to P''' # 2 is topo order, 1 is parent
            sub_tree_R = CLT()
            sub_tree_R.topo_order =cnet_R_arr.leaf_info_list[j][2]
            sub_tree_R.parents = cnet_R_arr.leaf_info_list[j][1]
            sub_tree_R.cond_cpt = cnet_R_arr.leaf_cpt_list[j]
            
            args = (sub_tree_R, cnet_Q_arr.leaf_cpt_list[j], marginal_P_noise[ind], pair_marginal_P_noise[ind], evid_sub_tree, non_evid_var_sub_tree, evid_flag_sub_tree)
        else:
            args = (sub_tree_R, cnet_Q_arr.leaf_cpt_list[j], marginal_P[ind], pair_marginal_P[ind], evid_sub_tree, non_evid_var_sub_tree, evid_flag_sub_tree)
        
        sub_nvariables = n_variables-len(cnet_Q_arr.path[j])
        # set the bound for all variables
        bnd = (0.001,0.999)
        bounds = [bnd,]*(4*sub_nvariables+1)
        
        x0 = np.zeros(4*sub_nvariables+1)
        x0[0] = 0.5  # initial value for lamda
        x0[1:] = cnet_R_arr.leaf_cpt_list[j].flatten()
        
        # constraint: valid prob
        normalize_cons = []
        for i in range (sub_nvariables):
            
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
        #print (sub_nvariables)
        #print (res.x)
        #print (cnet_R_arr.leaf_cpt_list[j])
        cnet_R_arr.leaf_cpt_list[j] = res.x[1:].reshape(sub_nvariables,2,2)
    
    
    
    samples = util_opt.sample_from_mt_evid_posterior(model_P, n_samples, evids, non_evid_var)
    
#    print ('1:', compute_cross_entropy_mt_sampling_evid(cnet_Q_arr, samples, []))
#    print ('2:', cnet_Q.computeLL(samples)/samples.shape[0])
    
    #print ('evids:', evids)
    #print (evid_list)

    
    P_Q = compute_cross_entropy_mt_sampling_evid(cnet_Q_arr, samples, list(evids))
    #print ('P||Q:', P_Q)
    P_R = compute_cross_entropy_mt_sampling_evid(cnet_R_arr, samples, list(evids))
    
    
    
    #cross_PR = util_opt.compute_cross_entropy_cnet(reload_mix_clt, cnet_R_arr)
    print ('P||Q:', P_Q)
    print ('P||R:', P_R)
    
    '''
    Update cnet R cnode parameters
    Code here is for version without evidence
    '''
    x0= np.zeros(2*n_cnodes+1)
    x0[0] = 0.5  # initial value for lamda
    
    
    x0[1:] = cnet_R_arr.cnode_info[1:3,:n_cnodes].flatten()
    
    
    args_cnode = (marginal_P_cnodes.flatten(), cnet_Q_arr.cnode_info[1:3,:n_cnodes].flatten())
    
    bnd = (0.001,0.999)
    bounds_cnode = [bnd,]*(2*n_cnodes+1)
    
    
    res = minimize(objective_cnode, x0, method='SLSQP', jac=derivative_cnode, # without normalization constraint
                   options={'ftol': 1e-6, 'disp': True, 'maxiter': 1000},
                   bounds=bounds_cnode, args = args_cnode)
    
    updated_cnode_weights = res.x[1:].reshape(2, n_cnodes)
    
    sum_val = np.sum(updated_cnode_weights, axis =0)
    #print (sum_val)
    updated_cnode_weights /= sum_val
    cnet_R_arr.cnode_info[1,:n_cnodes] = updated_cnode_weights[0]
    cnet_R_arr.cnode_info[2,:n_cnodes] = updated_cnode_weights[1]
    
    #print ('------')
    #print ('cnet_R_arr.cnode_info')
    #print (cnet_R_arr.cnode_info)
    
    #cross_PR2 = util_opt.compute_cross_entropy_cnet(reload_mix_clt, cnet_R_arr)
    #print ('P||R:', cross_PR2)
    
    #return P_Q, P_R
    
    output_rec = np.array([P_Q, P_R])
    #print (output_rec.shape)
    output_file = '../output_results/'+data_name+'/cnet_e_'+str(e_percent) +'_'+str(perturb_rate)
    with open(output_file, 'a') as f_handle:
        #print ("hahah")
        np.savetxt(f_handle, output_rec.reshape(1,2), fmt='%f', delimiter=',')


if __name__=="__main__":
    #main_cutset()
    #main_clt()
#    start = time.time()
#    n_times = 5
#
#    Q_arr =  np.zeros(n_times)
#    R_arr =  np.zeros(n_times)
#    for i in range (n_times):
#        Q_arr[i], R_arr[i] = main_opt_cnet('nltcs', 0.0)
#        
#    print ('avg P||Q:', np.sum(Q_arr)/n_times)
#    print ('avg P||R:', np.sum(R_arr)/n_times)
#    print ('Total running time: ', time.time() - start) 
    start = time.time()
    main_opt_cnet()
    print ('Total running time: ', time.time() - start)