'''
A range of utilities functions 
'''
import cobra
import pandas as pd
import numpy as np
import tqdm

def fba_wmc(model, kapp_df,P_max):
    '''
    Extend a stoichiometric model with enzymatic constraints and solves the resulting problem
    model: a COBRA metabolic model
    kapp_df: a pandas dataframe with the following columns:
             reaction_id: id of reaction in model
             direction: fwd or bwd
             kapp: kapp in 1/h
             mw: molecular weight in g/mmol
    P_max: proteomic mass bound g/gDW

    '''
    with model as m: #create a copy to return
        #Iterate through rows of kapp_df and prepare the enzymatic contraint bounds
        coefficients=dict() #will contain the linear coefficieny of the proteomic constraint
        for i in range(kapp_df.shape[0]):
            #parse current row
            cur_data=kapp_df.iloc[i]
            kapp=cur_data['kapp'] #current apparent kcat
            mw=cur_data['mw']     #current molecular weight of the enzyme

            cur_rxn=m.reactions.get_by_id(cur_data['reaction_id'])
            if cur_data['dir']=='fwd':
                coefficients[cur_rxn.forward_variable] = mw/kapp
            elif cur_data['dir']=='bwd':
                coefficients[cur_rxn.reverse_variable] = mw/kapp

        #Add the linear constraint to the cobra model
        mc_constraint = m.problem.Constraint(0,lb=0,ub=P_max)
        m.add_cons_vars(mc_constraint)
        m.solver.update()
        mc_constraint.set_linear_coefficients(coefficients=coefficients)
        #Now solve
        sol=m.optimize()
    return sol

def satFBA(model,kapp_df,P_max,uptake_flux_data,N):
    m=model.copy()
    #---
    v_up_id=uptake_flux_data['id']
    v_up_mw=uptake_flux_data['mw']
    v_up_kcat=uptake_flux_data['kcat']
    v_up_dir=uptake_flux_data['dir']
    #---
    v_up_kapp_range=np.linspace(1e-1,v_up_kcat,N)
    #prepare constraints
    coefficients=dict() #will contain the linear coefficient of the proteomic constraint
    for i in range(kapp_df.shape[0]):
        #parse current row
        cur_data=kapp_df.iloc[i]

        if cur_data['reaction_id']!=v_up_id:
            kapp=cur_data['kapp'] #current apparent kcat
            mw=cur_data['mw']     #current molecular weight of the enzyme

            cur_rxn=m.reactions.get_by_id(cur_data['reaction_id'])
            if cur_data['dir']=='fwd':
                coefficients[cur_rxn.forward_variable] = mw/kapp
            elif cur_data['dir']=='bwd':
                coefficients[cur_rxn.reverse_variable] = mw/kapp

    solutions=[]

    for v_up_kapp in v_up_kapp_range:
        with m:
            v_up_rxn=m.reactions.get_by_id(v_up_id)
            if v_up_dir=='fwd':
                coefficients[v_up_rxn.forward_variable]=v_up_mw/v_up_kapp
            elif v_up_dir=='bwd':    
                coefficients[v_up_rxn.reverse_variable]=v_up_mw/v_up_kapp
            mc_constraint = m.problem.Constraint(0,lb=0,ub=P_max)
            m.add_cons_vars(mc_constraint)
            m.solver.update()
            mc_constraint.set_linear_coefficients(coefficients=coefficients)
            solutions.append(m.optimize())
        
    return v_up_kapp_range, solutions

def parametric_fba(model, rxn_id, bound_range=(-100,-0.1),bound_ar=None,bound_type='lb',N=100,kapp_df=None,P_max=None):
    if bound_ar is None:
        vp_ar=np.linspace(bound_range[0],bound_range[1],N)
    else:
        vp_ar=bound_ar
    
    sols=[]
    for b in tqdm.tqdm(vp_ar):
        with model:
            if bound_type=='lb':
                model.reactions.get_by_id(rxn_id).lower_bound=b
            elif bound_type=='ub':
                model.reactions.get_by_id(rxn_id).upper_bound=b
            else:
                raise ValueError('bound_type has to be \'lb\' or \'ub\'')
            if kapp_df is None:
                sols.append(model.optimize())
            else:
                sols.append(fba_wmc(model,kapp_df,P_max)

                )
    return vp_ar, sols
            

            