# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:18:16 2021

@author: marco
"""

#%%
import pandas as pd
import numpy as np
import os
import cobra
import copy
import math
from sbtab import SBtab
import copy
#%Miscellaneous function
def reaction_list(model):
    #obtain a list of reaction IDs from the COBRA model
    reaction_list=[]
    for r in model.reactions:
        reaction_list.append(r.id)
    return reaction_list    

def filter_km_entries(df,column,filters_to_keep,filters_to_exclude):
    #Filter out entries depending on whether certain keywords are present or not 
    #in a specific column of the dataframe.
    #filters_to_keep: retain the entry ONLY IF all of these keywords are present
    #filters_to_exclude: exclude entry IF any of these keywords are present
    df_filtered=copy.deepcopy(df)
    df_filtered=df_filtered.reset_index()
    n_rows=len(df)
    index_to_remove=[]
    for i in range(n_rows):
        cur_row=df.iloc[i]
        cur_filter_column=cur_row[column]
        #Substitute nan with ''
        if cur_filter_column!=cur_filter_column: 
            cur_filter_column=''
            
        #Loop through keywords to keep
        for keyword in filters_to_keep:
            if cur_filter_column.find(keyword)==-1:
                #The keyword was not found: remove entry
                index_to_remove.append(i)
        #Loop for keywords to exclude
        for not_keyword in filters_to_exclude:
            if cur_filter_column.find(not_keyword)!=-1:
                #The filter word was found. Remove entry
                index_to_remove.append(i)
    #Remove duplicate indices from the list
    index_to_remove = list(dict.fromkeys(index_to_remove))        
    #Remove entries from the dataframe and retuen
    df_filtered=df_filtered.drop(index_to_remove)
    return df_filtered

def kegg2bigg(kegg_id,id_type,model):
    #Converts Kegg IDs to Bigg IDs based on available annotations in the model
    met_id=[]
    if id_type=='m':
        for met in model.metabolites:
            if ('kegg_compound' in met.annotation):
                for kegg_annotation in met.annotation['kegg_compound']:
                    if kegg_annotation==kegg_id:
                        met_id.append(met.id)
        return met_id
    #=======================================
    if id_type=='r':
        for r in model.reactions:
            if ('kegg_reaction' in r.annotation):
                if r.annotation['kegg_reaction'][0]==kegg_id:
                    return r.id
        return ''
    
def bigg2kegg(bigg_id,id_type,model):
    if id_type=='m':
        met=model.metabolites.get_by_id(bigg_id)
        if 'kegg_compound' in met.annotation:
            return met.annotation['kegg_compound'][0]
        else:
            return '' 
    if id_type=='r':
        r=model.reactions.get_by_id(bigg_id)
        if 'kegg_reaction' in r.annotation:
            return r.annotation['kegg_reaction'][0]
        else:
            return ''         
        
    
def find_km_metabolite(entry,r_id,model):
    #Tries Find the bigg ID of a reaction-metabolite pair from the Brenda Km entry. 
    #Returns '' if the metabolite could not be found
    entry_bigg=str(entry['bigg.metabolite']) 
    entry_kegg= str(entry['KEGG_ID'])
    entry_name=str(entry['LigandName'])    
    for m in model.reactions.get_by_id(r_id).metabolites:
        cur=m.id
        if entry_bigg!='nan'and cur.find(entry_bigg)>=0:
            return cur
        elif entry_kegg!='nan':
            bigg_matches=kegg2bigg(entry_kegg,'m',model)
            for b in bigg_matches:
             if(cur==b):
                 return cur
    return ''
         
#%#Get data directories
path = os.getcwd() #current path
parent = os.path.dirname(path) #parent directory
parent_2=os.path.dirname(parent)
parent_3=os.path.dirname(parent_2)
#----------------
data_path=parent+ '\data_sources'
export_path=parent+'\out'
model_path=parent_3+'\Model'
#%Load model and data sources in pandas
model=cobra.io.load_json_model(model_path+'\Escherichia_coli_iCH360.json')
km_data_full=pd.read_csv(data_path+'\km.csv',encoding='ISO-8859-1',index_col='EC_number')
km_data_ecoli=pd.read_csv(data_path+'\km_ecoli.csv',encoding='ISO-8859-1',index_col='EC_number')
#km_data_ecoli.rename(columns={"ï»¿entry_ID": "entry_ID"})
kcat_data=pd.read_excel(data_path+'\Data from Heckman et al.xls', sheet_name='Dataset_S1C_turnover_n',index_col='react_id')
keq_data=pd.read_csv(data_path+'\Equilibrator_data.tsv',index_col='reaction_id',sep='\t')
#%% 
#Create kcat data frame for parameter balancing
template={'!QuantityType':[],
          '!Compound':[]	,
          '!Reaction':[]	,
          '!Unit':[]	,
          '!Mean':[]	,
          '!Std':[],
          '!Compound:Identifiers:kegg.compound':[],
          '!Reaction:Identifiers:kegg.reaction':[]
    }
kcat_df = pd.DataFrame(data=template)
#%% 1. PARSE Kcat DATA
#Generate reaction list from model and remove BIOMASS
r_list=reaction_list(model)
r_list.remove('RBIO')
#Loop through reactions in the model. Keep track of forward and backward kcat that were not found
forward_not_found =[]
backward_not_found=[]
#Data from Heckman et al uses two extrapolation methods (median and ensemble model). Not 100% sure
#what's the difference so allowing the option to select. 
#Currently not including SD
kcat_type='kappmax_KO_ALE_per_pp_per_s_ensemble_model'
for r_id in r_list:
    #First, check if the kcat for the forward reaction exist--------------
    if r_id in kcat_data.index:
        #kappmax value for the forward reaction
        mean=kcat_data.loc[r_id][kcat_type]
        #get reaction kegg id, if available
        r_kegg_id=bigg2kegg(r_id, 'r', model)
        #create new row for the dataframe
        new_row={'!QuantityType':'product catalytic rate constant',
          '!Compound':['']	,
          '!Reaction':['R_'+r_id	],
          '!Unit':['1/s'	],
          '!Mean':[str(mean)],
          '!Std':['NaN'],
          '!Compound:Identifiers:kegg.compound':[''],
          '!Reaction:Identifiers:kegg.reaction':[r_kegg_id]
          }
        #add the row
        kcat_df=kcat_df.append(pd.DataFrame(data=new_row),ignore_index='True')
    else:
        forward_not_found.append(r_id)
    #------------------------------------------------------------
    #Then, check the backward reaction-----------------------
    back_r_id=r_id+'_b'
    if back_r_id in kcat_data.index:
        #kappmax value for the forward reaction
        mean=kcat_data.loc[back_r_id][kcat_type]
        #get reaction kegg id, if available
        r_kegg_id=''
        if 'kegg_reaction' in model.reactions.get_by_id(r_id).annotation:
            r_kegg_id=model.reactions.get_by_id(r_id).annotation['kegg_reaction'][0]
        #create new row for the dataframe
        new_row={'!QuantityType':'substrate catalytic rate constant',
          '!Compound':['']	,
          '!Reaction':['R_'+r_id	],
          '!Unit':['1/s'	],
          '!Mean':[str(mean)],
          '!Std':['NaN'],
          '!Compound:Identifiers:kegg.compound':[''],
          '!Reaction:Identifiers:kegg.reaction':[r_kegg_id]
          }
        #add the row
        kcat_df=kcat_df.append(pd.DataFrame(data=new_row),ignore_index='True')
    else:
        backward_not_found.append(r_id)    
#Create table
kcat_sbtable=SBtab.SBtabTable.from_data_frame(kcat_df,table_id='ParameterKcat',table_type='Quantity')  
kcat_sbtable.write(export_path+'\PB_kcat.tsv')
#=========================================================================
#%% 
#Create Km data frame for parameter balancing
template={'!QuantityType':[],
          '!Compound':[]	,
          '!Reaction':[]	,
          '!Unit':[]	,
          '!Mean':[]	,
          '!Std':[],
          '!Compound:Identifiers:kegg.compound':[],
          '!Reaction:Identifiers:kegg.reaction':[],
          '!Km_entry_key':[],
          '!Km_commentary':[],
          '!Provenance:BrendaReferenceID':[],  
    }
km_df = pd.DataFrame(data=template)

#%% 2. Km values from e.coli
#Generate reaction list from model and remove BIOMASS
r_list=reaction_list(model)
r_list.remove('RBIO')
#----------------------------------------------------------------------------
#1. Map Reaction Ids to E.C numbers, based on available annotation in the model
EC_not_found=[]
EC_map=dict()
#Here one can add some filters to look for in the commentary on the entry. For instance, one may wish to
#exclude entries with the word 'mutant' in the commentary
commentary_to_keep=[]
commentary_to_exclude=[]
for r_id in r_list:
    if 'ec_code' in model.reactions.get_by_id(r_id).annotation:
        EC_map[r_id]=model.reactions.get_by_id(r_id).annotation['ec_code'][0]
    else:
        EC_not_found.append(r_id)
print('EC number was not found in model annotation for '+ str(len(EC_not_found))+' reactions')
#Most of these are exchanges or transport. I will deal manually with the actual reactions for which EC annotation is not available
#------------------------------------------------------------------------------------
#2 Based on the generated Reaction_ID->EC_number map, parse data from the E.coli Km database
for r_id in r_list:
    if r_id in EC_map:
        cur_ec=EC_map[r_id]
        if cur_ec in km_data_ecoli.index:
            #Collect all available entries for that EC number
            available_entries=km_data_ecoli.loc[[cur_ec]]
            #Filter results based on specific commentary keywords
            available_entries_filtered=filter_km_entries(available_entries,'Commentary',commentary_to_keep,commentary_to_exclude)
            #Loop through available entries
            for k in range(len( available_entries_filtered)):
                cur_entry=available_entries_filtered.iloc[k]
            #avoiding the -999 values
                if float(cur_entry['KM_Value'])>0:
                    #Create new entry for the Parameter Balancing table
                    bigg_metabolite=find_km_metabolite(cur_entry,r_id,model)

                    
                    if bigg_metabolite!='':
                        r_kegg_id=bigg2kegg(r_id, 'r', model)
                        m_kegg_id=bigg2kegg(bigg_metabolite,'m',model)                        
                        new_row={'!QuantityType':'Michaelis constant',
                        '!Compound':['M_'+bigg_metabolite],
                        '!Reaction':['R_'+r_id],
                        '!Unit':['mM'],
                        '!Mean':[str(cur_entry['KM_Value'])],
                        '!Std':['NaN'],
                        '!Compound:Identifiers:kegg.compound':[m_kegg_id],
                        '!Reaction:Identifiers:kegg.reaction':[r_kegg_id],
                        '!Km_entry_key':[str(cur_entry['ï»¿entry_ID'])],
                        '!Km_commentary':[str(cur_entry['Commentary'])],
                        '!Provenance:BrendaReferenceID':[str(cur_entry['Literature'])],  
          }
                        km_df=km_df.append(pd.DataFrame(data=new_row),ignore_index='True')
#Create table
km_sbtable=SBtab.SBtabTable.from_data_frame(km_df,table_id='ParameterKM',table_type='Quantity')  
km_sbtable.write(export_path+'\PB_km.tsv')
#%%
#Create keq data frame for parameter balancing
template={'!QuantityType':[],
          '!Compound':[]	,
          '!Reaction':[]	,
          '!Unit':[]	,
          '!Mean':[]	,
          '!Std':[],
          '!Compound:Identifiers:kegg.compound':[],
          '!Reaction:Identifiers:kegg.reaction':[]
    }
keq_df = pd.DataFrame(data=template)
#%%
RT=2.479 #kj/mol
for i in range(len(keq_data)):
    cur_entry=keq_data.iloc[[i]]
    dg=float(cur_entry['standard_dg_prime_in_kJ_per_mol'])
    sigma_dg=float(cur_entry['dg_sigma_in_kJ_per_mol'])
    keq=np.exp(-dg/RT)
    sigma_keq=(1/RT)*sigma_dg*keq #error propagation
    
    r_id=cur_entry.index[0][2:]
    r_kegg_id=bigg2kegg(r_id, 'r', model)
    new_row={'!QuantityType':'equilibrium constant',
              '!Compound':[''],
              '!Reaction':[cur_entry.index[0]],
              '!Unit':['dimensionless'],
              '!Mean':[str(keq)],
              '!Std':[str(sigma_keq)],
              '!Compound:Identifiers:kegg.compound':[''],
              '!Reaction:Identifiers:kegg.reaction':[r_kegg_id],
}
    keq_df=keq_df.append(pd.DataFrame(data=new_row),ignore_index='True')
keq_sbtable=SBtab.SBtabTable.from_data_frame(keq_df,table_id='ParameterEq',table_type='Quantity')  
keq_sbtable.header_row+=" StandardConcentration='M'"
keq_sbtable.write(export_path+'\PB_Keq.tsv')    

#%%
#Create DGr^o data frame for parameter balancing
template={'!QuantityType':[],
          '!Compound':[]	,
          '!Reaction':[]	,
          '!Unit':[]	,
          '!Mean':[]	,
          '!Std':[],
          '!Compound:Identifiers:kegg.compound':[],
          '!Reaction:Identifiers:kegg.reaction':[]
    }
dg_df = pd.DataFrame(data=template)
#%% FREE ENERGIES OF REACTIONS

for i in range(len(keq_data)):
    cur_entry=keq_data.iloc[[i]]
    dg=float(cur_entry['standard_dg_prime_in_kJ_per_mol'])
    sigma_dg=float(cur_entry['dg_sigma_in_kJ_per_mol'])
    
    r_id=cur_entry.index[0][2:]
    r_kegg_id=bigg2kegg(r_id, 'r', model)
    new_row={'!QuantityType':'standard Gibbs free energy of reaction',
              '!Compound':[''],
              '!Reaction':[cur_entry.index[0]],
              '!Unit':['kJ/mol'],
              '!Mean':[str(dg)],
              '!Std':[str(sigma_dg)],
              '!Compound:Identifiers:kegg.compound':[''],
              '!Reaction:Identifiers:kegg.reaction':[r_kegg_id],
}
    dg_df=dg_df.append(pd.DataFrame(data=new_row),ignore_index='True')
dg_sbtable=SBtab.SBtabTable.from_data_frame(dg_df,table_id='ParameterEq',table_type='Quantity')  
dg_sbtable.header_row+=" StandardConcentration='M'"
dg_sbtable.write(export_path+'\PB_dg.tsv')    

#%% BOUNDS ON FREE ENERGIES

#Create DGr_bounds data frame for parameter balancing
template={'!QuantityType':[],
          '!Reaction':[]	,
          '!Unit':[]	,
          '!Min':[]	,
          '!Max':[],
          '!Reaction:Identifiers:kegg.reaction':[]
    }
dg_bounds_df = pd.DataFrame(data=template)
#=====================================================
for r in model.reactions:
    r_lb=r.bounds[0]
    r_ub=r.bounds[1]
    #--------
    new_row=copy.deepcopy(template)
    new_row['!QuantityType']= ['Gibbs free energy of reaction']
    new_row['!Unit']=['kJ/mol']
    new_row['!Reaction']=['R_'+ r.id]
    new_row['!Reaction:Identifiers:kegg.reaction']=[bigg2kegg(r.id, 'r', model)]
    if r_lb>=0: #If the reaction is irreversible in the forward direction       
        new_row['!Max']=[str(0)]
        new_row['!Min']=['']
        dg_bounds_df=dg_bounds_df.append(pd.DataFrame(data=new_row),ignore_index='True')
    elif r_ub<=0: #reaction is irreversible in the backward direction
        new_row['!Min']=[str(0)]
        new_row['!Max']=['']
        dg_bounds_df=dg_bounds_df.append(pd.DataFrame(data=new_row),ignore_index='True')
dg_bounds_sbtable=SBtab.SBtabTable.from_data_frame(dg_bounds_df,table_id='Gibbs_free_energy_constraints',table_type='Quantity')  
dg_bounds_sbtable.header_row+=" StandardConcentration='M'"
dg_bounds_sbtable.write(export_path+'\dg_bounds.tsv')    

#%% BOUNDS ON  METABOLITE CONCENTRATIONS
#Create a dictionare with the fixed conentrations of specific metabolites
fixed_conc={
 'nadph_c': 0.11389157,
 'nadh': 0.02741513,
 'nadp_c':0.1,
 'nad_c': 1,
 'atp_c': 3.4491395,
 'adp_c': 0.60461237,
 'co2_c': 0.01,
 'glc__D_c':12,
 'q8h2_c': 1,
 'q8_c':0.1,
 'h2o_c':1,
 'pi_c':10,
 'coa_c':1,
 }
 
template={'!QuantityType':[],
          '!Compound':[]	,
          '!Unit':[]	,
          '!Min':[]	,
          '!Max':[],
          '!Compound:Identifiers:kegg.compound':[]
    }
conc_bounds_df = pd.DataFrame(data=template)

for m in fixed_conc.keys():
    new_row=copy.deepcopy(template)
    for met in model.metabolites:
        if met.id==m:
            new_row['!QuantityType']=['concentration']
            new_row['!Unit']=['mM']
            new_row['!Compound']=['M_'+m]
            new_row['!Min']=[str(fixed_conc[m])]
            new_row['!Max']=[str(fixed_conc[m])]
            new_row['!Compound:Identifiers:kegg.compound']=[bigg2kegg(met.id,'m',model)]
            conc_bounds_df=conc_bounds_df.append(pd.DataFrame(data=new_row),ignore_index='True')
conc_bounds_sbtable=SBtab.SBtabTable.from_data_frame(conc_bounds_df,table_id='ConcentrationConstrains',table_type='Quantity')  
conc_bounds_sbtable.write(export_path+'\conc_bounds.tsv')    
#Save The tables
