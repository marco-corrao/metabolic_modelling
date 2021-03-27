# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:18:29 2021

@author: marco
"""

import cobra
import copy

#Model where the exchange reactions are flagged
m_flag=cobra.io.load_json_model('Escherichia_coli_iCH360.json')
#Model where the exchange reactions are removed
m_remove=copy.deepcopy(m_flag)
#%% Flag Exchange reactions
for r in m_flag.reactions:
    if r.id[0:3]=='EX_':
        r.notes['exchange_reaction']='true'
    else:
        r.notes['exchange_reaction']='false'

#Save the updated model
cobra.io.save_matlab_model(m_flag, 'model_w_exchange_flagged/Escherichia_coli_iCH360_exchange_flagged.json')
cobra.io.write_sbml_model(m_flag, 'model_w_exchange_flagged/Escherichia_coli_iCH360_exchange_flagged.xml')

#%% Remove Exchange reacions
for r in m_remove.reactions:
    if r.id[0:3]=='EX_':
        m_remove.remove_reactions([r])

cobra.io.save_matlab_model(m_flag, 'model_wo_exchange/Escherichia_coli_iCH360_wo_exchange.json')
cobra.io.write_sbml_model(m_flag, 'model_wo_exchange/Escherichia_coli_iCH360_wo_exchange.xml')
