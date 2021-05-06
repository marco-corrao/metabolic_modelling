# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cobra
import copy

#Model where the exchange and diffusion reactions are flagged
m_flag=cobra.io.load_json_model('Escherichia_coli_iCH360.json')

#%% Flag Exchange reactions
for r in m_flag.reactions:
    if r.id[0:3]=='EX_':
        r.annotation['exchange_reaction']='true'
    else:
        r.annotation['exchange_reaction']='false'
    if r.name.find('diffusion')!=-1 and r.gene_reaction_rule.find('s0001')!=-1 :
        r.annotation['diffusion_reaction']='true'
    else:
        r.annotation['diffusion_reaction']='false'
cobra.io.save_json_model(m_flag,'Escherichia_coli_iCH360_exchange_diffusion_flagged.json')