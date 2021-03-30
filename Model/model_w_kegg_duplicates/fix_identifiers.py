# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:47:15 2021

@author: marco
"""

#%%
import cobra
#Name of the COBRA input file
input_name='Escherichia_coli_iCH360.json'

#Name of the COBRA output file
output_name='Escherichia_coli_iCH360'

correct_identifier_reaction={
    'kegg_compound':'kegg.compound',
    'kegg_reaction':'kegg.reaction',
    'bigg_reaction':'bigg.reaction',
    'metanetx_reaction':'metanetx.reaction',
    'metanetx_chemical':'metanetx.chemical',
    'seed_compound':'seed.compound',
    'seed_reaction':'seed.reaction',
    'sabiork':'sabiork.reaction',
    'reactome_reaction': 'reactome',
    'inchi_key': 'inchikey',
    'ec_code':'ec-code'
    }

correct_identifier_metabolite={
    'kegg_compound':'kegg.compound',
    'kegg_reaction':'kegg.reaction',
    'bigg_metabolite':'bigg.metabolite',
    'bigg_reaction':'bigg.reaction',
    'metanetx_reaction':'metanetx.reaction',
    'metanetx_chemical':'metanetx.chemical',
    'seed_compound':'seed.compound',
    'seed_reaction':'seed.reaction',
    'sabiork':'sabiork.compound', #<--
    'reactome_compound': 'reactome',
    'inchi_key': 'inchikey',
    'kegg_drug':'kegg.drug'
    }

model=cobra.io.load_json_model(input_name)

#1. Correct reactions
for r in model.reactions:
    for identifier in r.annotation:
        if identifier in correct_identifier_reaction:
            r.annotation[correct_identifier_reaction[identifier]]=r.annotation[identifier]
            del  r.annotation[identifier]
for m in model.metabolites:
    for identifier in m.annotation:
        if identifier in correct_identifier_metabolite:
            m.annotation[correct_identifier_metabolite[identifier]]=m.annotation[identifier]
            del  m.annotation[identifier]            

            
#%% EXPORT
cobra.io.save_json_model(model, output_name+'.json')
cobra.io.write_sbml_model(model, output_name+'.xml')