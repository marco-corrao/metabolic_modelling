# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from equilibrator_api import ComponentContribution, Q_
from equilibrator_pathway import ThermodynamicModel
import numpy as np
from tqdm import tqdm
import pandas as pd


# %%
cc=ComponentContribution()


# %%
#Compute Standard Free energies
pp =ThermodynamicModel.from_sbtab("MDF_model_noFA_noEX.tsv", comp_contrib=cc)
pp.dg_sigma = pp.dg_sigma.real
pp.update_standard_dgs()


# %%
#Run MDF and store results
sol=pp.mdf_analysis()
a=sol.reaction_df
a.to_csv('out1.csv')


