# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from equilibrator_api import ComponentContribution, Q_
from equilibrator_pathway import ThermodynamicModel
import numpy as np
from tqdm import tqdm
import pandas as pd


# %%
cc = ComponentContribution()


# %%
#Compute Standard Free energies
pp = ThermodynamicModel.from_sbtab("MDF_model_noFA_noEX.tsv", comp_contrib=cc)
pp.dg_sigma = pp.dg_sigma.real
pp.update_standard_dgs()

# %%
#Run MDF and store results
sol = pp.mdf_analysis()

reaction_df = sol.reaction_df[["reaction_id", "reaction_formula"]].copy()
reaction_df["standard_dg_prime_in_kJ_per_mol"] = pp.standard_dg_primes.m_as("kJ/mol").round(2)
reaction_df["dg_sigma_in_kJ_per_mol"] = np.sqrt(np.diag((pp.dg_sigma @ pp.dg_sigma.T).m_as("kJ**2/mol**2"))).round(2)
reaction_df.to_csv('out1.csv')