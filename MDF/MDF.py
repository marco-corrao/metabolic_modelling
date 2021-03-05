# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from equilibrator_api import ComponentContribution, Q_
from equilibrator_pathway import ThermodynamicModel
import numpy as np
import scipy.io
import sbtab

# %%
cc = ComponentContribution()


# %%
# Compute Standard Free energies
pp = ThermodynamicModel.from_sbtab("MDF_model_noFA_noEX.tsv", comp_contrib=cc)
mu, sigma = cc.standard_dg_prime_multi(
    pp.reactions, uncertainty_representation="fullrank", minimize_norm=True
)

standard_dg_prime_in_kJ_per_mol = mu.m_as("kJ/mol").round(2)
sigma = sigma.m_as("kJ/mol")
dg_cov = sigma @ sigma.T
dg_sigma_in_kJ_per_mol = np.sqrt(np.diag(dg_cov)).round(2)

# %%
# Run MDF and store results
sol = pp.mdf_analysis()

reaction_df = sol.reaction_df[["reaction_id", "reaction_formula"]].copy()
reaction_df["standard_dg_prime_in_kJ_per_mol"] = standard_dg_prime_in_kJ_per_mol
reaction_df["dg_sigma_in_kJ_per_mol"] = dg_sigma_in_kJ_per_mol

sbtabdoc = sbtab.SBtab.SBtabDocument()
sbtabdoc.add_sbtab(
    sbtab.SBtab.SBtabTable.from_data_frame(
        reaction_df.applymap(str), table_id="Thermodynamics", table_type="Quantity"
    )
)
sbtabdoc.write("out1.tsv")

#%%
# Save the Covariance matrix as a mat file
mdic = {'dg_mu':standard_dg_prime_in_kJ_per_mol,"dg_cov": dg_cov, "rxn_id": reaction_df["reaction_id"].to_list()}
scipy.io.savemat("dg_covariance.mat", mdic)
