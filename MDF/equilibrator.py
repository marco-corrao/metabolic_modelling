from equilibrator_api import ComponentContribution
from equilibrator_api.model import StoichiometricModel
import numpy as np
import scipy.io
from scipy.linalg import pinvh
import sbtab
import pandas as pd
import sys


cc = ComponentContribution()

# Load the model from the SBtab file
pp = StoichiometricModel.from_sbtab("MDF_model_noFA_noEX.tsv", comp_contrib=cc)

# recalculate the standard_dg using the 'minimize_norm' option
mu, dg_cov = cc.standard_dg_prime_multi(
    pp.reactions, uncertainty_representation="cov", minimize_norm=True
)
standard_dg_prime_in_kJ_per_mol = mu.m_as("kJ/mol").round(2)
dg_sigma_in_kJ_per_mol = np.sqrt(np.diag(dg_cov.m_as("kJ**2/mol**2"))).round(2)

# write the estimated values to a SBtab ifle (out1.tsv)
reaction_df = pd.DataFrame(
    zip(pp.reaction_ids, pp.reaction_formulas, standard_dg_prime_in_kJ_per_mol, dg_sigma_in_kJ_per_mol),
    columns=["reaction_id", "reaction_formula", "standard_dg_prime_in_kJ_per_mol", "dg_sigma_in_kJ_per_mol"]
)
sbtabdoc = sbtab.SBtab.SBtabDocument()
sbtabdoc.add_sbtab(
    sbtab.SBtab.SBtabTable.from_data_frame(
        reaction_df.applymap(str), table_id="Thermodynamics", table_type="Quantity"
    )
)
sbtabdoc.write("out1.tsv")

# Save the Precision matrix to dg_precision.mat in the Matlab binary file format
try:
    _, dg_precision = cc.standard_dg_prime_multi(
        pp.reactions, uncertainty_representation="precision"
    )
    dg_precision = dg_precision.m_as("mol**2/kJ**2")
except ValueError:
    sys.stderr.write(
        "uncertainty_representation = 'precision' is not implemented "
        "in this version of equilibrator-api.\ninverting the covariance matrix "
        "using the pseudoinverse function of scipy."
    ) 
    dg_precision = pinvh(dg_cov.m_as("kJ**2/mol**2"))

mdic = {"dg_precision": dg_precision, "rxn_id": list(pp.reaction_ids)}
scipy.io.savemat("dg_precision.mat", mdic)

