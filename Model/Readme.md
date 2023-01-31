iCH360 metabolic model in COBRA (.json) and SBML (.xml) format.
- The original COBRA annotation was modified to remove duplicate KEGG annotation
- The stoichiometry of cytochromic reactions was changed from the original COBRA version to avoid having non-integer (0.5) stoichiometric coefficients for oxygen.

Changelog with respect to iML1515
-Transhydrogenase (THD2pp) translocates one proton instead of two (Bizouarn et al., Biochim Biophys Acta 2005). This error also exists in the core model
-Homoserine dehydrogenase (HSDy) produces homoserine from aspartate-semialdehyde irreversibly (He et al., Metabolic Engineering 2020)
-Isocitrate lyase (ICL) is reversible (MacKintosh & Nimmo, Biochem J 1988).