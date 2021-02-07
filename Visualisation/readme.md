In order to visualise the reduced model in a unified map in Escher, long linear pathways (>2 reactions) were drawn in the map as compressed pathways,i.e. toy reaction accounting for any net consumption or production through the pathway. Such toy reactions can be distinguished from 'real'reactions by a (CP) at the end of their reactions ID.

For this reason, in order to be visualised correctly (without gaps), FBA results need to be converted through the function flux_visulisation() defined in the new_functions.ipynb Notebook. This will assign to each compressed pathway the flux of the last reaction int he pathway. 

The model visualisation_model.json contain this toy reactions, and can be loaded with the map in Escher, but cannot be used for FBA.