# Resilience-Quantification-Msc-Thesis

This GitHub folder represents the Appendix of my Msc thesis developed at Utrecht University in collaboaration with Deltares. 

The main aim of this thesis was to quantify under one holistic value the concept of resilience to flooding in urban environments. To do so, 4 indicators (Distributional Impacts, Welfare Loss, Beyond Design Event, and Recovery Time) were chosen to be operationalized using the cities of Semarang (Indonesia) and Vientiane (Lao PDR) as case studies. This processed was done using the output of hydrological models and global datasets to focus on the study of the post-hazard response of the cities. The hydrological models used in this step of the research were ran as plugins of the main model HydroMT (https://github.com/Deltares/hydromt) and these were SFINCS (https://sfincs.readthedocs.io/en/latest/) and Delft-FIAT (https://publicwiki.deltares.nl/display/DFIAT/Delft-FIAT+Home). Once the indicators were quantified, the second part of the research consisted into stepping from 4 values (one per indicator) to a single one able to represent the concept of urban flood resilience. This was done by fitting an equation (found in literature by Bertilsson et al., 2019) to the current research by clustering the indicators together and assigning weights to increase the level of flexibility to single case studies. Once the equation was solved for the different types of flooding (pluvial, fluvial, and coastal) in the 2 case studies a sensitivity analysis was performed to test the change of the weights of the 4 indicators and by finding the 95 % of interval of the final resilience values.  

Therefore, the workflow created in this thesis has a threefold means of application:
1 - It could be used as an initial input to start including resilience in risk assessment analysis for disaster management
2 - Cities might be compared through lists to evaluate which ones are more vulnerable and in need of, for instance, financial help
3 - This methodology is able to point out in which fields better adaptation measurments might be applied as responses to flooding hazards.

Future research might create newer and more accurate models studying the extent of the floods to test once again the results obtained from the current study. However, it is relevant to highlight that the main aim of this study was to create a replicable methodology able to assess flooding resilience in urban environments on a global scale.

The complete document can be found in this folder togheter with the list of appendices showing the applications to the different softwares used during the research to guarantee replicability and transparency. 

Following, a list of the application used is given with the details describing the content of each subfolder.

Appendix A - QGIS Applications: this folder contains all the shapefiles, vector and raster layers created during the research.
Appendix B - SFINCS Applications: this folder contains all the notebooks and input files needed to produce the results of the Recovery Time Indicator and to model the Semarang Pluvial Flooding event with 50 - years Return Period. 
Appendix C - Delft-FIAT Applications: this folder contains the input prompt comands given to the model as well has the input files and the outputs.
Appendix D - Excel Applications: the folder provides the file used as a dashboard of each case study and the sensitivity analysis.
Appendix E - RStudio Applications: the folder contains the file used to create the descriptive statistics used to show the results.
