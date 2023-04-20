# Predicting GDP Per Capita By Data Blending of Micro and Macro Economic Indicators
Datasets were obtained from two open sources that are publically available summarized here

1. IMF data: rows = 43,417,651 (feature/country/ year combination), Number of Features=	13,457 (after processing), Size =	4.78 GB, Source=	https://data.imf.org/datasets

2. World Bank data: rows=	383,572, Number of features=	1,444	size = 204MB	source = https://datacatalog.worldbank.org/home

Since the data is massive in size and cannot all be upload here, we have provided sample data in case one needs to test the scripts.
How to run our code:

There are 3 folders with notebooks 

A. Capstone_WBData:  Contains an analysis of World Bank Data – Simply Clone the folder with sample data ‘WDISeries.csv', WDICountry.csv, WDIData.csv. Then run WorldBank_models_V1.ipynb' : This script cleans, transforms, provides analysis and models. It also writes a dataframe that has gone through multicolinearity computation as features to the overall script of the project. 

B. Capstone_IMFData:  Contains an analysis of IMF Data. The data is contained in multiple files. ‘CapstoneIMF_Pipeline.ipynb’ runs main data accumulation code and initial model fitting. PCA Process.ipynb performs PCA analysis and integrates the components into models. Variance detection.ipynb performs the VIF calculations. Main files are feature_yearly_data_2023-04-04.csv, file_metadata_df.csv, imf_gdp_pc_target_values.csv, test_feature_data_2021_2023-04-04.csv, train_feature_data_2020_2023-04-04.csv, vif_train_model_df_2020_2023-04-04.csv, as well as files which could be uploaded in each data set folder.

C. CapstoneWB_IMF  The Capstone_WBData  and Capstone_IMFData are used in this main project folder. The data samples are given : WB_feature_vif.csv and feature_yearly_data_2023-04-01.csv. Then run the model script : Capstone_models_WB_IMF.ipynb
