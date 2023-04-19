import pandas as pd
import numpy as np

import random

import time
start_time = time.perf_counter()

from datetime import date

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
import math

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler

from sklearn.inspection import permutation_importance

from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import altair as alt

import altair as alt
from vega_datasets import data

world = alt.topo_feature(data.world_110m.url, feature='countries')

import warnings
# SettingWithCopyWarning: 
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead#
# warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame.")
warnings.filterwarnings("ignore", message='''A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead''')

#https://data.imf.org/?sk=388dfa60-1d26-4ade-b505-a05a558d9a42&sId=1479329132316

course_file_path = '/Users/stephenontko/Documents/University of Michigan/UMSI/UMSI MADS/UMSI MADS 2023/SIADS 699 Capstone/Data/IMF/'

imf_target_values_import = pd.read_csv('/Users/stephenontko/Documents/University of Michigan/UMSI/UMSI MADS/UMSI MADS 2023/SIADS 699 Capstone/Data/imf_gdp_pc_target_values.csv')

file_model_df = imf_target_values_import.copy()

file_model_df = file_model_df#.iloc[1:,:]
file_model_df['Country Name'] = file_model_df['GDP per capita, current prices\n (U.S. dollars per capita)']
file_model_df.index = file_model_df['Country Name']
file_model_df=file_model_df.drop(columns = ['Country Name','GDP per capita, current prices\n (U.S. dollars per capita)'])
file_model_df.set_axis([int(float(x)) for x in file_model_df.columns],axis = 'columns', inplace=True)

country_exclusion_list = [
'Africa (Region)',
'Asia and Pacific',
'Australia and New Zealand',
'Caribbean',
'Central America',
'Central Asia and the Caucasus',
'East Asia',
'Eastern Europe ', 
'Europe',
'Middle East (Region)',
'North Africa',
'North America',
'Pacific Islands ', 
'South America',
'South Asia',
'Southeast Asia',
'Sub-Saharan Africa (Region) ',
'Western Europe',
'Western Hemisphere (Region)',
'ASEAN-5',
'Advanced economies',
'Emerging and Developing Asia',
'Emerging and Developing Europe',
'Emerging market and developing economies',
'Euro area',
'European Union',
'Latin America and the Caribbean',
'Major advanced economies (G7)',
'Middle East and Central Asia',
'Other advanced economies',
'Sub-Saharan Africa',
'World',
'nan',
'NaN',
'©IMF, 2022'
]

model_df = file_model_df.copy()
model_df = model_df[~model_df.index.isin(country_exclusion_list)]
model_df=model_df.fillna(0).replace('no data',0)
model_df = model_df.loc[model_df.index.notna()]

completed_file_dfs = [
          'BOPAGG_02-04-2023 07-59-39-92_timeSeries/BOPAGG_02-04-2023 07-59-39-92_timeSeries.csv'#3
         ,'CDIS_01-15-2023 18-58-45-21_timeSeries/CDIS_01-15-2023 18-58-45-21_timeSeries.csv'
         ,'CPI_02-07-2023 16-19-35-85_timeSeries/CPI_02-07-2023 16-19-35-85_timeSeries.csv'#6
         ,'ED_02-07-2023 04-41-04-68_timeSeries/ED_02-07-2023 04-41-04-68_timeSeries.csv'#8
         ,'EQ_02-04-2023 18-05-06-26_timeSeries/EQ_02-04-2023 18-05-06-26_timeSeries.csv'#9
         ,'FAS_02-04-2023 13-20-29-76_timeSeries/FAS_02-04-2023 13-20-29-76_timeSeries.csv'#10
         ,'FDI_07-14-2022 10-45-25-27_timeSeries/FDI_07-14-2022 10-45-25-27_timeSeries.csv'#11
         ,'FISCALDECENTRALIZATION_11-13-2022 06-45-12-25_timeSeries/FISCALDECENTRALIZATION_11-13-2022 06-45-12-25_timeSeries.csv'
         ,'FM_02-07-2023 04-43-40-20_timeSeries/FM_02-07-2023 04-43-40-20_timeSeries.csv'#13
         ,'FSI_02-03-2023 08-19-04-78_timeSeries/FSI_02-03-2023 08-19-04-78_timeSeries.csv'#14
         ,'FSIRE_02-04-2023 08-23-40-70_timeSeries/FSIRE_02-04-2023 08-23-40-70_timeSeries.csv'#15
         ,'GENDER_EQUALITY_01-25-2023 14-59-11-92_timeSeries/GENDER_EQUALITY_01-25-2023 14-59-11-92_timeSeries.csv'#16
         ,'GFSCOFOG_02-07-2023 04-44-34-74_timeSeries/GFSCOFOG_02-07-2023 04-44-34-74_timeSeries.csv'
         ,'GFSE_02-07-2023 04-44-54-86_timeSeries/GFSE_02-07-2023 04-44-54-86_timeSeries.csv'#18
         ,'GFSFALCS_02-07-2023 04-45-22-02_timeSeries/GFSFALCS_02-07-2023 04-45-22-02_timeSeries.csv'
         ,'GFSIBS_02-06-2023 14-42-18-39_timeSeries/GFSIBS_02-06-2023 14-42-18-39_timeSeries.csv'
         ,'GFSMAB_02-07-2023 04-46-16-64_timeSeries/GFSMAB_02-07-2023 04-46-16-64_timeSeries.csv'
         ,'GFSR_02-07-2023 04-46-55-75_timeSeries/GFSR_02-07-2023 04-46-55-75_timeSeries.csv'
         ,'GFSSSUC_02-07-2023 04-47-28-17_timeSeries/GFSSSUC_02-07-2023 04-47-28-17_timeSeries.csv'
         ,'HPDD_04-11-2020 23-37-06-60_timeSeries/HPDD_04-11-2020 23-37-06-60_timeSeries.csv'#24
         ,'IFS_02-07-2023 12-51-31-57_timeSeries/IFS_02-07-2023 12-51-31-57_timeSeries.csv'#25
         ,'IRFCL_02-07-2023 08-35-05-78_timeSeries/IRFCL_02-07-2023 08-35-05-78_timeSeries.csv'#26
         ,'PCTOT_01-31-2023 17-02-07-08_timeSeries/PCTOT_01-31-2023 17-02-07-08_timeSeries.csv'#29
         ,'WHDREO_02-06-2023 03-53-04-14_timeSeries/WHDREO_02-06-2023 03-53-04-14_timeSeries.csv' #31
         ,'WoRLD_02-08-2023 01-04-45-48_timeSeries/WoRLD_02-08-2023 01-04-45-48_timeSeries.csv' #32
         ] 

# feature_df = pd.DataFrame(columns=['Country Name','Country Code','Indicator Name'
#                                       ,'Indicator Code','Year','Feature Value','filename'])
train_year = 2020
inputdate = '2023-04-04'

train_y_df =model_df[[train_year+1]].replace('no data',0).fillna(0)
test_y_df =model_df[[train_year+2]].replace('no data',0).fillna(0)

completed_file_dfs_sample = [
              'BOPAGG_02-04-2023 07-59-39-92_timeSeries/BOPAGG_02-04-2023 07-59-39-92_timeSeries.csv'#3
         ,'CDIS_01-15-2023 18-58-45-21_timeSeries/CDIS_01-15-2023 18-58-45-21_timeSeries.csv'
        #  ,'CPI_02-07-2023 16-19-35-85_timeSeries/CPI_02-07-2023 16-19-35-85_timeSeries.csv'#6
        #  ,'ED_02-07-2023 04-41-04-68_timeSeries/ED_02-07-2023 04-41-04-68_timeSeries.csv'#8
        # ,'FISCALDECENTRALIZATION_11-13-2022 06-45-12-25_timeSeries/FISCALDECENTRALIZATION_11-13-2022 06-45-12-25_timeSeries.csv'
        # ,'FM_02-07-2023 04-43-40-20_timeSeries/FM_02-07-2023 04-43-40-20_timeSeries.csv'#13
        #  ,'FSI_02-03-2023 08-19-04-78_timeSeries/FSI_02-03-2023 08-19-04-78_timeSeries.csv'#14
        #  ,'FSIRE_02-04-2023 08-23-40-70_timeSeries/FSIRE_02-04-2023 08-23-40-70_timeSeries.csv'#15
        #  ,'GENDER_EQUALITY_01-25-2023 14-59-11-92_timeSeries/GENDER_EQUALITY_01-25-2023 14-59-11-92_timeSeries.csv'#16
        #  ,'GFSCOFOG_02-07-2023 04-44-34-74_timeSeries/GFSCOFOG_02-07-2023 04-44-34-74_timeSeries.csv'
        #  ,'GFSE_02-07-2023 04-44-54-86_timeSeries/GFSE_02-07-2023 04-44-54-86_timeSeries.csv'#18
        #  ,'GFSFALCS_02-07-2023 04-45-22-02_timeSeries/GFSFALCS_02-07-2023 04-45-22-02_timeSeries.csv'
        #  ,'GFSIBS_02-06-2023 14-42-18-39_timeSeries/GFSIBS_02-06-2023 14-42-18-39_timeSeries.csv'
        #  ,'GFSMAB_02-07-2023 04-46-16-64_timeSeries/GFSMAB_02-07-2023 04-46-16-64_timeSeries.csv'
        #  ,'GFSR_02-07-2023 04-46-55-75_timeSeries/GFSR_02-07-2023 04-46-55-75_timeSeries.csv'
        #  ,'GFSSSUC_02-07-2023 04-47-28-17_timeSeries/GFSSSUC_02-07-2023 04-47-28-17_timeSeries.csv'
        #  ,'HPDD_04-11-2020 23-37-06-60_timeSeries/HPDD_04-11-2020 23-37-06-60_timeSeries.csv'#24
        #  ,'IFS_02-07-2023 12-51-31-57_timeSeries/IFS_02-07-2023 12-51-31-57_timeSeries.csv'#25
        #  ,'IRFCL_02-07-2023 08-35-05-78_timeSeries/IRFCL_02-07-2023 08-35-05-78_timeSeries.csv'#26
        #  ,'PCTOT_01-31-2023 17-02-07-08_timeSeries/PCTOT_01-31-2023 17-02-07-08_timeSeries.csv'#29
        #  ,'WoRLD_02-08-2023 01-04-45-48_timeSeries/WoRLD_02-08-2023 01-04-45-48_timeSeries.csv' #32
            ]

k10_all_lr_results = pd.DataFrame(columns = ['r_score','rmse_score'])
k10_all_dt_results = pd.DataFrame(columns = ['r_score','rmse_score'])
k10_all_mlp_results = pd.DataFrame(columns = ['r_score','rmse_score'])
k10_all_rf_results = pd.DataFrame(columns = ['r_score','rmse_score'])

pca_train_df = pd.DataFrame(index = train_y_df.index)
pca_test_df = pd.DataFrame(index = test_y_df.index)

meta_dict = {}
for file_name in completed_file_dfs:
# for file_name in completed_file_dfs_sample:
    # start_time = time.perf_counter()
    print(file_name)
    print(course_file_path+file_name.split('/')[0]+'/'+file_name.split('_')[0])
    df_file = pd.read_csv(course_file_path+file_name.split('/')[0]+'/'+file_name.split('_')[0]+'_df.csv')
    meta_of_df = pd.read_csv(course_file_path+file_name.split('/')[0]+'/metadata_'+file_name.split('/')[1])
    meta_subject = meta_of_df[meta_of_df['Metadata Attribute']=='Dataset']['Metadata Value']
    print(meta_subject)
    meta_dict[file_name.split('_')[0]] = meta_subject[1]

    meta_dict['ED'] = 'Export Diversification (ED)'
    meta_dict['EQ'] = 'Export Quality (EQ)'
    meta_dict['FSI'] = 'Financial Soundness Indicators (FSI)'
    meta_dict['FSIRE'] = 'Financial Soundness Indicators Reporting Entities (FSI)'
    meta_dict['GFSCOFOG'] = 'Government Financial Statistics Expenditure by Function of Government (GFSCOFOG)'
    meta_dict['GFSE'] = 'Government Financial Statistics Expense (GFSE)'
    meta_dict['GFSFALCS'] = 'Government Financial Statistics Financial Assets and Liabilities Counterpart (GFSFALCS)'
    meta_dict['GFSIBS'] = 'Government Financial Statistics Integrated Balance Sheet (GFSIBS)'
    meta_dict['GFSMAB'] = 'Government Financial Statistics Main Aggregates and Balances (GFSMAB)'
    meta_dict['GFSR'] = 'Government Financial Statistics Revenue (GFSR)'
    meta_dict['GFSSSUC'] = 'Government Financial Statistics Statement of Sources and Uses of Cash (GFSSSUC)'

    file_index = completed_file_dfs.index(file_name)
    print(file_index)

    # print(df_file)
    # print(df_file.columns)
    df_file = df_file[['Country Name','Country Code','Indicator Name','Indicator Code','Year','Feature Value','filename']]
    column_exclusion_list = ['Xgdppc']
    df_file = df_file[~df_file['Indicator Code'].isin(column_exclusion_list)]
    # print(df_file)
    # print(df_file.columns)

# def process_data_for_model(train_year, inputdate):
#     print('training year: ',train_year, ' date: ',inputdate)
    # data_df = pd.read_csv(course_file_path+'feature_yearly_data_'+str(inputdate)+'.csv')
    # data_df=data_df[['Country Name','Indicator Code','Year','Feature Value']]
    train_data_df = df_file[df_file['Year'] <= train_year]
    train_data_df = train_data_df[['Country Name','Indicator Code','Feature Value']]
    train_data_df = pd.pivot_table(train_data_df, values = 'Feature Value'
                                   ,index = ['Country Name']
                                   ,columns = ['Indicator Code']
                                   ,aggfunc=np.mean).fillna(0).astype(float)

    # train_data_df = np.read_csv(course_file_path+'train_feature_data_'+str(train_year)+'_'+str(date.today())+'.csv')
    # print(train_data_df)
    # print(train_data_df.columns)

    test_data_df = df_file[df_file['Year'] <= train_year+1]
    test_data_df = test_data_df[['Country Name','Indicator Code','Feature Value']]
    test_data_df = test_data_df[['Country Name','Indicator Code','Feature Value']]
    test_data_df = pd.pivot_table(test_data_df, values = 'Feature Value'
                                   ,index = ['Country Name']
                                   ,columns = ['Indicator Code']
                                   ,aggfunc=np.mean).fillna(0).astype(float)
    
    # print(train_data_df)
    # print(train_data_df.columns)

    train_data_df = train_y_df.merge(train_data_df  , left_on='Country Name', right_on='Country Name', how='inner')#.reset_index()
    train_X = train_data_df.drop(columns =[train_year+1])
    
    # print(train_X)
    # print(train_X.columns)

    # print(test_data_df)
    # print(test_data_df.columns)
    
    test_data_df = test_y_df.merge(test_data_df  , left_on='Country Name', right_on='Country Name', how='inner')#.reset_index()
    test_X = test_data_df.drop(columns =[train_year+2])
    test_X = test_data_df
    
    # print(test_data_df)
    # print(test_data_df.columns)

    feature_list = [x for x in train_X.columns]
    feature_list = [x for x in feature_list if x in test_X.columns]

    train_X = train_X[feature_list].astype(float)
    test_X = test_X[feature_list].astype(float)

    print(train_X.shape)

    # if file_index == 1:
    #     train_X = train_X.iloc[:, :72]
    #     test_X = train_X.iloc[:, :72]
    # elif file_index == 2:
    #     train_X = train_X.iloc[:, 72:]
    #     test_X = train_X.iloc[:, 72:]

    train_y = train_data_df[[train_year+1]].astype(float)
    test_y = test_data_df[[train_year+2]].astype(float)

    # train_X_scaled = StandardScaler().fit(train_X).transform(train_X)#.astype(float)
    # train_X_scaled = add_constant(train_X_scaled)

    # test_X_scaled = StandardScaler().fit(test_X).transform(test_X)#.astype(float)
    # test_X_scaled = add_constant(test_X_scaled)

    train_y_label_encoded = preprocessing.LabelEncoder().fit_transform(train_y)
    test_y_label_encoded = preprocessing.LabelEncoder().fit_transform(test_y)

    ci = .95
    svd_solver = 'auto'#'full'
    nca_pca = PCA(n_components=ci, svd_solver=svd_solver)
    # nca_pca.fit(nca_pca.transform(train_X), train_y_label_encoded)
    nca_pca.fit(train_X, train_y_label_encoded)

    print(nca_pca.get_params(deep=True))

    # print(dir(nca_pca))

    # print(nca_pca.transform(train_X))
    print(nca_pca.transform(train_X).shape)
    # print(nca_pca.components_)
    print(nca_pca.components_.shape)

    folder_train_pca_df = pd.DataFrame(nca_pca.transform(train_X)
                                 ,columns = [file_name.split('_')[0]+'_component_'+str(x) for x in range(nca_pca.components_.shape[0])]
                                 ,index=train_X.index)
    folder_test_pca_df = pd.DataFrame(nca_pca.transform(test_X)
                                 ,columns = [file_name.split('_')[0]+'_component_'+str(x) for x in range(nca_pca.components_.shape[0])]
                                 ,index=test_X.index)

    pca_train_df = pd.concat([pca_train_df, folder_train_pca_df],axis= 1).fillna(0)
    pca_test_df = pd.concat([pca_test_df, folder_test_pca_df],axis= 1).fillna(0)
    
kf10 = KFold(n_splits = 10)
kf10.get_n_splits(train_X)

print(train_X)
print(test_X)

k10_all_lr_results = pd.DataFrame(columns = ['r_score','rmse_score'])
k10_all_dt_results = pd.DataFrame(columns = ['r_score','rmse_score'])
k10_all_mlp_results = pd.DataFrame(columns = ['r_score','rmse_score'])
k10_all_rf_results = pd.DataFrame(columns = ['r_score','rmse_score'])

pca_test_df = pca_test_df.merge(test_y  , left_on='Country Name', right_on='Country Name', how='inner')#.reset_index()
pca_test_df.drop(columns=[train_year+2], inplace=True)
for i, (train_index, test_index) in enumerate(kf10.split(train_X)):
    print(f"Fold {i}:")

    lr_rgr = LinearRegression().fit(pca_train_df.iloc[train_index,:], train_y.iloc[train_index,:])
    lr_init_regr_score = lr_rgr.score(pca_train_df.iloc[test_index,:],train_y.iloc[test_index,:])
    # print(pca_test_df)
    # print(pca_test_df.columns)
    lr_regr_predict = lr_rgr.predict(pca_test_df)

    lr_rmse = round(math.sqrt(mean_squared_error(test_y, lr_regr_predict)),3)
    # print(lr_rgr.__class__.__name__, lr_init_regr_score, lr_rmse)

    k10_all_lr_results = pd.concat([k10_all_lr_results,pd.DataFrame([[lr_init_regr_score,lr_rmse]],columns = ['r_score','rmse_score'])],axis=0)

        # dt_rgr = DecisionTreeRegressor(max_features= 'auto', max_depth= 25, ccp_alpha = .25).fit(train_X.iloc[train_index,:], train_y.iloc[train_index,:])
    dt_rgr = DecisionTreeRegressor(max_features= 'auto', max_depth= 25, ccp_alpha = .25).fit(pca_train_df.iloc[train_index,:], train_y.iloc[train_index,:])
        # print(dt_rgr.__class__.__name__)
        # dt_init_regr_score = dt_rgr.score(train_X.iloc[test_index,:],train_y.iloc[test_index,:])
    dt_init_regr_score = dt_rgr.score(pca_train_df.iloc[test_index,:],train_y.iloc[test_index,:])
        # dt_regr_predict = dt_rgr.predict(test_X)
    dt_regr_predict = dt_rgr.predict(pca_test_df)
    rmse = round(math.sqrt(mean_squared_error(test_y, dt_regr_predict)),3)
        # print(dt_rgr.__class__.__name__, dt_init_regr_score, rmse)

    k10_all_dt_results = pd.concat([k10_all_dt_results,pd.DataFrame([[dt_init_regr_score,rmse]],columns = ['r_score','rmse_score'])],axis=0)
    
        # mlp_rgr = MLPRegressor(learning_rate='adaptive',alpha=.5).fit(train_X.iloc[train_index,:], train_y.iloc[train_index,:])
    mlp_rgr = MLPRegressor().fit(folder_train_pca_df.iloc[train_index,:], train_y.iloc[train_index,:])
        # mlp_rgr_score = mlp_rgr.score(train_X.iloc[test_index,:],train_y.iloc[test_index,:])
    mlp_rgr_score = mlp_rgr.score(folder_train_pca_df.iloc[test_index,:],train_y.iloc[test_index,:])
        # print(mlp_rgr.__class__.__name__)
        # print(mlp_rgr_score)
        # mlp_rgr_predict = mlp_rgr.predict(test_X)
    mlp_rgr_predict = mlp_rgr.predict(folder_test_pca_df)
    rmse = round(math.sqrt(mean_squared_error(test_y, mlp_rgr_predict)),3)
    # print(rmse)

    k10_all_mlp_results = pd.concat([k10_all_mlp_results,pd.DataFrame([[mlp_rgr_score,rmse]],columns = ['r_score','rmse_score'])],axis=0)
        # rf_rgr = RandomForestRegressor().fit(train_X.iloc[train_index,:], train_y.iloc[train_index,:])
    rf_rgr = RandomForestRegressor().fit(folder_train_pca_df.iloc[train_index,:], train_y.iloc[train_index,:])
        # rf_rgr_score = mlp_rgr.score(train_X.iloc[test_index,:],train_y.iloc[test_index,:])
    rf_rgr_score = rf_rgr.score(folder_train_pca_df.iloc[test_index,:],train_y.iloc[test_index,:])
    # print(rf_rgr.__class__.__name__)
    # print(rf_rgr_score)
        # rf_rgr_predict = mlp_rgr.predict(test_X)
    rf_rgr_predict = rf_rgr.predict(folder_test_pca_df)
    rmse = round(math.sqrt(mean_squared_error(test_y, rf_rgr_predict)),3)
    # print(rmse)

    k10_all_rf_results = pd.concat([k10_all_rf_results,pd.DataFrame([[rf_rgr_score,rmse]],columns = ['r_score','rmse_score'])],axis=0)
    # k10_fold_pred_results = pd.concat([k10_fold_pred_results,pd.DataFrame(np.array(dt_regr_predict).reshape(-1,1),columns = [i],index=test_y.index)],axis=1)

    # print(lr_rgr.__class__.__name__)

    # k10_all_lr_results=pd.concat([k10_all_lr_results, pd.DataFrame([[k10_fold_lr_results.mean(axis=0)[0],k10_fold_lr_results.mean(axis=0)[1]]]
                                                                #    ,columns = ['r_score','rmse_score'])],axis=0)
                                  
    # print(dt_rgr.__class__.__name__)

    # k10_all_dt_results=pd.concat([k10_all_dt_results, pd.DataFrame([[k10_fold_dt_results.mean(axis=0)[0],k10_fold_dt_results.mean(axis=0)[1]]]
                                                                #    ,columns = ['r_score','rmse_score'])],axis=0)

    # print(mlp_rgr.__class__.__name__)

    # k10_all_mlp_results=pd.concat([k10_all_mlp_results, pd.DataFrame([[k10_fold_mlp_results.mean(axis=0)[0],k10_fold_mlp_results.mean(axis=0)[1]]]
                                                                #    ,columns = ['r_score','rmse_score'])],axis=0)
                                   
    # print(rf_rgr.__class__.__name__)  

    # k10_all_rf_results=pd.concat([k10_all_rf_results, pd.DataFrame([[k10_fold_rf_results.mean(axis=0)[0],k10_fold_rf_results.mean(axis=0)[1]]]
                                                                #    ,columns = ['r_score','rmse_score'])],axis=0)

    # print(k10_fold_lr_results )
    # print(k10_fold_dt_results )
    # print(k10_fold_mlp_results )
    # print(k10_fold_rf_results )

print(k10_all_lr_results )
print(k10_all_dt_results )
print(k10_all_mlp_results )
print(k10_all_rf_results )

print(k10_all_lr_results.mean(axis=0))    
print(k10_all_dt_results.mean(axis=0))    
print(k10_all_mlp_results.mean(axis=0))    
print(k10_all_rf_results.mean(axis=0))    
