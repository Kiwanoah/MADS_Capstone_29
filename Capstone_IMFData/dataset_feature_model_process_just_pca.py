import pandas as pd
import numpy as np

import random

import time
start_time = time.perf_counter()

from datetime import date

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeRegressor

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
        #  ,'WHDREO_02-06-2023 03-53-04-14_timeSeries/WHDREO_02-06-2023 03-53-04-14_timeSeries.csv' #31
        #  ,'WoRLD_02-08-2023 01-04-45-48_timeSeries/WoRLD_02-08-2023 01-04-45-48_timeSeries.csv' #32
            ]

iso_country_codes = {
    'Afghanistan':4,
	'Albania':8,	
	'Algeria':12,
	'American Samoa':16,
	'Andorra':20	,
	'Angola':24,
	'Antigua and Barbuda':28,
	'Argentina':32,
    'Armenia':51,
    'Aruba':533,
	'Australia':36,
	'Austria':40,
    'Azerbaijan':31,
    'Bahamas, The':44,
    'Bahrain':48,	
	'Bangladesh':50,
    'Barbados':52,
    'Belarus':112,
	'Belgium':56,
    'Belize':84,
    'Benin':204,
	'Bermuda':60,
	'Bhutan':64,	
	'Bolivia':68,
	'Bosnia and Herzegovina':70,
	'Botswana':72,
    'Brazil':76,
    'Brunei Darussalam':96,
	'Bulgaria':100,
    'Burkina Faso':854,
    'Burma':104,
    'Burundi':108,
    'Cabo Verde':132,
	'Cambodia':116,
	'Cameroon':120,
    'Canada':124,
    'Central African Republic':140,
    'Chad':148,
    'Chile':152,
    "China, People's Republic of":156,
	'Colombia':170,
	'Comoros':174,
	'Congo, Republic of ':178,	
	'Congo, Dem. Rep. of the':180,
    'Costa Rica':188,
    "Côte d'Ivoire":384,
	'Croatia':191,	
	'Cuba':192,	
	'Cyprus':196,	
	'Czech Republic':203,
    'Denmark':208,	
    'Djibouti':262,
	'Dominica':212,
	'Dominican Republic':214,
	'Ecuador':218,
    'Egypt':818,
	'El Salvador':222,
	'Equatorial Guinea':226,
	'Eritrea':232,
	'Estonia':233,
    'Eswatini':748,
    'Ethiopia':231,
	'Falkland Islands':238,
	'Faroe Islands':234,
	'Fiji':242,
    'Finland':246,
    'France':250,
	'Gabon':266,
	'Gambia, The':270,	
	'Georgia':268,
    'Germany':276,
    'Ghana':288,
    'Greece':300,
	'Grenada':308,
	'Guadeloupe':312,
	'Guam':316,
	'Guatemala':320,
	'Guernsey':831,
	'Guinea':324,
	'Guinea-Bissau':624,
	'Guyana':328,
    'Haiti':332,
    'Honduras':340,
    'Hong Kong SAR':344,
    'Hungary':348,
    'Iceland':352,
    'India':356,
	'Indonesia':360,
	'Iran':364,
	'Iraq':368,
	'Ireland':372,
	'Israel':376,
	'Italy':380,
	'Jamaica':388,
	'Japan':392,
	'Jordan':400,
	'Kazakhstan':398,
	'Kenya':404,
	'Kiribati':296,
	"Korea (Democratic People's Republic of)":408,
	"Korea, Republic of":410,
    'Kosovo':None,
	'Kuwait':414,
	'Kyrgyz Republic':417,
	"Lao P.D.R.":418,	
	'Latvia':428,
	'Lebanon':422,
	'Lesotho':426,
	'Liberia':430,
	'Libya':434,
	'Liechtenstein':438,
	'Lithuania':440,
	'Luxembourg':442,
	'Macao SAR':446,
	'Madagascar':450,
	'Malawi':454,
	'Malaysia':458,
	'Maldives':462,
	'Mali':466,
	'Malta':470,
	'Marshall Islands':584,
	'Mauritania':478,	
	'Mauritius':480,
	'Mexico':484,
	'Micronesia, Fed. States of':583,
	'Moldova':498,
	'Monaco':492,
	'Mongolia':496,
	'Montenegro':499,
	'Montserrat':500,
	'Morocco':504,
	'Mozambique':508,
	'Namibia':516,
	'Nauru':520,
	'Nepal':524,
	'Netherlands':528,
	# 'New Caledonia':540,
	'New Zealand':554,
	'Nicaragua':558,
	'Niger':562,
	'Nigeria':566,
	'Niue':570,
	'Norfolk Island':574,
	'North Macedonia ':807,
	'Northern Mariana Islands':580,
	'Norway':578,
    'Oman':512,
	'Pakistan':586,
	'Palau':585,
	# 'Palestine, State of':275,
	'Panama':591,
	'Papua New Guinea':598,
	'Paraguay':600,
	'Peru':604,
	'Philippines':608,
	'Pitcairn':612,
	'Poland':616,
	'Portugal':620,
	'Puerto Rico':630,
	'Qatar':634,	
	'Romania':642,
	'Russian Federation':643,
    'Rwanda':646,
    'Saint Kitts and Nevis':659,
	'Saint Lucia':662,
	'Saint Martin':663,
	'Saint Pierre and Miquelon':666,
	'Saint Vincent and the Grenadines':670,
	'Samoa':882,
	'San Marino':674,
	'São Tomé and Príncipe':678,
	'Saudi Arabia':682,
	'Senegal':686,
	'Serbia':688,
	'Seychelles':690,
	'Sierra Leone':694,
	'Singapore':702,
	'Slovak Republic':703,
	'Slovenia':705,
	'Solomon Islands':90,
	'Somalia':706,
	'South Africa':710,
	'South Georgia':239,
    'South Sudan, Republic of':728,
    'Spain':724,	
	'Sri Lanka':144,
	'Sudan':729,
	'Suriname':740,
	'Sweden':752,
	'Switzerland':756,
	'Syria':760,#Syrian Arab Republic	
	'Taiwan':158,
	'Tajikistan':762,
	'Tanzania':834,
	'Thailand':764,
	'Timor-Leste':626,
	'Togo':768,
	'Tonga':776,
	'Trinidad and Tobago':780,
	'Tunisia':788,
	'Türkiye, Republic of':792,
	'Turkmenistan':795,
	'Tuvalu':798,
	'Uganda':800,
	'Ukraine':804,
	'United Arab Emirates':784,
	'United Kingdom':826,
	'United States':840,
	'Uruguay':858,
	'Uzbekistan':860,
	'Vanuatu':548,
	'Venezuela':862,
	'Vietnam':704,
    'West Bank and Gaza':275,
	'Yemen':887,
	'Zambia':894,
	'Zimbabwe':716
    # ,'nan':None
}


def choropleth_dfs(test_y, train_year, predictions):
    choropleth_df = test_y.copy().astype(float)
    choropleth_df['Country Name'] = choropleth_df.index
    choropleth_df['GDP Per Capita'] = [x for x in predictions]
    choropleth_df['id'] = [iso_country_codes[x] for x in choropleth_df.index]

    choropleth_actual_df = choropleth_df[['Country Name',train_year+2,'id']]
    choropleth_actual_df.rename(columns={train_year+2:'GDP Per Capita'},inplace=True)
    choropleth_actual_df = choropleth_actual_df[['Country Name','GDP Per Capita','id']]

    choropleth_df = choropleth_df[['Country Name','GDP Per Capita','id']]
    return choropleth_df, choropleth_actual_df

def choropleth_chart(test_y, train_year,predictions, model, model_score,rmse):
    choropleth_df, choropleth_actual_df = choropleth_dfs(test_y, train_year, predictions)
    gdpactual= alt.Chart(
        world
          , title = ["Actual GDP Per Capita"
                    ,'for Year: ' + str(train_year +2)]).mark_geoshape(stroke='black' #'white'
                                                                ,fillOpacity=1,strokeWidth=.05).transform_lookup(
    lookup='id',
    from_=alt.LookupData(data=choropleth_actual_df, key = 'id', fields=['id','Country Name','GDP Per Capita'])
    ).encode(
    # tooltip='properties.geounit:N',
    color=alt.Color('GDP Per Capita:Q'
                    ,scale= alt.Scale(domain=[
                                                # 0,100000
                                            choropleth_actual_df['GDP Per Capita'].min()
                                              ,choropleth_actual_df['GDP Per Capita'].max()
                                              ]
                                        # ,range=['red','yellow','green']
                                        ,range=['lightgreen','darkblue']
                                      ))
    , tooltip=['Country Name:N', 'GDP Per Capita:Q']

    ).project(
    type='mercator'
    ).properties(
    width=800,
    height=800
    )
    gdpactual#.show()

    gdpforecast= alt.Chart(
        world
          , title = ["GDP Per Capita Forecasting : "
                    ,'Forecast Year: ' + str(train_year +2) 
                    ,'Model: '+model.__class__.__name__+ ' Score: '+str(round(model_score,4))
                    +' RMSE: '+str(rmse)]
                    ).mark_geoshape(stroke='black' #'white'
                                    ,fillOpacity=1,strokeWidth=.05).transform_lookup(
    lookup='id',
    from_=alt.LookupData(data=choropleth_df, key = 'id', fields=['id','Country Name','GDP Per Capita'])
    ).encode(
    # tooltip='properties.geounit:N',
    color=alt.Color('GDP Per Capita:Q'
                    ,scale= alt.Scale(domain=[
                                           # 0
                                        choropleth_df['GDP Per Capita'].min()
                                              ,choropleth_df['GDP Per Capita'].max()]
                                        # ,range=['red','yellow','green']
                                        ,range=['lightgreen','darkblue']
                                      ))
    , tooltip=['Country Name:N', 'GDP Per Capita:Q']

    ).project(
    type='mercator'
    ).properties(
    width=800,
    height=800
    )
    gdpforecast#.show()
    return alt.hconcat(gdpactual, gdpforecast)#.show()

def decision_tree_kfold(train_X, train_y, test_X, test_y, code_names, max_features, max_depth, ccp_alpha, trainindex, testindex):
    # print('decision tree: ')

    # print(train_X)
    # print(train_y)

    percent_threshold = 0.00009

    dt_regr = DecisionTreeRegressor(max_features= max_features, max_depth= max_depth, ccp_alpha = ccp_alpha).fit(pd.DataFrame(train_X).iloc[trainindex,:], pd.DataFrame(train_y).iloc[trainindex,:])
    dt_regr_score = dt_regr.score(pd.DataFrame(train_X).iloc[testindex,:], pd.DataFrame(train_y).iloc[testindex,:])
    dt_regr_predict = dt_regr.predict(pd.DataFrame(test_X).iloc[:,:])

    rmse = round(math.sqrt(mean_squared_error(test_y, dt_regr_predict)),3)
    # print(dt_regr_score,rmse)

    # print([list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold])

    dt_col_results = pd.DataFrame(columns = ['model','filename', 'max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse','column','avg_column_significance', 'column_descr'])

    for sig_i, sig_column in enumerate([code_names[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]]):
        # print(round([dt_regr.feature_importances_[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]][sig_i],4)
        #       ,sig_column
        #     , metadata_df[metadata_df['Indicator Code']==sig_column]['Indicator Name'].unique()[0]
        #     ,metadata_df[metadata_df['Indicator Code']==sig_column]['filename'].unique()[0]
        #     )
        col_score = round([dt_regr.feature_importances_[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]][sig_i],4)
        col_df = pd.DataFrame([[dt_regr.__class__.__name__, df_file[df_file['Indicator Code']==sig_column]['filename'].unique()[0]
                                , max_features, max_depth, ccp_alpha, dt_regr_score, rmse ,sig_column, col_score
                               ,df_file[df_file['Indicator Code']==sig_column]['Indicator Name'].unique()[0]]]
                               ,columns = ['model','filename', 'max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse','column','avg_column_significance', 'column_descr'])
        dt_col_results = pd.concat([dt_col_results, col_df], axis = 0)
        # print(dt_col_results) 
    # dt_results = pd.concat([dt_results,dt_col_results],axis = 0)
    return dt_regr, dt_regr_predict, dt_regr_score, rmse, dt_col_results

def k_unsup(train_X_standard, train_y, test_X_standard):#, test_y, code_names, max_features, max_depth, ccp_alpha, trainindex, testindex):
    print('k unsupervised: ')

    # train_X_standard = StandardScaler().fit(train_X).transform(train_X).astype(float)
    # test_X_standard = StandardScaler().fit(test_X).transform(test_X).astype(float)

    highest_kneighbors_score = [2,-1]
    highest_kmeans_score = [2,-1,-1]

    kmeans_results = pd.DataFrame(columns = ['Cluster','Inertia','Silhouette Score','K Neighbore Score'])

    for x in range(2,22):
        # print(x)
        kneigh = KNeighborsRegressor(n_neighbors=x)
        # print(train_X_standard)
        # print(train_y)
        kneigh.fit(train_X_standard, train_y)
        # kneigh.fit(nca_pca.transform(train_X_standard), train_y_label_encoded)

        kneighscore = kneigh.score(train_X_standard,train_y) 
        # kneighscore = kneigh.score(nca_pca.transform(test_X_standard), test_y_label_encoded)
        kmeans = KMeans(n_clusters = x)
        # kmeans = KMeans(n_clusters = x,algorithm = 'elkan')
        kmeans.fit(train_X_standard, train_y)
        # kmeans.fit(nca_pca.transform(train_X_standard), train_y_label_encoded)

        kmeansscore = kmeans.score(train_X_standard, train_y)
        # kmeansscore = kmeans.score(nca_pca.transform(test_X_standard), test_y_label_encoded)

        silhouettescore = round(silhouette_score(train_X_standard, kmeans.labels_),4)
        # print(silhouettescore)
        inertia = round(kmeans.inertia_)
        cluster_df = pd.DataFrame([[x,inertia,silhouettescore,kneighscore]],columns = ['Cluster','Inertia','Silhouette Score','K Neighbor Score'])
        kmeans_results=kmeans_results.append(cluster_df)

        if kneighscore > highest_kneighbors_score[1]:
            highest_kneighbors_score[0] = x
            highest_kneighbors_score[1] = np.round(kneighscore,4)
        if silhouettescore > highest_kmeans_score[1]:
            highest_kmeans_score[0] = x
            highest_kmeans_score[1] = silhouettescore #silhouette_score(X,kmeans.labels_)
            highest_kmeans_score[2]=kmeans.inertia_

    # print(highest_kneighbors_score)
    # print(highest_kmeans_score)

    kmeans = KMeans(n_clusters = highest_kmeans_score[0])
    # kmeans = KMeans(n_clusters = highest_kmeans_score[0],algorithm = 'elkan') 
    kmeans.fit(train_X_standard,train_y)
    # kmeans.fit(nca_pca.transform(train_X_standard), train_y_label_encoded)

    kneigh = KNeighborsRegressor(n_neighbors=highest_kneighbors_score[0]).fit(train_X_standard, train_y)#, weights = 'distance', algorithm = 'ball_tree', 'kd_tree', 'brute'
    # kneigh = KNeighborsRegressor(n_neighbors=highest_kneighbors_score[0]).fit(nca_pca.transform(train_X_standard), train_y_label_encoded)#, weights = 'distance', algorithm = 'ball_tree', 'kd_tree', 'brute'

    # kneigh = KNeighborsRegressor(n_neighbors=highest_kneighbors_score[0], weights = 'distance', algorithm = 'ball_tree').fit(train_X_standard, train_y)#, weights = 'distance', algorithm = 'ball_tree', 'kd_tree', 'brute'
    # kneigh = KNeighborsRegressor(n_neighbors=highest_kneighbors_score[0], weights = 'distance', algorithm = 'brute').fit(train_X_standard, train_y)#, weights = 'distance', algorithm = 'ball_tree', 'kd_tree', 'brute'
    # kneigh = KNeighborsRegressor(n_neighbors=highest_kneighbors_score[0], algorithm = 'ball_tree').fit(train_X_standard, train_y)#, weights = 'distance', algorithm = 'ball_tree', 'kd_tree', 'brute'
    # kneigh = KNeighborsRegressor(n_neighbors=highest_kneighbors_score[0], algorithm = 'brute').fit(train_X_standard, train_y)#, weights = 'distance', algorithm = 'ball_tree', 'kd_tree', 'brute'

    # print(train_X_standard)
    # print(test_X_standard)
    kneighpredict = kneigh.predict(test_X_standard)
    # kneighpredict = kneigh.predict(nca_pca.transform(test_X_standard))

    return kneighpredict, kmeans_results, highest_kneighbors_score, highest_kmeans_score

def kneigh_results_charts(kneigh_results, highest_kneighbors_score, highest_kmeans_score):

    kneighbor_chart = alt.Chart(kneigh_results,title=["K-Neighbor Model Results"
                                                       ,'Highest K-Neighbor Score: '+str(highest_kneighbors_score[1])
                                                       + ' at cluster: ' + str(highest_kneighbors_score[0])]
                                                       ).mark_line().encode(
        x=alt.X('Cluster:N'),
        y = alt.Y('K Neighbor Score:Q')
    ).properties(
        width=400,
        height=300
    )

    silhouette_chart = alt.Chart(kneigh_results,title=[file_name.split('_')[0]
                                                        ,"K-Means Model Silhouette Results" 
                                                       ,'Highest K-Means Score: '+str(highest_kmeans_score[1])
                                                       + ' at cluster: ' + str(highest_kmeans_score[0]) ] ).mark_line().encode(
        x=alt.X('Cluster:N'),
        y = alt.Y('Silhouette Score:Q')
    ).properties(
        width=400,
        height=300
    )

    inertia_chart = alt.Chart(kneigh_results,title=["K-Means Model Inertia Results" 
                                                    ,'Highest K-Means Score: '+str(highest_kmeans_score[1])
                                                       + ' at cluster: ' + str(highest_kmeans_score[0])
                                                    ,'with an inertia of: '+str(highest_kmeans_score[2])]).mark_line().encode(
    x=alt.X('Cluster:N'),
    y = alt.Y('Inertia:Q')
   ).properties(
    width=400,
    height=300
    )

    alt.hconcat(kneighbor_chart,silhouette_chart, inertia_chart).show()

def k_unsup_choropleth(k_predictions, test_y):
    # print(df)
    k_predictions = pd.DataFrame(k_predictions)
    gdppc_chart_df = k_predictions.copy()
    gdppc_chart_df = gdppc_chart_df[[train_year+2]].rename(columns={train_year+2:'GDP Per Capita'}).replace('no data',0).fillna(0).astype(float)
    gdppc_chart_df['Country Name'] = gdppc_chart_df.index
    gdppc_chart_df['id'] = [iso_country_codes[x] for x in gdppc_chart_df.index]
    # return gdppc_chart_df

    # gdppc_chart_df = process_target_df(file_model_df, train_year + 2)[['Country Name','GDP Per Capita','id']]
    # print(gdppc_chart_df)

    gdppc= alt.Chart(
        world
          , title = ["GDP Per Capita : "
                    ,'Year: ' + str(train_year+2)]
                    ).mark_geoshape(stroke='black' #'white' 
                                    ,fillOpacity=1,strokeWidth=.05).transform_lookup(
    lookup='id',
    from_=alt.LookupData(data=gdppc_chart_df, key = 'id', fields=['id','Country Name','GDP Per Capita'])
    ).encode(
    # tooltip='properties.geounit:N',
    color=alt.Color('GDP Per Capita:Q'
                    ,scale= alt.Scale(domain=[
                                           # 0
                                        gdppc_chart_df['GDP Per Capita'].min()
                                              ,gdppc_chart_df['GDP Per Capita'].max()]
                                        ,range=['lightgreen','darkblue']
                                      ))
                ,size=alt.Size('GDP Per Capita Predicted:Q',legend = alt.Legend(title='GDP Per Capita'))                                      
    , tooltip=['Country Name:N', 'GDP Per Capita:Q']

    ).project(
    type='mercator'
    ).properties(
    width=800,
    height=800
    )
    gdppc#.show()

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

    train_X_scaled = StandardScaler().fit(train_X).transform(train_X)#.astype(float)
    train_X_scaled = add_constant(train_X_scaled)

    test_X_scaled = StandardScaler().fit(test_X).transform(test_X)#.astype(float)
    test_X_scaled = add_constant(test_X_scaled)

    train_y_label_encoded = preprocessing.LabelEncoder().fit_transform(train_y)
    test_y_label_encoded = preprocessing.LabelEncoder().fit_transform(test_y)

    # nca_pca = NeighborhoodComponentsAnalysis()
    # nca_pca.fit(train_X, train_y_label_encoded)

    # train_y_label_encoded = preprocessing.LabelEncoder().fit_transform(train_y)
    # test_y_label_encoded = preprocessing.LabelEncoder().fit_transform(test_y)
    # nca_pca.fit(train_X_standard, train_y_label_encoded)
    # nca_pca.transform(train_X_standard, train_y_label_encoded)
    # print(dir(nca_pca))
    # kneigh.fit(nca_pca.transform(train_X_standard), train_y_label_encoded)
    # print(kneigh.score(nca_pca.transform(test_X_standard), test_y_label_encoded))

    # print(train_X)
    # print(train_X.shape)
    # nca_pca = NeighborhoodComponentsAnalysis(n_components=int(train_X.shape[1]/2))

    folder_corr_df = pd.DataFrame(abs(train_X.corr()), index = train_X.columns, columns = train_X.columns)
    folder_corrMatrix = folder_corr_df.reset_index().melt('index')
    folder_corrMatrix.columns = ['var1', 'var2', 'correlation']
    folder_corrMatrix.sort_values(by=['correlation'],ascending=True,inplace=True)
    folder_corrMatrix['model'] = 'correlation matrix'

    folder_corrMatrix_heatmap = folder_corrMatrix[:5000]

    alt.Chart(folder_corrMatrix_heatmap, title = ["Feature Correlation Heatmap of "+meta_dict[file_name.split('_')[0]]+" Features: "
                    ,'Number of Principal Components: '+str(folder_corr_df.shape[0]) 
                    ,'PCA Confidence Interval: ']
                    ).mark_rect().encode(
    x='var2:O',
    y='var1:O'
    ,color='correlation:Q'
    , tooltip=['correlation:Q'],
            #    ,'var2_feature_names:O','var1_feature_names:O'],
    ).properties(height=750, width = 900)#.show()

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

    # print(nca_pca.transform(train_X).shape)
    # print(folder_train_pca_df)
    pca_train_df = pd.concat([pca_train_df, folder_train_pca_df],axis= 1).fillna(0)
    pca_test_df = pd.concat([pca_test_df, folder_test_pca_df],axis= 1).fillna(0)
 
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

# train_year = 2017
# process_data_for_model(train_year, '2023-04-04')
print(train_y_df)
print(test_y_df)
print(pca_train_df)
print(pca_test_df)

# pca_train_df.to_csv(course_file_path+'all_folders_pca_train_'+str(train_year)+'_ci'+str(ci)+'.csv', index=True)
# pca_test_df.to_csv(course_file_path+'all_folders_pca_test_'+str(train_year)+'_ci'+str(ci)+'.csv', index=True)

train_y_df =model_df[[train_year+1]].replace('no data',0).fillna(0).astype(float)
test_y_df =model_df[[train_year+2]].replace('no data',0).fillna(0).astype(float)

print(train_y_df)
print(test_y_df)
# print(pca_train_df)
# print(pca_test_df)

# cis = [x for x in [90,95,99]]
# for ci in cis:
#     print(ci)

    # pca_train_df = pd.concat([pca_train_df, folder_train_pca_df],axis= 1).fillna(0)
    # pca_test_df = pd.concat([pca_test_df, folder_test_pca_df],axis= 1).fillna(0)
 
    # pca_train_df =pd.read_csv(course_file_path+'all_folders_pca_train_'+str(train_year)+'_ci'+str(ci)+'.csv')
    # pca_test_df =pd.read_csv(course_file_path+'all_folders_pca_test_'+str(train_year)+'_ci'+str(ci)+'.csv')
    
train_X = pca_train_df
test_X = pca_test_df

kf10 = KFold(n_splits = 10)
kf10.get_n_splits(train_X)

predictions_avg = pd.DataFrame(index=train_y_df.index)
model_score_results = pd.DataFrame(columns = ['model','max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse'])
col_score_results =pd.DataFrame(columns = ['model','filename','column','avg_column_significance', 'column_descr'])

print('all pca kfold split process: ')
for i, (train_index, test_index) in enumerate(kf10.split(train_X)):
        # print(f"Fold {i}:")
    code_names = pca_train_df.columns
    max_features = 'auto'
    max_depth = 15
    ccp_alpha = .75 #float when increasing from 0 to 1, tree is pruned more to better generalize

    percent_threshold = 0.0009

    dt_regr = DecisionTreeRegressor(max_features= max_features, max_depth= max_depth, ccp_alpha = ccp_alpha).fit(pd.DataFrame(pca_train_df).iloc[train_index,:], pd.DataFrame(train_y_df).iloc[train_index,:])
    dt_regr_score = dt_regr.score(pd.DataFrame(pca_train_df).iloc[test_index,:], pd.DataFrame(train_y_df).iloc[test_index,:])
    dt_regr_predict = dt_regr.predict(pd.DataFrame(pca_test_df).iloc[:,:])

    rmse = round(math.sqrt(mean_squared_error(train_y_df, dt_regr_predict)),3)

    dt_col_results = pd.DataFrame(columns = ['model','filename', 'max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse','column','avg_column_significance', 'column_descr'])
    # print(dt_regr.feature_importances_)
    print([code_names[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]])
    for sig_i, sig_column in enumerate([code_names[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]]):
        col_score = round([dt_regr.feature_importances_[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]][sig_i],4)
        col_df = pd.DataFrame([[dt_regr.__class__.__name__, sig_column.split('_')[0]
                                , max_features, max_depth, ccp_alpha, dt_regr_score, rmse ,sig_column, col_score
                                ,meta_dict[sig_column.split('_')[0]]
                               ]]
                               ,columns = ['model','filename', 'max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse','column','avg_column_significance', 'column_descr'])
        dt_col_results = pd.concat([dt_col_results, col_df], axis = 0)

    model_score_results = pd.concat([model_score_results,dt_col_results[['model','max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse']].drop_duplicates()],axis=0)
    col_score_results = pd.concat([col_score_results,dt_col_results[['model','filename','column','avg_column_significance', 'column_descr']]],axis=0)

        # print(dt_regr_predict)
        # print(test_y_df)
    model_predictions = pd.DataFrame(dt_regr_predict, index=test_y_df.index)
    predictions_avg = pd.concat([predictions_avg, model_predictions],axis=1)

        # print(dt_col_results)
model_score_results=model_score_results.groupby(['model','max_features','max_depth','ccp_alpha'],as_index = False).mean().reset_index()
col_score_results=col_score_results.groupby(['model','filename','column', 'column_descr'],as_index = False)['avg_column_significance'].aggregate(['mean']).reset_index().sort_values(by=['mean'], ascending=False)
print(nca_pca.get_params(deep=True))
print(model_score_results)
print(col_score_results.columns)
print(col_score_results)

# col_score_results['column_descr'] = [meta_dict[x.split('_')[0]][1] for x in col_score_results['column_descr']]
col_score_results['model_avg_r_score'] = model_score_results['avg_r_score'].values[0]
col_score_results['avg_rmse'] = model_score_results['avg_rmse'].values[0]
# model_score_results.to_csv(course_file_path+'all_folders_model_score_results_'+str(train_year)+'_pcaci'+str(int(ci*100))+'.csv', index=True)
# col_score_results.to_csv(course_file_path+'all_folders_col_score_results_'+str(train_year)+'_pcaci'+str(int(ci*100))+'.csv', index=True)

end_time = time.perf_counter()
print('Process time: ',f"{end_time-start_time:0.4f} seconds {(end_time-start_time)/60:0.4f} minutes.")

vif = pd.DataFrame()
vif['feature_code'] = pca_train_df.columns
vif['VIF'] = [round(variance_inflation_factor(pca_train_df, i),4) for i in range(pca_train_df.shape[1])]
vif_all = vif.copy()
col_score_results['vif'] = [vif_all[vif_all['feature_code']==x]['VIF'] for x in col_score_results['column']]
col_score_results = col_score_results.sort_values(by=['mean'])
print(col_score_results)
print(col_score_results.columns)
print(col_score_results.shape)

vif= vif[vif['VIF'] != np.inf]
vif= vif[vif['VIF'] < 5]
vif = vif.sort_values(by=['VIF'], ascending=False).round(4)
vif['feature_name'] = [meta_dict[x.split('_')[0]] for x in vif['feature_code'] ]
vif['model'] = 'vif'

print(vif)
print(vif.shape)

corr_df = pd.DataFrame(abs(pca_train_df.corr()), index = pca_train_df.columns, columns = pca_train_df.columns)
corrMatrix = corr_df.reset_index().melt('index')
corrMatrix.columns = ['var1', 'var2', 'correlation']
corrMatrix.sort_values(by=['correlation'],ascending=True,inplace=True)
corrMatrix['model'] = 'correlation matrix'
corrMatrix['var1_feature_names'] = [meta_dict[x.split('_')[0]] for x in corrMatrix['var1']]
corrMatrix['var2_feature_names'] = [meta_dict[x.split('_')[0]] for x in corrMatrix['var2']]

corrMatrix_heatmap = corrMatrix[:5000]

alt.Chart(corrMatrix_heatmap, title = ["Feature Correlation Heatmap of PCA Features: "
                    ,'Number of Principal Components: '+str(corr_df.shape[0]) 
                    ,'PCA Confidence Interval: '+ str(ci)]
                    ).mark_rect().encode(
    x='var2:O',
    y='var1:O'
    ,color='correlation:Q'
    , tooltip=['correlation:Q','var2_feature_names:O','var1_feature_names:O'],
    ).properties(height=750, width = 900).show()

corrMatrix_fileavg=corrMatrix
corrMatrix_fileavg['var1'] = [x.split('_')[0] for x in corrMatrix_fileavg['var1']]
corrMatrix_fileavg['var2'] = [x.split('_')[0] for x in corrMatrix_fileavg['var2']]
corrMatrix_fileavg['var1_feature_names'] = [meta_dict[x.split('_')[0]] for x in corrMatrix_fileavg['var1']]
corrMatrix_fileavg['var2_feature_names'] = [meta_dict[x.split('_')[0]] for x in corrMatrix_fileavg['var2']]

corrMatrix_fileavg=corrMatrix_fileavg.groupby(['var1','var2','var1_feature_names','var2_feature_names','model'],as_index=False)['correlation'].aggregate(['mean']).reset_index().sort_values(by=['mean'], ascending=True)
corrMatrix_fileavg = corrMatrix_fileavg.reset_index()
corrMatrix_fileavg.loc[corrMatrix_fileavg.var1 == corrMatrix_fileavg.var2, 'mean'] = 1

alt.Chart(corrMatrix_fileavg, title = ["Feature Correlation Heatmap of PCA Data Files: "
                    ,'PCA Confidence Interval: '+ str(ci)]
                    ).mark_rect().encode(
    x='var2:O',
    y='var1:O'
    ,color='mean:Q'
    , tooltip=['mean:Q','var2_feature_names:O','var1_feature_names:O'],
    ).properties(height=800, width = 900).show()


predictions_avg = pd.DataFrame(index=train_y_df.index)
model_score_results = pd.DataFrame(columns = ['model','max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse'])
col_score_results =pd.DataFrame(columns = ['model','filename','column','avg_column_significance', 'column_descr'])

print('all pca vif filtered kfold split process: ')
for i, (train_index, test_index) in enumerate(kf10.split(train_X)):
        # print(f"Fold {i}:")
    code_names = pca_train_df.columns
    max_features = 'auto'
    max_depth = 15
    ccp_alpha = .75 #float when increasing from 0 to 1, tree is pruned more to better generalize

    percent_threshold = 0.0009

    # max_features = 'sqrt'
    max_depth = 25
    ccp_alpha = .9 #float when increasing from 0 to 1, tree is pruned more to better generalize

    # print([pca_train_df.columns.index(x) for x in vif['feature_code']])
    dt_regr = DecisionTreeRegressor(max_features= max_features, max_depth= max_depth, ccp_alpha = ccp_alpha).fit(pd.DataFrame(pca_train_df).iloc[train_index,[x for x in vif.index]], pd.DataFrame(train_y_df).iloc[train_index,:])
    dt_regr_score = dt_regr.score(pd.DataFrame(pca_train_df).iloc[test_index,[x for x in vif.index]], pd.DataFrame(train_y_df).iloc[test_index,:])
    dt_regr_predict = dt_regr.predict(pd.DataFrame(pca_test_df).iloc[:,[x for x in vif.index]])

    rmse = round(math.sqrt(mean_squared_error(train_y_df, dt_regr_predict)),3)

    dt_col_results = pd.DataFrame(columns = ['model','filename', 'max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse','column','avg_column_significance', 'column_descr'])
    # print(dt_regr.feature_importances_)
    print([code_names[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]])
    for sig_i, sig_column in enumerate([code_names[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]]):

        col_score = round([dt_regr.feature_importances_[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]][sig_i],4)
        col_df = pd.DataFrame([[dt_regr.__class__.__name__, sig_column.split('_')[0]
                                , max_features, max_depth, ccp_alpha, dt_regr_score, rmse ,sig_column, col_score
                                ,meta_dict[sig_column.split('_')[0]]
                               ]]
                               ,columns = ['model','filename', 'max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse','column','avg_column_significance', 'column_descr'])
        dt_col_results = pd.concat([dt_col_results, col_df], axis = 0)

    model_score_results = pd.concat([model_score_results,dt_col_results[['model','max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse']].drop_duplicates()],axis=0)
    col_score_results = pd.concat([col_score_results,dt_col_results[['model','filename','column','avg_column_significance', 'column_descr']]],axis=0)

    model_predictions = pd.DataFrame(dt_regr_predict, index=test_y_df.index)
    predictions_avg = pd.concat([predictions_avg, model_predictions],axis=1)

model_score_results=model_score_results.groupby(['model','max_features','max_depth','ccp_alpha'],as_index = False).mean().reset_index()
col_score_results=col_score_results.groupby(['model','filename','column', 'column_descr'],as_index = False)['avg_column_significance'].aggregate(['mean']).reset_index().sort_values(by=['mean'], ascending=False)
print(model_score_results)
print(col_score_results)
print(col_score_results[['column','column_descr','mean']])
# print(col_score_results.columns)
print(nca_pca.get_params(deep=True))

print(col_score_results)
print(col_score_results.columns)
print(vif)
print(vif.columns)
col_score_results['vif'] = [vif[vif['feature_code']==x]['VIF'] for x in col_score_results['column']]
#for failure analysis:
prediction_eval = pd.DataFrame(test_y_df,columns=[train_year+2],index = test_y_df.index)
prediction_eval = pd.concat([prediction_eval,pd.DataFrame(predictions_avg.mean(axis=1),columns=['avg_prediction'])],axis=1)
prediction_eval = pd.concat([prediction_eval,pd.DataFrame(predictions_avg.std(axis=1),columns=['std_avg_prediction'])],axis=1).astype(float)
prediction_eval = prediction_eval.replace(0,np.nan)
prediction_eval['prediction_difference_percent'] = ((prediction_eval[train_year+2] - prediction_eval['avg_prediction'])/prediction_eval[train_year+2])*100
prediction_eval['prediction_difference_percent_abs'] = abs(prediction_eval['prediction_difference_percent'])
prediction_eval=prediction_eval.sort_values(by=['prediction_difference_percent_abs'],ascending=False)
prediction_eval['GDP Per Capita'] = prediction_eval[train_year+2]
prediction_eval['Country Name'] = prediction_eval.index
prediction_eval.drop(columns=[train_year+2],inplace=True)
# prediction_eval = prediction_eval[prediction_eval['prediction_difference_percent_abs']<100]

forecast_failure_chart = alt.Chart(prediction_eval.iloc[:30, :]
                                   , title = ["Countries with the Greatest Forecast Discrepancies:"
                    ,'Forecast Year: ' + str(train_year +2) 
                    ,'Model: '+model_score_results['model'][0]+ ' Score: '+str(round(model_score_results['avg_r_score'][0],4))
                    +' RMSE: '+str(round(model_score_results['avg_rmse'][0],4))]
                    
                    ).mark_geoshape(stroke='black' 
                                    ,fillOpacity=1,strokeWidth=.05
                                   ).mark_bar().encode( 
    x = alt.X('prediction_difference_percent:Q',scale = alt.Scale(domain=[prediction_eval['prediction_difference_percent'][:30].min(),
                                                             prediction_eval['prediction_difference_percent'][:30].max()])),  
    y=alt.Y("Country Name:O", sort=alt.SortField("prediction_difference_percent_abs", order="descending"))
    , tooltip=['GDP Per Capita:Q', 'prediction_difference_percent:Q'],
    color=alt.Color('GDP Per Capita:Q', scale= alt.Scale(range=['darkred','darkblue']
                                                         ,domain=[prediction_eval['GDP Per Capita'][:30].min(),#
                                                             prediction_eval['GDP Per Capita'][:30].max()]#
                                                          ))
    ).properties(height=800, width = 900)

forecast_failure_chart.show()

def dt_model_kfold_pca_vif(max_features, max_depth, ccp_alpha ,pca_train_df, train_y_df, pca_test_df, test_y_df, splits):
    param_tune_df = pd.DataFrame(columns = ['max_features','max_depth','alpha','avg_r_score','r_std','avg_rmse_score','rmse_std'])
    kf10 = KFold(n_splits = splits)
    kf10.get_n_splits(pca_train_df)

    k10_fold_r_results = pd.DataFrame(columns = ['r_score','rmse_score'])

    for i, (train_index, test_index) in enumerate(kf10.split(pca_train_df)):
        # print(f"Fold {i}:")
        dt_regr = DecisionTreeRegressor(max_features= max_features, max_depth= max_depth, ccp_alpha = ccp_alpha).fit(pd.DataFrame(pca_train_df).iloc[train_index,[x for x in vif.index]], pd.DataFrame(train_y_df).iloc[train_index,:])
        dt_regr_score = dt_regr.score(pd.DataFrame(pca_train_df).iloc[test_index,[x for x in vif.index]], pd.DataFrame(train_y_df).iloc[test_index,:])
        dt_regr_predict = dt_regr.predict(pd.DataFrame(pca_test_df).iloc[:,[x for x in vif.index]])

        rmse = round(math.sqrt(mean_squared_error(test_y_df, dt_regr_predict)),3)

        k10_fold_r_results = pd.concat([k10_fold_r_results,pd.DataFrame([[dt_regr_score,rmse]],columns = ['r_score','rmse_score'])],axis=0)

        r_mean = k10_fold_r_results['r_score'].mean(axis=0)
        r_std = k10_fold_r_results['r_score'].std(axis=0)
        rmse_mean = k10_fold_r_results['rmse_score'].mean(axis=0)
        rmse_std=k10_fold_r_results['rmse_score'].std(axis=0)
        param_tune_df = pd.concat([param_tune_df, pd.DataFrame([[max_features, max_depth,ccp_alpha,r_mean ,r_std ,rmse_mean ,rmse_std]],columns = ['max_features','max_depth','alpha','avg_r_score','r_std','avg_rmse_score','rmse_std'])],axis=0)
    param_tune_df=param_tune_df.groupby(['max_features','max_depth','alpha'],as_index = False).mean().reset_index()
    return param_tune_df

#parameter tuning
all_max_features = ['log2','auto','sqrt']
max_depths = [5, 10, 15, 20, 25, 30]
alpha_values = [.1, .25, .5, .75, .9]

param_tune_df = pd.DataFrame(columns = ['max_features','max_depth','alpha','avg_r_score','avg_rmse_score'])

for max_feat in all_max_features:
    for max_dep in max_depths:
        for alph in alpha_values: 
            iter_param_tune_df = dt_model_kfold_pca_vif(max_feat, max_dep, alph ,pca_train_df, train_y_df, pca_test_df, test_y_df, 10)
            param_tune_df = pd.concat([param_tune_df,iter_param_tune_df[['max_features','max_depth','alpha','avg_r_score','avg_rmse_score']]],axis=0)
param_tune_df = param_tune_df.sort_values(by=['avg_r_score'], ascending=False)
# param_tune_df.to_csv('/Users/stephenontko/Documents/University of Michigan/UMSI/UMSI MADS/UMSI MADS 2023/SIADS 699 Capstone/Data/IMF/alL_data_sets_pca_decision_tree_param_tuning.csv',index=True)
# param_tune_df = pd.read_csv('/Users/stephenontko/Documents/University of Michigan/UMSI/UMSI MADS/UMSI MADS 2023/SIADS 699 Capstone/Data/IMF/alL_data_sets_pca_decision_tree_param_tuning.csv')
print(param_tune_df.head(20))

#choropleth map chart of predictions
def choropleth_dfs(test_y, train_year, predictions):
    choropleth_df = test_y.copy().astype(float)
    choropleth_df['Country Name'] = choropleth_df.index
    choropleth_df['GDP Per Capita'] = [x for x in predictions]
    choropleth_df['id'] = [iso_country_codes[x] for x in choropleth_df.index]

    choropleth_actual_df = choropleth_df[['Country Name',train_year+2,'id']]
    choropleth_actual_df.rename(columns={train_year+2:'GDP Per Capita'},inplace=True)
    choropleth_actual_df = choropleth_actual_df[['Country Name','GDP Per Capita','id']]

    choropleth_df = choropleth_df[['Country Name','GDP Per Capita','id']]
    return choropleth_df, choropleth_actual_df

def chart_compilation(choropleth_df, choropleth_actual_df, train_year, model, model_score,rmse):
    gdpactual= alt.Chart(
        world
          , title = ["Actual GDP Per Capita"
                    ,'for Year: ' + str(train_year +2)]).mark_geoshape(stroke='black' #'white'
                                                                ,fillOpacity=1,strokeWidth=.05).transform_lookup(
    lookup='id',
    from_=alt.LookupData(data=choropleth_actual_df, key = 'id', fields=['id','Country Name','GDP Per Capita'])
    ).encode(
    # tooltip='properties.geounit:N',
    color=alt.Color('GDP Per Capita:Q'
                    ,scale= alt.Scale(domain=[
                                                # 0,100000
                                            choropleth_actual_df['GDP Per Capita'].min()
                                              ,choropleth_actual_df['GDP Per Capita'].max()
                                              ]
                                        # ,range=['red','yellow','green']
                                        ,range=['lightgreen','darkblue']
                                      ))
    , tooltip=['Country Name:N', 'GDP Per Capita:Q']

    ).project(
    type='mercator'
    ).properties(
    width=800,
    height=800
    )
    gdpactual#.show()

    gdpforecast= alt.Chart(
        world
          , title = ["GDP Per Capita Forecasting : "
                    ,'Forecast Year: ' + str(train_year +2) 
                    ,'Model: '+model.__class__.__name__+ ' Score: '+str(round(model_score,4))
                    +' RMSE: '+str(rmse)]
                    ).mark_geoshape(stroke='black' #'white'
                                    ,fillOpacity=1,strokeWidth=.05).transform_lookup(
    lookup='id',
    from_=alt.LookupData(data=choropleth_df, key = 'id', fields=['id','Country Name','GDP Per Capita'])
    ).encode(
    # tooltip='properties.geounit:N',
    color=alt.Color('GDP Per Capita:Q'
                    ,scale= alt.Scale(domain=[
                                           # 0
                                        choropleth_df['GDP Per Capita'].min()
                                              ,choropleth_df['GDP Per Capita'].max()]
                                        # ,range=['red','yellow','green']
                                        ,range=['lightgreen','darkblue']
                                      ))
    , tooltip=['Country Name:N', 'GDP Per Capita:Q']

    ).project(
    type='mercator'
    ).properties(
    width=800,
    height=800
    )
    gdpforecast#.show()
    return alt.hconcat(gdpactual, gdpforecast).show()

choropleth_df, choropleth_actual_df = choropleth_dfs(test_y_df, train_year, predictions_avg.mean(axis=1))
print(model_score_results.columns)
print(model_score_results['model'][0])
print(model_score_results['avg_r_score'][0])
print(model_score_results['avg_rmse'][0])
chart_compilation(choropleth_df, choropleth_actual_df, train_year, model_score_results['model'][0]
                  , model_score_results['avg_r_score'][0], model_score_results['avg_rmse'][0])

#learning rate
learning_rates = [25, 50, 75, 100, 125, 150, 175, train_y_df.shape[0]]

learning_rate_df = pd.DataFrame(columns = ['max_features','max_depth','alpha','avg_r_score','avg_rmse_score','samples'])

for learning_rate in learning_rates:
    print(learning_rate)
    random_sample_indices = random.sample(range(0,learning_rate),learning_rate)
    #sqrt, 15, .75
    max_features = 'auto'
    max_depth = 25
    ccp_alpha = .9
    iter_learning_rate_df = dt_model_kfold_pca_vif(max_features, max_depth, ccp_alpha ,pca_train_df.iloc[random_sample_indices,:], train_y_df.iloc[random_sample_indices,:], pca_test_df.iloc[random_sample_indices,:], test_y_df.iloc[random_sample_indices,:], 10)
    iter_learning_rate_df['samples'] = learning_rate
    learning_rate_df = pd.concat([learning_rate_df,iter_learning_rate_df[['max_features','max_depth','alpha','avg_r_score','avg_rmse_score','samples']]],axis=0)
print(learning_rate_df)
# learning_rate_df.to_csv('/Users/stephenontko/Documents/University of Michigan/UMSI/UMSI MADS/UMSI MADS 2023/SIADS 699 Capstone/Data/IMF/alL_data_sets_pca_decision_tree_learning_rate.csv',index=True)
# param_tune_df = pd.read_csv('/Users/stephenontko/Documents/University of Michigan/UMSI/UMSI MADS/UMSI MADS 2023/SIADS 699 Capstone/Data/IMF/alL_data_sets_pca_decision_tree_param_tuning.csv')
# print(param_tune_df)

# iter pca
# iter_pca_model_score_results = pd.DataFrame(columns = ['model','max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse'])
iter_pca_col_score_results =pd.DataFrame(columns = ['model','filename','iteration','column', 'column_descr','avg_column_significance'])

for iter_pca in range(1,5):
    print(iter_pca)
    # model_score_results = pd.DataFrame(columns = ['model','max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse'])
    pca_col_model_results =pd.DataFrame(columns = ['model','filename','iteration','column','avg_column_significance', 'column_descr'])

    print('all pca vif filtered kfold split process: ')
    for i, (train_index, test_index) in enumerate(kf10.split(train_X)):
        # print(f"Fold {i}:")
        code_names = pca_train_df.columns
        max_features = 'auto'
        max_depth = 15
        ccp_alpha = .75 #float when increasing from 0 to 1, tree is pruned more to better generalize

        percent_threshold = 0.0009

    # max_features = 'sqrt'
        max_depth = 25
        ccp_alpha = .9 #float when increasing from 0 to 1, tree is pruned more to better generalize

    # print([pca_train_df.columns.index(x) for x in vif['feature_code']])
        dt_regr = DecisionTreeRegressor(max_features= max_features, max_depth= max_depth, ccp_alpha = ccp_alpha).fit(pd.DataFrame(pca_train_df).iloc[train_index,[x for x in vif.index]], pd.DataFrame(train_y_df).iloc[train_index,:])
        dt_regr_score = dt_regr.score(pd.DataFrame(pca_train_df).iloc[test_index,[x for x in vif.index]], pd.DataFrame(train_y_df).iloc[test_index,:])
        dt_regr_predict = dt_regr.predict(pd.DataFrame(pca_test_df).iloc[:,[x for x in vif.index]])

        rmse = round(math.sqrt(mean_squared_error(train_y_df, dt_regr_predict)),3)

        pca_col_score_results = pd.DataFrame(columns = ['model','filename','iteration','avg_r_score','avg_rmse','column','avg_column_significance', 'column_descr'])
    # print(dt_regr.feature_importances_)
        # print([code_names[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]])
        for sig_i, sig_column in enumerate([code_names[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]]):

            col_score = round([dt_regr.feature_importances_[x] for x in [list(dt_regr.feature_importances_).index(x) for x in dt_regr.feature_importances_ if x > percent_threshold]][sig_i],4)
            col_df = pd.DataFrame([[dt_regr.__class__.__name__, sig_column.split('_')[0],iter_pca
                                , dt_regr_score, rmse ,sig_column, col_score
                                ,meta_dict[sig_column.split('_')[0]]
                               ]]
                               ,columns = ['model','filename','iteration','avg_r_score','avg_rmse','column','avg_column_significance', 'column_descr'])
            # print(col_df)
            pca_col_score_results = pd.concat([pca_col_score_results, col_df], axis = 0)
        # print(pca_col_score_results)
        # model_score_results = pd.concat([model_score_results,dt_col_results[['model','max_features','max_depth','ccp_alpha','avg_r_score','avg_rmse']].drop_duplicates()],axis=0)
        pca_col_model_results = pd.concat([pca_col_model_results,pca_col_score_results[['model','filename','iteration','column','avg_column_significance', 'column_descr']]],axis=0)
        
        # print(iter_pca_col_score_results)

    print(pca_col_model_results)
    # model_score_results=model_score_results.groupby(['model','max_features','max_depth','ccp_alpha'],as_index = False).mean().reset_index()
    iter_pca_col_score_results_gp = pca_col_model_results.groupby(['model','filename','iteration','column', 'column_descr'],as_index = False)['avg_column_significance'].aggregate(['mean']).reset_index().sort_values(by=['mean'], ascending=False)
    print(iter_pca_col_score_results_gp)
    iter_pca_col_score_results = pd.concat([iter_pca_col_score_results,iter_pca_col_score_results_gp[['model','filename','iteration','column', 'column_descr','mean']]],axis=0)

print(iter_pca_col_score_results)
print(col_score_results.columns)
print(nca_pca.get_params(deep=True))

print(col_score_results.columns)
print(vif_all)
iter_pca_col_score_results['vif'] = [vif_all[vif_all['feature_code']==x]['VIF'].values[0] for x in iter_pca_col_score_results['column']]

print(iter_pca_col_score_results)
print(iter_pca_col_score_results[['iteration','column','column_descr','mean','vif']])

# iter_pca_col_score_results = iter_pca_col_score_results.rename(columns={'mean':'mean_score'})
across_pca_iter_avg = iter_pca_col_score_results.groupby(['column'],as_index = False)['mean'].aggregate(['mean']).reset_index().sort_values(by=['mean'], ascending=False)
print(across_pca_iter_avg)
iter_pca_col_score_results['overall_mean'] = [across_pca_iter_avg[across_pca_iter_avg['column']==x]['mean'].values[0] for x in iter_pca_col_score_results['column']]

column_chart_dict = {}

for iter_chart in range(1,5):
    print(iter_chart)
    iter_df = iter_pca_col_score_results[iter_pca_col_score_results['iteration'] == iter_chart].sort_values(by=['overall_mean'], ascending=False)#.iloc[:10,:]
    iter_df_more_agg = iter_df.groupby(['model','filename','iteration','column','column_descr'],as_index = False).mean().reset_index()
    print(iter_df_more_agg)
    print(iter_df)
    # iter_df['count'] = iter_df['column']
    # iter_df['count'] = [dt_col_results.groupby(['column']).size().loc[x] for x in iter_df['column']]
    print(iter_df.columns)
    iter_df = iter_df.sort_values(by=['mean'])
    column_chart_dict[iter_chart] = alt.Chart(iter_df.iloc[:5,:]).mark_bar().encode(
    x = alt.X('mean:Q'
    # ,scale = alt.Scale(domain=[iter_pca_col_score_results['mean'].min(),
    #                                                             iter_pca_col_score_results['mean'].max()])
                    , sort=alt.EncodingSortField(field="overall_mean", op='values', order='descending')
                    ),  #op="count", 
    y=alt.Y("column:O" )
    , tooltip='column_descr:O',

    color=alt.Color('vif:Q', scale= alt.Scale(range=['lightgreen','darkblue'] ))
    ).properties(height=150, width = 700)

    print(iter_df)
    print(iter_df.columns)

alt.vconcat(column_chart_dict[1], column_chart_dict[2], column_chart_dict[3]
            , column_chart_dict[4]).show()