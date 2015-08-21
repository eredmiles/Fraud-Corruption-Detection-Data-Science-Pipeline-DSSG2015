import csv
from IPython.display import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn import cross_validation
from sklearn import ensemble
from sklearn import metrics
import seaborn as sns
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import sys
sys.stdout = open('rank_list.csv', 'w')
# generating features dataframe:

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_file',help='Contract data file')
args = parser.parse_args()

df_supplier = pd.read_csv(args.input_file);




#Subsetting on only main allegation outcomes
df_supplier_= df_supplier[(df_supplier['allegation_outcome'] == 'Substantiated') |
                           (df_supplier['allegation_outcome']== 'Unfounded') | 
                           (df_supplier['allegation_outcome']= 'Unsubstantiated')]


#remove duplicate columns from sql merging
cols_fixed = []
for col in df_supplier_.columns:
    pattern_y = re.compile('.*_y')
    pattern_x = re.compile('.*_x')
    if pattern_y.match(col):
        df_supplier_.drop(col,axis=1,inplace=True)
    elif pattern_x.match(col):
        cols_fixed.append(col[:-2])
    else:
        cols_fixed.append(col)

df_supplier_.columns = cols_fixed




#setup feature groups
col_group_names = ['supplier_major_sectors','supplier_major_sectors_anon',
                   'supplier_countries','supplier_countries_anon',
                   'supplier_regions','supplier_regions_anon',
                   'network']
col_groups = [ ['major_sector_cum_contracts_.*','major_sector_percent_contracts_.*','.*_sector_dominance'], 
               ['sector_dominance_\d+','sector_percent_\d+'],  
               ['country_cum_contracts_.*','country_percent_contracts_.*','.*_country_dominance'],
               ['country_dominance_\d+','country_percent_\d+'],
               ['region_cum_contracts_.*','region_percent_contracts_.*','.*_region_dominance'],
               ['region_dominance_\d+','region_percent_\d+'],
               ['.*centrality.*','.*giant_component.*','.*neighbor.*','.*betweeness.*','.*dist_invest.*']]

col_group_dict = {}

for i,col_group in enumerate(col_groups):

    col_list = []

    for regexp in col_group:
        pattern = re.compile(regexp)

        for col in df_supplier_.columns:
            if pattern.match(col) and col not in col_list:
                col_list.append(col)

    col_group_dict[col_group_names[i]] = col_list

col_group_dict['country_specific'] = ['business_disclosure_index',
                                      'firms_competing_against_informal_firms_perc',
                                      'payments_to_public_officials_perc',
                                      'do_not_report_all_sales_perc',
                                      'legal_rights_index',
                                      'time_to_enforce_contract',
                                      'bribes_to_tax_officials_perc',
                                      'property_rights_rule_governance_rating',
                                      'transparency_accountability_corruption_rating',
                                      'gdp_per_capita',
                                      'primary_school_graduation_perc',
                                      'gini_index',
                                      'unemployment_perc',
                                      'country_mean',
                                      'year_mean',
                                      'business_disclosure_index_mean',
                                      'firms_competing_against_informal_firms_perc_mean',
                                      'payments_to_public_officials_perc_mean',
                                      'do_not_report_all_sales_perc_mean',
                                      'legal_rights_index_mean',
                                      'time_to_enforce_contract_mean',
                                      'bribes_to_tax_officials_perc_mean',
                                      'property_rights_rule_governance_rating_mean',
                                      'transparency_accountability_corruption_rating_mean',
                                      'gdp_per_capita_mean',
                                      'primary_school_graduation_perc_mean',
                                      'gini_index_mean',
                                      'unemployment_perc_mean']


col_group_dict['contract'] = ['major_sector',
                              'proc_categ',
                              'proc_meth',
                              'domestic_pref_allwed',
                              'domestic_pref_affect',
                              'date_diff',
                              'price_escaltn_flag',
                              '#_supp_awd',
                              'project_proportion',
                              'amount_standardized']
col_group_dict['contract_country'] = ['region',
                                      'country',  
                                      'supp_ctry']
col_group_dict['project'] = ['project_total_amount']




#select feature groups
#col_set = ['contract','project','network','supplier_major_sectors','supplier_countries_anon','supplier_regions']
col_set = ['supplier_major_sectors','supplier_major_sectors_anon',
                   'supplier_countries','supplier_countries_anon',
                   'supplier_regions','supplier_regions_anon',
                   'network','contract','project']

col_selection = []
for cset in col_set:
    col_selection += col_group_dict[cset]
 
df_new = df_supplier_[col_selection]
#df_new.drop('region',axis=1,inplace=True)





#select labels
label = df_supplier_['outcome_of_overall_investigation_when_closed']
print ('labels data', label.shape)

y = label.copy()
y.replace('Substantiated',1,inplace=True)
y.replace('Unsubstantiated',0,inplace=True)
y.replace('Unfounded',0,inplace=True)


#make dummy variables from categoricals
categorical = df_new.select_dtypes(include=[object])

for col in categorical.columns:

    #print(categorical[col])
    #print (col)
    if  categorical[col].nunique() > 2:
        dummy_features = pd.get_dummies(categorical[col]) 
        dummy_features.columns = ['is_' + '_'.join(c.split()).lower() for c in dummy_features.columns]

        df_new.drop(col,axis=1,inplace=True)
        df_new = df_new.merge(dummy_features,left_index=True,right_index=True)


# 
#fill NaNs/drop columns with all null and handle NaNs
df_new.replace([np.inf, -np.inf], np.nan,inplace=True);
for col in df_new.columns:
    null_count = df_new[col].isnull().sum()
    percent_of_null = float(100*((null_count)/len(df_new[col])))
    if percent_of_null == 100.0:
        df_new.drop(col, axis=1, inplace=True)
        #print ('dropping',col)
    elif null_count >0:
        df_new[col+'_is_null'] = df_new[col].isnull()
       # df_new = df_new.merge(col+'isnull',left_index=True,right_index=True)
        df_new[col].fillna(-99999.99,inplace=True)


x_train,x_test,y_train,y_test = train_test_split(df_new,y,test_size = 0.2)
clf_rf = RandomForestClassifier(n_estimators=100,max_depth=80)
clf_rf.fit(x_train,y_train)
y_pred = []
prob_score = clf_rf.predict_proba(x_train)
a = prob_score[:,1]
for idx,item in enumerate(a):
    if item>= 0.55:
        item = 1
    else:
        item =0
    y_pred.append(item)

prob_score = [];
for idx,item in enumerate(x_test):
    a = clf_rf.predict_proba(item)
    prob_score.append([a[:,1], idx])
prob_score.sort()
b = prob_score[::-1]

b = np.array(b)
index = b.T[1]
column = ['wb_contract_number','supplier','major_sector_x']
for i in index:
    for item in column:
        print str(df_supplier.iloc[i][item]) + ',',
    print ""
