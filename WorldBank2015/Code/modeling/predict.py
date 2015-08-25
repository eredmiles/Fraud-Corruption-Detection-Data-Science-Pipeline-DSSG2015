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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import yaml
#from treeinterpreter import treeinterpreter as ti

import model_pipeline_script


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-tf','--training_table',help='Contract data file')
parser.add_argument('-pf','--prediction_table',help='File for prediction',default='')
parser.add_argument('-ac','--allegation_category',help='allegation category for prediction (contracts)',default='')
parser.add_argument('-fl','--feature_sets_log_file',
                    help='Feature Set Log File, typically feature_sets_log.yaml',
                    default='feature_sets_log.yaml')
parser.add_argument('-wf','--output_file',help='File to write output (ranked list')
parser.add_argument('-pred_id','--predict_table_id',help='Identifier for reading feature tables',default='')
parser.add_argument('-train_id','--train_table_id',help='Identifier for reading feature tables')
args = parser.parse_args()


def main():

    clf = GradientBoostingClassifier(n_estimators = 1000, min_samples_split=15, learning_rate = 0.1, max_depth=160)
    feature_set_id = '59'

    feature_sets_file = args.feature_sets_log_file

    feature_set_dict = {}
    with open(feature_sets_file, 'r') as stream:
        feature_set_dict = yaml.load(stream)

    feature_set = feature_set_dict[feature_set_id]

    engine = model_pipeline_script.get_engine()
    con = engine.connect()
    
    if args.prediction_table != '':
        contract_flag = True
    else:
        contract_flag = False
    
    contracts_data = pd.read_sql(args.training_table,engine)
    if contract_flag:
        prediction_data = pd.read_sql(args.prediction_table,engine)

    print contracts_data.columns
    #proccess training data
    contracts_data['amt_standardized'] = contracts_data['amount_standardized']
    contracts_data['contract_signing_date'] = pd.to_datetime(contracts_data['contract_signing_date'])
    #Subsetting on only main allegation outcomes
    train_data = contracts_data[(contracts_data['allegation_outcome'] == 'Substantiated') |
                                (contracts_data['allegation_outcome'] == 'Unfounded') | 
                                (contracts_data['allegation_outcome'] == 'Unsubstantiated')]
    
    train_data,col_group_dict_train = model_pipeline_script.join_features(engine,con,contracts_data,args.train_table_id)
    col_group_dict_train,col_group_keys_train = model_pipeline_script.define_feature_sets(col_group_dict_train)

    if contract_flag:
    #process prediction data
        prediction_data['amt_standardized'] = prediction_data['amount_standardized']
        prediction_data['contract_signing_date'] = pd.to_datetime(prediction_data['contract_signing_date'])
        prediction_data['allegation_category']=args.allegation_category

        prediction_data,col_group_dict_predict = model_pipeline_script.join_features(engine,con,prediction_data,args.predict_table_id)
        col_group_dict_predict,col_group_keys_predict = model_pipeline_script.define_feature_sets(col_group_dict_predict)


    train_df = train_data[ train_data['allegation_outcome'].notnull() ]
    if not contract_flag:
        predict_df = train_data[ train_data['allegation_outcome'].isnull() ]

        predict_df.drop('allegation_outcome',1,inplace=True)
    else:
        predict_df = prediction_data


    feature_set_new = []
    for feat_set in feature_set:
        if 'cntrcts_splr_ftr_set_train' in feat_set:
            feat_set = feat_set.replace('cntrcts_splr_ftr_set_train','cntrcts_splr_ftr_set_' + args.train_table_id)
        feature_set_new.append(feat_set)
    feature_set = feature_set_new

    df_features_train,y_train = model_pipeline_script.select_features(train_df,col_group_dict_train,feature_set)

    print 'feat_sets:'
    if args.predict_table_id != '':
        feature_set_new = []
        for feat_set in feature_set:
            print feat_set
            if 'cntrcts_splr_ftr_set_' + args.train_table_id in feat_set:
                feat_set = feat_set.replace('cntrcts_splr_ftr_set_' + args.train_table_id,'cntrcts_splr_ftr_set_' + args.predict_table_id)
            feature_set_new.append(feat_set)
        feature_set = feature_set_new

    print 'shape: '
    print predict_df.shape,feature_set
    if contract_flag:
        df_features_predict,y_predict = model_pipeline_script.select_features(predict_df,col_group_dict_predict,feature_set)
    else:
        df_features_predict,y_predict = model_pipeline_script.select_features(predict_df,col_group_dict_train,feature_set)
    print df_features_predict.shape

    df_to_write = df_features_train.merge(pd.DataFrame(y_train),left_index=True,right_index=True)
    df_to_write.to_csv('features_and_outcomes.csv')

    matching_cols = [val for val in df_features_train.columns if val in set(df_features_predict.columns)]
    print len(matching_cols),len(df_features_train.columns),len(df_features_predict.columns)

    df_features_train = df_features_train[matching_cols]
    df_features_predict = df_features_predict[matching_cols]


    x_train = np.array(df_features_train)
    y_train = np.array(y_train)
    x_train = x_train.astype(float)

    x_predict = np.array(df_features_predict)
    x_predict = x_predict.astype(float)


    print 'Fitting....'
    clf.fit(x_train,y_train)

    print 'Predicting...'
    y_pred = clf.predict(x_predict)
    y_proba= clf.predict_proba(x_predict).T[1]
   
#code for printing out top features
    #try:



#    print 'Feature importance...'
#        print df_features_train.columns,df_features_train.shape
     #   top_features = model_pipeline_script.get_feature_importance(clf,x_train,y_train,df_features_train.columns,nfeatures=50)
      #  print top_features
        #feat_idx = []
        #for feat in top_features:
        #    print feat
        #    idx = 
        
#        model_pipeline_script.decision_surface_plot(clf,df_features_train,y_train,top_features)

   # except IOError:
    #    ''

#code for plotting distribution of prediction scores
   # plt.hist(y_proba,bins=30)
   # if contract_flag:
   #     plt.title('Prediction Scores on Contracts')
   # else:
   #     plt.title('Prediction Scores on Uninvestigated Complaints')
   # plt.xlabel('Prediction Score')
   # if contract_flag:
   #     plt.ylabel('Number of Contracts')
   # else:
   #     plt.ylabel('Number of Complaints')
   # plt.show()

    prediction_data = predict_df
    prediction_data['prediction_score'] = y_proba

    grouped = prediction_data[['country','prediction_score']].groupby('country').aggregate(['mean','median','std','count'])

    grouped.columns = [' '.join(col).strip() for col in grouped.columns.values]
#    print prediction_data.columns


#    prediction_data[['country','prediction_score']].to_sql('prediction_scores_complaints_by_country_nocountryfeatures',engine,if_exists='replace')
    if contract_flag:
        output_df=prediction_data[['wb_contract_number','fiscal_year','region','country','project_id','project_name','contract_description','supplier','borrower_contract_reference_number','amount','prediction_score']]
    else:
	output_df=prediction_data[['wb_contract_number','fiscal_year','region','country','project_id','project_name','contract_description','supplier','borrower_contract_reference_number','amount','allegation_category','prediction_score']]

    if '.csv' not in args.output_file:
        output_file = args.output_file + '.csv'
        output_table = args.output_file
    else:
        output_file = args.output_file
        output_table = re.sub(r'\.csv$', '', args.output_file)
        output_table_array = output_table.split("/")
	print output_table_array
	output_table = output_table_array[len(output_table_array)-1]
    output_df.to_csv(output_file,encoding='utf-8') 

    if len(output_table) > 63:
        output_table = output_table[:63]

    output_df.to_sql(output_table,engine,if_exists='replace')
        
#    grouped.to_sql('contract_set_w_prediction_nocountries',engine,if_exists='replace')


if __name__ == "__main__":
    main()
