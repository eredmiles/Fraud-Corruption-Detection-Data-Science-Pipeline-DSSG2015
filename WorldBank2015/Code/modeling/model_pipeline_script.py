import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib as matplotlib
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import pandas as pd
import numpy as np
import ConfigParser
import sklearn
from sqlalchemy import create_engine

from sklearn import cross_validation
from sklearn import ensemble
from sklearn import metrics
import seaborn as sns
import re
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve,precision_score,recall_score, roc_auc_score,roc_curve
from sklearn.preprocessing import normalize

from itertools import product,combinations,cycle
from dateutil import rrule
from datetime import datetime,timedelta
import os
from scipy.interpolate import interp1d

import time
import re


def plot_precision_recall_n(y_true, y_prob, model_name):
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
  
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    #plt.show()

#def prec_top_n(y_pred,y_proba,y_true):

    

def get_engine():

    config = ConfigParser.RawConfigParser()
    config.read('config')
    password = config.get('SQL','password')

    engine = create_engine(r'postgresql://dssg:' + password + '@localhost/world_bank')


    return engine

def plot_confusion(classifier,threshold =0.4):
    x_train,x_test,y_train,y_test = train_test_split(df_new,y,test_size = 0.2)
    y_pred = []
    try:
        prob_score = clf_grid.predict_proba(x_train)
    except:
	prob_score = clf_grid.predict_proba(np.float_(x_train))
    a = prob_score[:,1]
    for idx,item in enumerate(a):
        if item>= threshold:
            item = 1
        else:
            item =0
        y_pred.append(item)
    # Plotting                                                                                                              

    class_name = classifier.__repr__()
    class_name = re.sub(r'\([^)]*\)','',class_name)
    print ("")
    print ("")
    print("Legends")
    print ('1 - Substantiated')
    print ('0 - Unfounded')
    print("")
    print("Confusion Matrix: "+ class_name+ " (threshold- " +str(threshold)+")"  )
    sns.heatmap(metrics.confusion_matrix(y_pred, y_train), annot=True, cmap="YlGnBu",fmt ="d")
    plt.xlabel('Predicted')
    plt.ylabel('True')

def dummy_from_categorical(data):

    #make dummy variables from categoricals
    categorical = data.select_dtypes(include=[object])

    for col in categorical.columns:

        if  categorical[col].nunique() > 2:
            dummy_features = pd.get_dummies(categorical[col]) 
            dummy_features.columns = ['is_' + '_'.join([col] + c.split()).lower() for c in dummy_features.columns]

            data.drop(col,axis=1,inplace=True)
            data = data.merge(dummy_features,left_index=True,right_index=True)
    
    return data


def replace_nans(data):

    #fill NaNs/drop columns with all null and handle NaNs
    data.replace([np.inf, -np.inf], np.nan,inplace=True);
    for col in data.columns:
        null_count = data[col].isnull().sum()
        try:
            percent_of_null = float(100*((null_count)/len(data[col])))
        except:
            null_count = null_count.values[0]
            percent_of_null = float(100*((null_count)/len(data[col])))
            
            
        if percent_of_null == 100.0:
            data.drop(col, axis=1, inplace=True)
            print ('dropping',col)
        elif null_count >0:
            data[col+'_is_null'] = data[col].isnull()
            data[col].fillna(-99999.99,inplace=True)
    
    return data


def heat_map(clf,score_name='Precision'):

    """Make heat map of grid search scores"""
    
    #read grid scores into array
    param_list = clf.best_params_.keys()
    
    param_value_ar = []
    num_value_list = []
    for param in param_list:
        param_values = []
        for item in clf.grid_scores_:
            p = item.parameters[param]
            param_values.append(p)
        unique_values = np.array(list(set(param_values)))
        unique_values.sort()
        param_value_ar.append(unique_values)
        num_values = len(set(param_values))
        num_value_list.append(num_values)
    
    print(param_value_ar)
    
    grid_params = np.zeros((num_value_list[0],num_value_list[1]))
    
    
    for item in clf.grid_scores_:
        score = item.mean_validation_score
    
        p1 = item.parameters[param_list[0]]
        p2 = item.parameters[param_list[1]]
    
        idx1 = np.where(param_value_ar[0] == p1)
        idx2 = np.where(param_value_ar[1] == p2)
    
        grid_params[idx1,idx2] = score
    
    print(grid_params)
        
    #plot heat map
    with sns.axes_style("white"):
        im = plt.imshow(grid_params,interpolation='nearest',origin='lower')
        im.set_cmap('Reds')
    plt.yticks(range(num_value_list[0]), param_value_ar[0])
    plt.xticks(range(num_value_list[1]), param_value_ar[1])
    plt.ylabel(param_list[0])
    plt.xlabel(param_list[1])
    cbar = plt.colorbar(im)
    cbar.set_label('Mean ' + score_name)
    #plt.show()




def GridSearch(classifier,x_train,x_test,y_train,y_test, param={'n_estimators':[50,100],'max_depth':[2,5],
                                 'learning_rate':[0.1,0.5],'min_samples_split':[2,4]}, score = 'precision'):
      
    # Setting the parameters for CV:
    valid_params = {}
    valid_keys = classifier.get_params()
    for key,value in zip(param.keys(),param.values()):
        if key in valid_keys.keys():
            valid_params[key] = value

    
    print("No. of Tuning hyper-parameters for %s" %score)
    clf_grid = GridSearchCV(classifier,valid_params,cv=5, scoring= score)
    clf_grid.fit(x_train,y_train)
    print("Best parameters set found on development set:")
    print(clf_grid.best_params_)
    print()
    print("Detailed classification report:")
    print()
    y_true, y_pred = y_test, clf_grid.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
    
    return clf_grid


#plot confusion matrix
def plot_confusionMatrix(classifier,x_train,x_test,y_train,y_test):
    classifier = clf_grid.best_estimator_
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print ("")
    print ("")
    print("Legends")
    print ('1 - Substantiated')
    print ('0 - Unfounded')
    print("")
    print(" ---------------- Confusion Matrix:-------------")
    sns.heatmap(metrics.confusion_matrix(y_pred, y_test), 
                annot=True, cmap="YlGnBu",fmt ="d")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    print("The mean accuacy on validation set for the model: ",
          (100*clf.score(x_test,y_test)),"%")


# Getting the feature importance using the above optimal parameters:
def get_feature_importance(classifier,x_train,y_train,columns,nfeatures=10):
    
    i = x_train.shape[-1]
    a = classifier.__repr__()
    classifier_name = re.sub(r'\([^)]*\)', '', a)
    
    clf = classifier.fit(x_train,y_train)
    importance_features = classifier.feature_importances_
   
#    std_feature = np.std([tree.feature_importances_ for tree in clf.estimators_],
#                         axis = 0)
    index = np.argsort(importance_features)[::-1]
    #print importance_features.shape
    print ("Ranking of top ten features:")
    top_features = []
    for item in range(nfeatures):
        top_features.append(index[item])
        print("%d feature %s (%f, %i)" % (item +1, columns[index[item]], importance_features[index[item]],index[item]))
    index = index[:nfeatures]    
    plt.title("Feature Importance Score - " + classifier_name + " - Top Ten")
    plt.ylabel('% Variance Explained')
    plt.bar(range(nfeatures),100.0*importance_features[index],
            color = 'green', 
            align = 'center')
    plt.xticks(range(nfeatures), columns[index],fontsize=14)
    plt.xlim([-1,nfeatures])
#    plt.show()
    plt.clf()
    plt.cla()

    return top_features

def feature_direction(idx,dataframe,label,threshold):
    y_pred = [];
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=80,
                               min_samples_split=5)
    x_trainr,x_test,y_train,y_test = train_test_split(dataframe,label,test_size = 0.2)
    col_names = list(dataframe.columns.values)
    maximum_val = x_train[:,idx].max()
    minimum_val = x_train[:,idx].min()
    feature_name = col_names[idx]
    for i,col in enumerate(x_train):
        if i != idx:
            x_train[:,i] = np.mean(x_train[:,i])
    
    clf_feature = clf_rf.fit(x_train,y_train)
    proba_score_feature=clf_feature.predict_proba(x_train)    
    score = proba_score_feature[:,1]
    for item in score:
        if item >=threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    plt.scatter(x_train[:,idx],y_pred)
    plt.xlabel(feature_name)
    plt.ylabel('prediction')
    plt.suptitle('Response Curve-  ' + feature_name)

    return top_features



def threshold_to_count(thresholds,y_pred_probs):

    counts = []
    for thresh in thresholds:
#        print thresh
        num_investigated = (y_pred_probs >= thresh).astype(int).sum()
#        print num_investigated
        counts.append(num_investigated)
    percents = 100.0*np.array(counts) / float(len(y_pred_probs))
#    print thresholds,counts,percents

    return percents

def precision_n_percent(y_test,y_pred_probs,n):

#    print 'PRECISION'

    sort_idx = y_pred_probs.argsort()

#    print y_pred_probs[sort_idx],y_test[sort_idx]

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
#    print 'start',thresholds,precision,recall

    precision = precision[:-1]
    recall = recall[:-1]


#    if 0. not in thresholds:
#        thresholds = np.concatenate((np.array([0]),thresholds))
#    if 1. not in thresholds:
#        thresholds = np.concatenate((thresholds,np.array([1])))
    counts = threshold_to_count(thresholds,y_pred_probs)

#    print 'end',thresholds,precision,recall

#    plt.plot(thresholds,precision,color='r')
#    plt.plot(thresholds,recall,color='b')
#    plt.show()

    if n > min(counts) and n < max(counts):
        precision_n = interp1d(counts, precision)(np.array([n]))[0]
        recall_n = interp1d(counts, recall)(np.array([n]))[0]
    elif n <= min(counts):
        precision_n = precision[-1]
        recall_n = recall[-1]
    elif n >= max(counts):
        precision_n = precision[0]
        recall_n = recall[0]


    return precision_n,recall_n

def precision_recall(classifier,x_train,x_test,y_train,y_test,ax,color):
    try:
        y_pred_probs = classifier.predict_proba(x_test)
    except:
        y_pred_probs = classifier.predict_proba(np.float_(x_test))
    y_pred_probs = y_pred_probs.T[1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs)
    precision = precision[:-1]
    recall = recall[:-1]

#    print thresholds,thresholds.size,precision.size
#    thresholds = np.concatenate((np.array([0]),thresholds))
#    print thresholds

    counts = threshold_to_count(thresholds,y_pred_probs)

    ax.plot(counts,recall,color='b')
    ax2 = ax.twinx()
    ax2.plot(counts,precision,color=color)
    
    for tl in ax.get_yticklabels():
        tl.set_color('b')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ax.set_xlabel('% of Complaints Investigated')
    ax.set_ylabel('recall',color='b')
    ax2.set_ylabel('precision',color='r')
    #ax.set(aspect=100)
    #ax2.set(aspect=100)
    ax2.set_ylim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlim(0,100)

def get_valid_params(clf,param_sets):

    """Find valid parameters in classifier"""

    valid_params = {}
    valid_keys = clf.get_params()
    print valid_keys
    for key,value in zip(param_sets.keys(),param_sets.values()):
        if key in valid_keys.keys():
            valid_params[key] = value

    return valid_params

def plot_roc_curve(y_test,y_proba,ax):

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba,pos_label=1)
                
    counts = threshold_to_count(thresholds,y_proba)

    ax.plot([0,1],[0,1],linestyle='--',color='k')
    sc = ax.scatter(fpr,tpr,c=counts,lw = 0,cmap=plt.cm.Blues)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Postive Rate')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    #ax.set(aspect=1)
    cbar = plt.colorbar(sc)
    cbar.set_label('% Complaints Investigated')
    #plt.show()
    #fig = plt.gca()


def select_features(labelled_data,col_group_dict,col_set):


        col_selection_temp = []
        for cset in col_set:
            if cset in col_group_dict.keys():
                col_selection_temp += col_group_dict[cset]


        col_selection = []
        for col in col_selection_temp:
            if col in labelled_data.columns:
                col_selection.append(col)


        df_new = labelled_data[col_selection]
        df_new = dummy_from_categorical(df_new)
        df_new = replace_nans(df_new)

        try:
            #select labels
            label = labelled_data['allegation_outcome']

            #create numerical labels
            y = label.copy()
            y.replace('Substantiated',1,inplace=True)
            y.replace('Unsubstantiated',0,inplace=True)
            y.replace('Unfounded',0,inplace=True)
        except KeyError:
            y = np.zeros(len(df_new.index))

        for col,dt in zip(df_new.columns,df_new.dtypes):
            if dt not in ['int64','float64','bool']:
                df_new.drop(col,axis=1,inplace=True)

        return df_new,y

def write_model_summary(clf_name,fig_name,train,test,params = {},features = {},scores = {}):

    """Output model configuration and performance to a markdown file"""

    text_output = '/mnt/data/world-bank/egrace/models/' + fig_name + '.md'

    if not os.path.isfile(text_output): 
        output = open(text_output,'w')
        output.write('#' + clf_name + '\n')
        output.write('##Params: ' + '\n')
        for param,value in zip(params.keys(),params.values()):
            output.write('\t' + str(param) + ': ' + str(value) + '\n')
        output.write('##Features: ' + '\n')
        for feature in features:
            output.write('\t' + feature + '\n')
    else:
        output = open(text_output,'a')
    output.write('#####Train through ' + str(train.strftime('%d %B %Y')))
    output.write(' <br/> Test from ' + str(train.strftime('%d %B %Y')) + ' to ' + str(test.strftime('%d %B %Y')) + '\n')
    for score_name,score in zip(scores.keys(),scores.values()):
        output.write('\t' + str(score_name) + ' = ' + str(score) + '\n')
    output.write('![](https://github.com/dssg/WorldBank2015/blob/master/Code/figures/' + fig_name + '_split' + train.strftime('%Y%m%d') + '.png)\n')
    output.close()


def get_feature_sets(feature_sets_file = 'feature_sets_log.yaml',col_group_keys=[],n_list=[],sets = []):

#    engine = get_engine()
#    con = engine.connect()

    feature_sets_file = 'feature_sets_log.yaml'
    import yaml

    feature_set_dict = {}
    with open(feature_sets_file, 'r') as stream:
        feature_set_dict = yaml.load(stream)

    if feature_set_dict.keys() == [0]:
        feature_set_dict = {}
        num_feature_sets = 0
    else:
        num_feature_sets = max(np.array(feature_set_dict.keys()).astype(int))
 
    feature_combos = []
    feature_combo_ids = []
    set_list = []
    if len(n_list) != 0:
        for n in n_list:
            for comb in combinations(col_group_keys,n):
                set_list.append(list(comb))
    elif len(sets) != 0:
        for comb in sets:
            set_list.append(list(comb))
    for comb in set_list:
        feature_combos.append(list(comb))
        if list(comb) not in feature_set_dict.values():
            num_feature_sets += 1
            feature_set_dict[str(num_feature_sets)] = list(comb)
            feature_combo_ids.append(num_feature_sets)
        else:
            for key, value in feature_set_dict.items():
                if value == list(comb):
                    feature_combo_ids.append(int(key))
 

    stream = file(feature_sets_file, 'w')
    yaml.dump(feature_set_dict, stream,default_flow_style=False)

    print len(feature_combos)
    return feature_combos,feature_combo_ids

def list_db_tables(con):

     result = con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name;")
     result = list(result.fetchall())
     tables = [r[0] for r in result]

     return tables


def feature_direction(clf,x_train,df,feature_idx,threshold=0.5):

    

    col_names = list(df.columns)
    maximum_val = x_train[:,feature_idx].max()
    minimum_val = x_train[:,feature_idx].min()
    feature_name = col_names[feature_idx]

    value_range = np.arange(100)/99.0 * (maximum_val - minimum_val) + minimum_val

    feature_means = []
    for i,col in enumerate(x_train.T):
        feature_ar = x_train[:,i]
        feature_means.append(np.median(feature_ar))
        
    x_test_ar = np.zeros((len(value_range),len(feature_means)))

    for i in range(len(value_range)):
        x_test_ar[i] = feature_means
        x_test_ar[i,feature_idx] = value_range[i]

    proba_score_feature=clf.predict_proba(x_test_ar)
    score = proba_score_feature[:,1]
    plt.plot(value_range,score)
    plt.xlabel(feature_name)
    plt.ylabel('prediction')
    plt.suptitle('Response Curve-  ' + feature_name)
    #plt.show()

def join_features(engine,con,data,table_criteria):

    tables = list_db_tables(con)

    col_group_dict = {}
    for table in tables:
        if 'cntrcts_splr_ftr_set' in table and table_criteria in table:
            print table
            table_df = pd.read_sql(table,engine)
            table_df['contract_signing_date'] = pd.to_datetime(table_df['contract_signing_date'])
            col_group_dict[table] = table_df.columns - ['supplier', 'wb_contract_number',
                                                         'contract_signing_date','amt_standardized',
                                                         'index','unique_id', 'supplier_reformat',
                                                         'ctry','sect','rgn','prc_ctg','prc_typ']
            #print data['unique_id'].unique(),data['unique_id'].nunique(),table_df['unique_id'].unique(),table_df['unique_id'].nunique()
            print 'New data: ',table_df.shape
            if data['unique_id'].nunique() == table_df['unique_id'].nunique():
                data = data.merge(table_df,on=['unique_id'],
                                  how='left')
            print data.shape

    return data,col_group_dict

def define_feature_sets(col_group_dict):
    col_group_dict['contract'] = ['major_sector',
                                  'procurement_category',
                                  'procurement_type',
                                  'project_proportion',
                                  'amount_standardized'
                                  'procument_method',
                                  'objective',
                                  'competitive',
                                  'region']
    col_group_dict['contract_country'] = ['country',  
                                          'supplier_country']
    col_group_dict['project'] = ['project_total_amount']
    
    col_group_dict['allegation'] = ['allegation_category']
    
    col_group_dict['supplier'] = ['supplier_reformat']
    
    col_group_keys = col_group_dict.keys()
        
    return col_group_dict,col_group_keys

def plot_decision_regions(X, y, clf, X_highlight=None, res=0.01, cycle_marker=True, legend=1, cmap=None):

    # check if data is numpy array                                                                                                                                           
    for a in (X, y):
        if not isinstance(a, np.ndarray):
            raise ValueError('%s must be a NumPy array.' % a.__name__)

    # check if test data is provided                                                                                                                                         
    plot_testdata = True
    if not isinstance(X_highlight, np.ndarray):
        if X_highlight is not None:
            raise ValueError('X_test must be a NumPy array or None')
        else:
            print 'I am setting to False!'
#            plot_testdata = False

    if len(X.shape) == 2 and X.shape[1] > 1:
        dim = '2d'
    else:
        dim = '1d'


    marker_gen = cycle('sxo^v')

    # make color map                                                                                                                                                         
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    n_classes = len(np.unique(y))
    if n_classes > 5:
        raise NotImplementedError('Does not support more than 5 classes.')

    if not cmap:
        cmap = matplotlib.colors.ListedColormap(colors[:n_classes])

    # plot the decision surface                                                                                                                                              

    if dim == '2d':
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    else:
        y_min, y_max = -1, 1

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    print x_min,x_max,res,res*(x_max-x_min)
    print y_min,y_max,res, res*(y_max-y_min)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, res*(x_max-x_min)),
                         np.arange(y_min, y_max, res*(y_max-y_min)))

    if dim == '2d':
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)

    else:
        y_min, y_max = -1, 1

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    if dim == '2d':
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    else:
        y_min, y_max = -1, 1
        Z = clf.predict(np.array([xx.ravel()]).T)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples                                                                                                                                                     

    for c in np.unique(y):
        if dim == '2d':
            y_data = X[y==c, 1]
        else:
            y_data = [0 for i in X[y==c]]

            plt.scatter(x=X[y==c, 0],
                    y=y_data,
                    alpha=0.8,
                    c=cmap(c),
                        marker=next(marker_gen),
                    label=c)

    if legend:
        plt.legend(loc=legend, fancybox=True, framealpha=0.5)

    print X
    if plot_testdata:
        if dim == '2d':
            plt.scatter(X[:,0], X[:,1], c='', alpha=1.0, linewidth=1, marker='o', s=80)
        else:
            plt.scatter(X, [0 for i in X], c='', alpha=1.0, linewidth=1, marker='o', s=80)

def decision_surface_plot(model,df_train,y_train,list_of_top_features):
    print list_of_top_features
    for item in combinations(list_of_top_features,2):

        a = list(item)
        print a                                                                                                                                                             
        X = df_train[a].values
        #replace_nan_inf(X)                                                                                                                                                  
        #replace_nan_inf(df_new)                                                                                                                                             
        xx = X[:,[0,1]]
        target = y_train#.as_matrix()
        #model = classifier
        model.fit(xx,target)
        plot_decision_regions(xx,target,model)
        plt.suptitle('Random Forest Decision Surface')
        plt.xlabel(df_train.columns[a[0]])
        plt.ylabel(df_train.columns[a[1]])
        plt.show()

def main():

    engine = get_engine()
    con = engine.connect()

#    contracts_data = pd.read_csv('/mnt/data/world-bank/joinedcontracts_features_phase4_supplier_features_labelled.csv');
#    contracts_data = pd.read_csv('/mnt/data/world-bank/data_for_dssg/labeled_contracts_cleaned_resolved_feature_gen_1.csv')
    contracts_data = pd.read_sql('labeled_contracts_cleaned_resolved_feature_gen_1',engine)

    contracts_data['amt_standardized'] = contracts_data['amount_standardized']
    contracts_data['contract_signing_date'] = pd.to_datetime(contracts_data['contract_signing_date'])

    #Subsetting on only main allegation outcomes
    labelled_data = contracts_data[(contracts_data['allegation_outcome'] == 'Substantiated') |
                                   (contracts_data['allegation_outcome'] == 'Unfounded') | 
                                   (contracts_data['allegation_outcome'] == 'Unsubstantiated')]


    labelled_data,col_group_dict = join_features(engine,con,labelled_data,'train')


    print labelled_data.shape
    labelled_data = labelled_data[ labelled_data['year'] > 2006 ]
    print labelled_data.shape

    labelled_data['contract_signing_date'] =  pd.to_datetime(labelled_data['contract_signing_date'])

    col_group_dict,col_group_keys = define_feature_sets(col_group_dict)
        

    models = {
        'logistic_regression':
            {'model':LogisticRegression(),
             'params':{'C':[0.1,0.5,1.0]}}, 
            'random_forest':
            {'model':RandomForestClassifier(),
             'params':{'n_estimators':[500,1000],
                       'max_depth':[40,80,160,500,1000],
                       'min_samples_split':[2,5,10],
                       'probability':[True]}},
            'ada_boost':
            {'model':AdaBoostClassifier(),
             'params':{'n_estimators':[500,1000],
                       'learning_rate':[0.1,0.5,0.75,1.0]}},
            'svc':
            {'model':SVC(),
             'params':{'C':[0.1,0.5,1.0],
                       'kernel':['linear','rbf'],
                       'probability':[True]}},
##            'decision_tree':
#            {'model':DecisionTreeClassifier(),
#             'params':{'n_estimators':[100,500,1000],
#                       'max_depth':[10,40,80,160,320],
 #                      'min_samples_split':[2,5,10],
 #                      'probability':[True]}},
            'gradient_boosting':
            {'model':GradientBoostingClassifier(),
             'params':{'n_estimators':[500,1000],
                       'max_depth':[40,80,160,500],
                       'min_samples_split':[2,5,10,15],
                       'learning_rate':[0.1,0.5,1.0],
                       'probability':[True]}},
            'kneighbors':
            {'model':KNeighborsClassifier(),
             'params':{'n_neighbors':[3,5,7,11,13,15,17,19]}},
#             'naive_bayes':
#            {'model':GaussianNB(),
#             'params':{}},
#            'SGD':
#            {'model':SGDClassifier(),
#             'params':{'n_estimators':[100,500,1000],
#                       'max_depth':[10,40,80,160,320],
#                       'min_samples_split':[2,5,10],
#                       'probability':[True],
#                       'loss':['log']}}
    }


    #empty dictionaries to store results
    score_dict = {}
    time_dict = {}

    feature_sets = [col_group_keys]

    feature_sets.append([x for x in col_group_keys if x not in ['contract','project','contract_country']])
    #    feature_sets.append(col_group_keys.remove('contract').remove('project').remove('contract_country'))
    

    base_set = ['contract','project','allegation','contract_country']
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation']
    feature_sets.append(base_set)


    base_set = ['contract','project','allegation','contract_country','supplier']
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation','supplier']
    feature_sets.append(base_set)


    base_set = ['contract','project','allegation']
    for key in col_group_keys:
        if 'pct_ct_dist' in key or 'pct_amt_dist' in key or 'sect' in key or 'prc_ctg' in key or 'prc_typ' in key:
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation']
    for key in col_group_keys:
        if 'pct_ct_dist' in key or 'pct_amt_dist' in key:
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation','contract_country']
    for key in col_group_keys:
        if '_amt_' in key:
            base_set.append(key)
    feature_sets.append(base_set)


    base_set = ['contract','project','allegation','contract_country']
    for key in col_group_keys:
        if '_ct_' in key:
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation','contract_country']
    for key in col_group_keys:
        if 'full' in key:
            base_set.append(key)
    feature_sets.append(base_set)


    base_set = ['contract','project','allegation','contract_country']
    for key in col_group_keys:
        if '1years' in key:
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation','contract_country']
    for key in col_group_keys:
        if '3years' in key:
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation','contract_country']
    for key in col_group_keys:
        if '5years' in key:
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation']
    for key in col_group_keys:
        if '_amt_' in key and ('pct_ct_dist' in key or 'pct_amt_dist' in key or 'sect' in key or 'prc_ctg' in key or 'prc_typ' in key):
            base_set.append(key)
    feature_sets.append(base_set)


    base_set = ['contract','project','allegation']
    for key in col_group_keys:
        if '_ct_' in key and ('pct_ct_dist' in key or 'pct_amt_dist' in key or 'sect' in key or 'prc_ctg' in key or 'prc_typ' in key):
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation']
    for key in col_group_keys:
        if 'full' in key and ('pct_ct_dist' in key or 'pct_amt_dist' in key or 'sect' in key or 'prc_ctg' in key or 'prc_typ' in key):
            base_set.append(key)
    feature_sets.append(base_set)


    base_set = ['contract','project','allegation']
    for key in col_group_keys:
        if '1years' in key and ('pct_ct_dist' in key or 'pct_amt_dist' in key or 'sect' in key or 'prc_ctg' in key or 'prc_typ' in key):
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation']
    for key in col_group_keys:
        if '3years' in key and ('pct_ct_dist' in key or 'pct_amt_dist' in key or 'sect' in key or 'prc_ctg' in key or 'prc_typ' in key):
            base_set.append(key)
    feature_sets.append(base_set)

    base_set = ['contract','project','allegation']
    for key in col_group_keys:
        if '5years' in key and ('pct_ct_dist' in key or 'pct_amt_dist' in key or 'sect' in key or 'prc_ctg' in key or 'prc_typ' in key):
            base_set.append(key)
    feature_sets.append(base_set)

    #select feature combinations to loop over
#    feature_combos,feature_combo_ids = get_feature_sets(col_group_keys= col_group_keys, n_list = [len(col_group_keys)])
    feature_combos,feature_combo_ids = get_feature_sets(col_group_keys= col_group_keys, sets = feature_sets)
#    feature_combos = [['contract','project']]
#    feature_combo_ids = [100]

    #test-train splits
    date_splits = rrule.rrule(rrule.MONTHLY,dtstart=datetime(2010,7,1),until=datetime(2014,1,1),interval=6)

    for date in date_splits:
        print date
        print date.strftime('%Y-%m-%d')


    for split in date_splits:

        print split, split + timedelta(days=365)
        #select indices of rows within appropriate time windows
        train_idx = np.where(labelled_data['contract_signing_date'] < split) 
        test_idx = np.where( (labelled_data['contract_signing_date'] > split) & 
                             (labelled_data['contract_signing_date'] <= split + timedelta(days=365)) ) 

        train_percent = float(len(train_idx[0])) / float(len(labelled_data.index))
        train_count = float(len(train_idx[0]))
        test_count = float(len(test_idx[0]))

        train_end = split
        test_start = split
        test_end = split + timedelta(days=365)

        #empty lists for storing models, parameters, features sets and scores
        all_models = []
        all_features = []
        all_params = []
        all_scores = []

        #loop over feature sets
        for feat_idx,feature_combo in zip(feature_combo_ids,feature_combos):

            df_features,y = select_features(labelled_data,col_group_dict,feature_combo) 

            #select train and test sets from feature subset using train/test indices
            df_train = df_features.iloc[train_idx[0],:]
            df_test = df_features.iloc[test_idx[0],:]
            y_train = y.iloc[train_idx[0]]
            y_test = y.iloc[test_idx[0]]


            x_train = np.array(df_train)
            x_test = np.array(df_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)


            #convert all features to float (needed for logistic, etc.)
            #normalize all features
            x_train_init = x_train.astype(float)
            x_test_init = x_test.astype(float)
 
            #loop over model types
            for model in models:    

                if model in ['logistic_regression','svc','kneighbors']:
                    x_train = normalize(x_train_init.astype(float),axis=0)
                    x_test = normalize(x_test_init.astype(float),axis=0)
                else:
                    x_train = x_train_init.astype(float)
                    x_test = x_test_init.astype(float)
                    

                print model,feat_idx

                #retrieve model and desired parameters ranges from model dictionary
                clf=models[model]['model']
                param_set = models[model]['params']

                #check if desired parameters are valid for this model
                params = get_valid_params(clf,param_set)
                param_names = params.keys()
                
                values = []
                for pn in param_names:
                    values.append(np.array(params[pn]))
            
                #loop over each set of possible parameters
                for item in  product(*params.values()):

                    #store parameter values in dictionary
                    param_dict = {}
                    for n,name in enumerate(param_names):
                        param_dict[name] = item[n]


                    try:
                        models_already_done = pd.read_sql('model_results3',engine)

                        #                    models_already_done.drop_duplicates(subset = ['classifier','params','feature_set','train_end'],inplace=True)
                        
                        #                    models_already_done.to_sql('model_results2',engine,if_exists='replace')
                        


                
                        clf_name = clf.__str__()
                        clf_name = re.sub(r'\([^)]*\)', '', clf_name)
                        check_model = models_already_done[ (models_already_done['classifier'] == clf_name) &
                                                           (models_already_done['params'] == str(param_dict)) &
                                                           (models_already_done['feature_set'] == str(feat_idx)) &
                                                           (models_already_done['train_end'] == split) ]

                        check_model_len = len(check_model.index)

                    except:
                        check_model_len = 0

                    if check_model_len == 0:

                        #set parameters of model
                        clf.set_params(**param_dict)
                        print clf

                        start = time.time()
                        #fit on training data and predict on test data
                        clf.fit(x_train,y_train)
                        y_pred = clf.predict(x_test)
                        time_elapsed = time.time() - start
                    
                        #get probability scores
                        try:
                            y_proba= clf.predict_proba(x_test).T[1]
                        except:
                            try:
                                y_proba=clf.predict_proba(x_test.astype(float)).T[1]	
                            except:
                                print "using loga because exception for:"+model
                                try:
                                    y_proba= clf.predict_log_proba(x_test).T[1]
                                except:
                                    y_proba=clf.predict_log_proba(np.float_(x_test)).T[1]

                        #model evaluation
                        precision = precision_score(y_test,y_pred)
                        recall = recall_score(y_test,y_pred)
                        auc_score = roc_auc_score(y_test, y_proba)

                        if True:
                            try:
                                top_features = get_feature_importance(clf,x_train,y_train,df_features.columns,nfeatures=20)
                            except:
                                ''
                            #for feat_idx in top_features:
                            #    feature_direction(clf,x_test,df_features,feat_idx)


                        #set unique name of model + param + feature combo
                        clf_name = clf.__str__()
                        clf_name = re.sub(r'\([^)]*\)', '', clf_name)
                        clf_name_full = clf_name + ' ' + str(param_dict) + 'feature_set' + str(feat_idx)

                        #store evaluation metrics in dictionary
                        score_set = {'classifier':clf_name,'params':str(param_dict),'feature_set':str(feat_idx),
                                     'train_end':train_end,'test_start':test_start,
                                     'test_end':test_end,'precision':precision,'recall':recall,'AUC':auc_score}
                        print 'AUC: ',auc_score
                        print 'Precision: ',precision
                        print 'Recall: ',recall
                        for n in [5,10,15,20,25,30,35,40,45,50]:
                            top_n_precision, top_n_recall =  precision_n_percent(y_test,y_proba,n)
                            score_set['precision_' + str(n)] = top_n_precision
                            print 'Precision Top ' + str(n) + '%:',top_n_precision
                            score_set['recall_' + str(n)] = top_n_recall
                            print 'Recall Top ' + str(n) + '%:',top_n_recall

                        score_set['train_count'] = train_count
                        score_set['test_count'] = test_count

                        df = pd.DataFrame([score_set.values()], columns = score_set.keys())
                        df.to_sql('model_results3',engine,if_exists='append')

                        #use unique name as dictionary key for scores
                        if clf_name_full not in score_dict.keys():
                            score_dict[clf_name_full] = [score_set['precision_20']]
                        else:
                            score_dict[clf_name_full].append(score_set['precision_20'])

                        if clf_name_full not in time_dict.keys():
                            time_dict[clf_name_full] = [time_elapsed]
                        else:
                            time_dict[clf_name_full].append(time_elapsed)
                
                        print 'train percent: ',train_percent,train_count,test_count
                        color = plt.cm.Reds(train_percent)
                        #plot ROC curve and precision/recall curves
                        fig, ((ax1, ax2)) = plt.subplots(1, 2,figsize=(12,6))
                        roc_plot = plot_roc_curve(y_test,y_proba,ax2)
                        pr_plot = precision_recall(clf,x_train,x_test,y_train,y_test,ax1,color)

                        fig_name = 'model_evaluation_' + clf_name 
                        for param,value in zip(param_dict.keys(),param_dict.values()):
                            fig_name += '_' + param +  str(value)
                        fig_name += '_feature_set' + str(feat_idx)

                        #write markdown file summarizing model + results
                        #write_model_summary(clf_name,fig_name,split,split + timedelta(days=365),params = param_dict,features=feature_combo,scores = score_set)

                        fig_name +=  '_split' + split.strftime('%Y%m%d')
                        fig_name = '/mnt/data/world-bank/egrace/figures/' + fig_name + '.png'

                        print fig_name,auc_score
                        #plt.savefig(fig_name)
                        #plt.show()

                        plt.clf()
                        plt.cla()
                        
                        #fill lists of models/params/features for selecting
                        #top overall results
                        all_models.append(clf_name)
                        all_features.append(feat_idx)
                        all_params.append(str(param_dict))
                        all_scores.append(score_set['precision_20'])


        if len(all_scores) > 0:
            #sort the models by score
            all_scores, all_models,all_params = zip(*sorted(zip(all_scores, 
                                                                all_models,
                                                                all_params)))

            #list models ranked by score
            for n in range(len(all_models)):
                print all_models[n],all_params[n],all_features[n],all_scores[n]
            print '---------'



    #find average score of each model/param/feature combo
    #across all train/test splits
    mean_score = {}
    std_score = {}
    for key,value in zip(score_dict.keys(),score_dict.values()):
        mean_score[key] = np.mean(value)
        std_score[key] = np.std(value)

    print 'Feature set reference: '
    for fidx,feature_set in zip(feature_combo_ids,feature_combos):
        print fidx,feature_set

    #sort by mean score and output results
    print 'Average scores:'
    for w in sorted(mean_score, key=mean_score.get, reverse=True):
        print w, mean_score[w]

    print 'Classifier Fit Times:'
    for key,value in zip(time_dict.keys(), time_dict.values()):
        print key, np.mean(value)
 
if __name__ == "__main__":
    main()

