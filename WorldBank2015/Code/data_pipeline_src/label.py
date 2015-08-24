import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c','--contracts_data',help='Contracts input file')
parser.add_argument('-i', '--investigations_data', help='Investigations Input File')
parser.add_argument('-wf','--output_file',default="",help='full path and file name where file should be written')
args = parser.parse_args()
if '.csv' in args.contracts_data:
  
    contracts=pd.read_csv(args.contracts_data, low_memory=False)
if '.xls' in args.contracts_data:
    contracts = pd.read_excel(args.contracts_data, low_memory=False)

if '.csv' in args.investigations_data:
    labels=pd.read_csv(args.investigations_data, low_memory=False)
if '.xls' in args.investigations_data:
    labels = pd.read_excel(args.investigations_data, low_memory=False)
print 'contracts shape: '
print contracts.shape
#print contracts['wb_contract_number']
print 'labels shape: '
print labels.shape
#print labels.columns
print 'Contracts: ',len(contracts.index)
print 'Labels: ', len(labels.index)
#print 'Num unique contracts: ', contracts.shape
#print 'Num unique labels: ', labels.shape
labels = labels.drop_duplicates()
#print 'Labels: ', len(labels.index)
labels = labels[(labels['allegation_outcome'] == 'Substantiated') |
                          (labels['allegation_outcome'] == 'Unfounded') | 
                         (labels['allegation_outcome'] == 'Unsubstantiated') |
			(pd.isnull(labels['allegation_outcome']))]
print 'Num with labels: ', len(labels.index)
labels=labels[labels['wb_id'].apply(lambda x: str(x).isdigit())]
#labels['wb_id']=labels['wb_id'].filter(lambda x: x.isdigit())
#print labels['wb_id']
labels['wb_id']=labels['wb_id'].map(lambda x: np.int_(x))#x.astype(int))
#print labels['wb_id']
labels = labels[(pd.notnull(labels['allegation_category']))]
labels=labels[['wb_id','allegation_category','allegation_outcome']]
#print labels.columns
equiv = {'Unfounded':0, 'Unsubstantiated':1, 'Substantiated':2}
labels["outcome_val"] = labels['allegation_outcome'].map(equiv.get)
just_subst  = labels[(labels['outcome_val'] == 2)]
just_unsubst  = labels[(labels['outcome_val'] == 1)]
just_unfound  = labels[(labels['outcome_val'] == 0)]
#print "percent substantiated: ", str(float(len(just_subst.index))/float(len(labels.index)))
#print "percent unsubstantiated: ", str(float(len(just_unsubst.index))/float(len(labels.index)))
#print "percent unfounded ", str(float(len(just_unfound.index))/float(len(labels.index)))
labels= pd.DataFrame(labels.groupby(['wb_id','allegation_category'])[['outcome_val']].max())
#print labels
labels.reset_index(inplace=True)    
labels = labels.drop_duplicates()
equiv = {0:'Unfounded',1:'Unsubstantiated',2:'Substantiated'}
labels['allegation_outcome']=labels['outcome_val'].map(equiv.get)
#print 'Labels: ', len(labels.index)
just_subst  = labels[(labels['outcome_val'] == 2)]
just_unsubst  = labels[(labels['outcome_val'] == 1)]
just_unfound  = labels[(labels['outcome_val'] == 0)]
#print "percent substantiated: ", str(float(len(just_subst.index))/float(len(labels.index)))
#print "percent unsubstantiated: ", str(float(len(just_unsubst.index))/float(len(labels.index)))
#print "percent unfounded ", str(float(len(just_unfound.index))/float(len(labels.index)))
contracts['wb_contract_number']=contracts['wb_contract_number'].map(lambda x: np.int_(x))
#print contracts['wb_contract_number']
cols_to_use = labels.columns - contracts.columns

merged = contracts.merge(labels[cols_to_use], left_on=['wb_contract_number'], right_on=['wb_id'])
print 'num unique wbid labels: '+str(labels['wb_id'].nunique())
print 'num unique contracts: '+str(contracts['wb_contract_number'].nunique())
print 'labels shape: '
print labels.shape
print 'contracts shape: '
print contracts.shape
print 'merged shape: '
print merged.shape
#print merged['wb_id'].nunique()
#print merged['supplier'].nunique()
#merged=merged.drop_duplicates()
#print cols_to_use
#print 'Merged: ', len(merged.index)
#print 'Merged columns: ', merged.columns
#print 'Merged shape: ', merged.shape
just_subst  = merged[(merged['outcome_val'] == 2)]
just_unsubst  = merged[(merged['outcome_val'] == 1)]
just_unfound  = merged[(merged['outcome_val'] == 0)]
#print "percent substantiated: ", str(float(len(just_subst.index))/float(len(merged.index)))
#print "percent unsubstantiated: ", str(float(len(just_unsubst.index))/float(len(merged.index)))
#print "percent unfounded ", str(float(len(just_unfound.index))/float(len(merged.index)))
#subset = merged[(pd.notnull(merged['allegation_outcome']))]
#print subset['wb_id'].nunique()
#print subset['supplier'].nunique()
#print subset['supplier']
#subset= subset[['supplier','allegation_outcome','allegation_category']]
#subset_w_dummies=pd.get_dummies(subset,columns=['allegation_outcome'])
#subset_w_dummies=subset_w_dummies.groupby(['supplier'])
#print subset_w_dummies.head()
#count=subset_w_dummies.count()
#subset_w_dummies=subset_w_dummies.sum()
#subset_w_dummies=subset_w_dummies
#subset_w_dummies=subset_w_dummies.apply(lambda x: 100.0*x/x.sum(), axis=1)
#subset_w_dummies=pd.get_dummies(subset_w_dummies['allegation_outcome'])
#print subset_w_dummies
#subset.to_csv('/mnt/data/world-bank/pipeline_data/labels.csv')
merged.to_csv(args.output_file)
