#feature generation script for contracts
#Emily Grace and Elissa Redmiles
import pandas as pd
import argparse
import datetime as dt
import numpy as np
from matplotlib import pyplot as plt

import currency

#arguments section
# inputs are:
# -f name of file to clean
# -p name of the column containing "procurement method"
# -wf name of the file to output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file_name',help='Input file')
    parser.add_argument('-p', '--proc_col_name', help='name of the column that contains procurement method', default='proc_meth')
    parser.add_argument('-wf','--output_file',default="",help='full path and file name without extension where file should be written')
    parser.add_argument('-ppp','--ppp_file',default="/mnt/data/world-bank/ppp.csv", help='path to currency conversion files')
     parser.add_argument('-ppp','--fcrf_file',default="/mnt/data/world-bank/fcrf.csv", help='path to currency conversion files')
    args = parser.parse_args()

    input_file = args.file_name

    if '.csv' in input_file:
        print "starting processing"
        df=pd.read_csv(input_file, low_memory=False)
        print("processing " + str(input_file) + "type: csv")
    if '.xls' in input_file:
        print "starting processing 2"
        df = pd.read_excel(input_file, low_memory=False)
 
    #drop first column (unnamed indexing column)
    df.drop(df.columns[0],axis=1,inplace=True)

    if 'amount_standardized' not in df.columns:
       df = standardize_amount(df)


    #collapse rows that are identical except in amount 
    cols_for_grouping = [col for col in df.columns if col not in ['amount','amount_standardized']]
    df= pd.DataFrame(df.groupby(cols_for_grouping)[['amount','amount_standardized']].sum())
    df.reset_index(inplace=True)

    #add column for relative value of contract
    if 'project_proportion' not in df.columns:
        df = proportion_of_project(df)

    #add columns for value of contract relative to  mean and total by year
    if 'contract_high_rel__mean_by_year_flag' not in df.columns:
        print type(df)
        df = rel__per_year(df)
        print type(df)
    
    #add binary columns for whether procurement method is objective, competitive
    df['objective'] = df[args.proc_col_name].map(is_objective)
    df['competitive'] = df[args.proc_col_name].map(is_objective)

    #write to output file
    if args.output_file is not "":
        df.to_csv(args.output_file) 


def proportion_of_project(data):

    """Add data features related to the total value of the project"""

    project_amounts = data.groupby('project_id')['amount_standardized'].sum().reset_index()

    project_amounts.rename(columns={'amount_standardized' : 'project_total_amount'},inplace=True)

    data = data.merge(project_amounts,on=('project_id'))

    data['project_proportion'] = data['amount_standardized'] / data['project_total_amount']
    
    return data

def rel__per_year(data):
    _mean = data.groupby(['country', 'fiscal_year'])['amount_standardized'].mean().reset_index()
    _mean.rename(columns={'amount_standardized' : 'country_mean_amount'},inplace=True)
    data = data.merge(_mean,on=(['country', 'fiscal_year']))  
    data['contract_high_rel__mean_by_year_flag'] = data['amount_standardized'] > data['country_mean_amount']
    print data['contract_high_rel__mean_by_year_flag'].describe()
    _total = data.groupby(['country','fiscal_year'])['amount_standardized'].sum().reset_index()
    _total.rename(columns={'amount_standardized' : 'country_total_amount'},inplace=True)
    data = data.merge(_total,on=(['country', 'fiscal_year']))
    data['contract_proportion__total_by_year'] = data['amount_standardized'] / data['country_total_amount']
    print data['contract_proportion__total_by_year'].describe()
    print type(data)
    return data
def is_objective(x):
    if x in ['International Competitive Bidding',
             'National Competitive Bidding',
             "Limited International Bidding",
             "SHOP",
             "Least Cost Selection"
             ]:
        return True
    elif x in [
               'Quality And Cost-Based Selection',
               "Quality Based Selection",
               'CQS',
               "Selection Under a Fixed Budget"
               ]:
        return False
    else:
        return None

def is_competitive(x):
    if x in ['International Competitive Bidding',
             'National Competitive Bidding',
             "Limited International Bidding",
             "SHOP",
             "Least Cost Selection"
             ]:
        return True
    elif x in [             
               'Quality And Cost-Based Selection',
               "Quality Based Selection",
               'CQS',
               "Selection Under a Fixed Budget"
               ]:
        return False
    else:
        return None

#Rescale contract $ amount to account for inflation
def standardize_amount(data):
    ppp_data = pd.read_csv(str(args.ppp_file),skiprows=2)
    currency_convert_data = pd.read_csv(str(args.fcrf_file),skiprows=0,low_memory=False)
    print currency_convert_data.columns
    ppp_data = currency.clean_world_bank_data(ppp_data,'ppp')
    currency_convert_data = currency.clean_world_bank_data(currency_convert_data,'currency_convert')
#    print data.columns
    data = data[data['contract_signing_date'].notnull()]
 #   print data.columns
    data['contract_signing_date']=pd.to_datetime(data['contract_signing_date'],errors='raise')
    data['contract_signing_date'] = data['contract_signing_date'].astype(dt.datetime)
    print data['contract_signing_date']
    data['year'] = data['contract_signing_date'].map(lambda x: x.year)
    data['month'] = data['contract_signing_date'].map(lambda x: x.month)
 
    data = currency.clean.fix_country_names(data)

    data = data.merge(ppp_data,left_on=['fiscal_year','country'],right_on=['year','Country Name'],how='left')
    data = data.merge(currency_convert_data,left_on=['fiscal_year','country'],right_on=['year','Country Name'],how='left')
    data['ppp'].fillna(0,inplace=True)
    data['currency_convert'].fillna(0,inplace=True)
    
    # amount ($) * currency_convert (local / $ ) * ppp ($ / local) 
    data['amount_standardized'] = data['amount'] * data['currency_convert'] #/ data['ppp']

    data['amount_standardized'][data['currency_convert'] == 0] = data['amount'][data['currency_convert'] == 0]

    return data

if __name__ == '__main__':
    main()

#compute difference between sign and award dates
#print df.columns
#new_col =(pd.to_datetime(df['ctr_sign_date']) - pd.to_datetime(df['ctr_awd_date']))
#df.insert(16,'date_diff', new_col.astype('timedelta64[D]'))

#some statistics about times per 
#by = new_df.groupby('country')
#print by['date_diff'].describe()
#bycontracttype = new_df.groupby('proc_meth')
#print bycontracttype['date_diff'].describe()
#byregion = new_df.groupby('region');
#print byregion['date_diff'].describe()

