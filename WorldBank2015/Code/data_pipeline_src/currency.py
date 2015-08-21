import datetime
from matplotlib import pyplot as plt
import os
import pandas as pd
import argparse
import numpy as np


import ConfigParser
import data_cleaning_util as clean
import sql

   

def import_data(data_file,source="RATEINF/CPI_USA"):

    """If data file exists read it in, else import data from Quandl"""

    
    if not os.path.isfile(data_file):
        data = get_quandl_data(source)
        data.to_csv(data_file)
    else:
        print 'Reading data from file'
        data = pd.read_csv(data_file,index_col=0)
        data.index = pd.to_datetime(data.index)

    return data
    

def clean_world_bank_data(data,indicator):

#    data.drop('Unnamed: 59', axis=1, inplace=True)


    #reshape data
    data = pd.melt(data, id_vars=['Country Name','Country Code','Indicator Name','Indicator Code'],
                       var_name='year',value_name=indicator)    

    data = data[data['year'] != 'Unnamed: 59']

    data['year'] = data['year'].astype(float)

    data = data[data['year'] >= 1986]
    data[indicator] = data[indicator].fillna(0)
    
    data['Country Name'] = [name.replace('Rep.','Republic').replace('RB','Republic').replace('FYR','former').replace('Dem.','Democratic')[:15] for name in data['Country Name']]

    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','-table_name',help='Input file')
    parser.add_argument('-wf','--output_file',default="",help='full path and file name without extension where file should be written')
    args = parser.parse_args()

    #read purchasing parity data
    ppp_data = pd.read_csv('/mnt/data/world-bank/pa.nus.ppp_Indicator_en_csv_v2.csv',skiprows=2)
    currency_convert_data = pd.read_csv('/mnt/data/world-bank/pa.nus.fcrf_Indicator_en_csv_v2.csv',skiprows=0)

    print currency_convert_data

    ppp_data = clean_world_bank_data(ppp_data,'ppp')
    currency_convert_data = clean_world_bank_data(currency_convert_data,'currency_convert')

    #read in inflation data
    inflation_data_file = 'output/inflation_data_quandl_rateinf_cpi_usa.csv'
    inflation_data = import_data(inflation_data_file)

    inflation_data['year'] = [date.year for date in inflation_data.index.to_pydatetime()]
    inflation_data['month'] = [date.month for date in inflation_data.index.to_pydatetime()]
    inflation_data['day'] = [date.day for date in inflation_data.index.to_pydatetime()]
    
    #read in contract data
#    data = sql.read_sql('joinedcontracts_reordered_lower_only_with_ids')
   # data = sql.read_sql('joinedcontracts_features_phase2')
    data = sql.read_sql(args.table_name)
    data = data[data['contract_signing_date'].notnull()]

    data['contract_signing_date'] = data['contract_signing_date'].astype(datetime.datetime)
    
    data['year'] = data['contract_signing_date'].map(lambda x: x.year)
    data['month'] = data['contract_signing_date'].map(lambda x: x.month)
   
    data = clean.fix_country_names(data)

    data = data.merge(ppp_data,left_on=['fiscal_year','country'],right_on=['year','Country Name'],how='left')
    data = data.merge(currency_convert_data,left_on=['fiscal_year','country'],right_on=['year','Country Name'],how='left')

    data['ppp'].fillna(0,inplace=True)
    data['currency_convert'].fillna(0,inplace=True)
    
    # amount ($) * currency_convert (local / $ ) * ppp ($ / local) 
    data['amount_standardized'] = data['amount'] * data['currency_convert'] #/ data['ppp']

    data['amount_standardized'][data['currency_convert'] == 0] = data['amount'][data['currency_convert'] == 0]

    data.to_csv(args.output_file)
#    sql.write_sql(data,'joinedcontracts_features_phase3')

if __name__ == '__main__':
    main()
