import pandas as pd
from datetime import date, timedelta
import time
import numpy as np
import re
import psycopg2
import ConfigParser
import argparse
from sqlalchemy import create_engine

import random
import sql

parser = argparse.ArgumentParser()
parser.add_argument('-cf','--contract_file',help='Contract data file')
parser.add_argument('-if','--invest_file',help='Labelled data file')
parser.add_argument('-a','--amounts',action='store_true',default=False,help='Calculate aggregated amount features')
parser.add_argument('-dist','-dist',action='store_true',default=True,help='Calculate distribution features')
parser.add_argument('-dom','-dom',action='store_true',default=False,help='Calculate dominance features')
parser.add_argument('-y','--num_years',default=0,help='Time periods in years')
parser.add_argument('-cat','--categ',default=['major_sector'],nargs='*',help='Categoricals to use')
parser.add_argument('-id','--table_id',default=time.strftime("%Y%m%d"),help='ID for SQL tables')
parser.add_argument('-lim','--contract_num_lim',default=5000,help='Maximum number of rows to use')

args = parser.parse_args()

def connect():

    """Connect to database"""

    #read password from config file                                                                                                             
    config = ConfigParser.RawConfigParser()
    config.read('config')
    password = config.get('SQL','password')

    #open connection with database                                                                                                              
    config = ConfigParser.RawConfigParser()
    config.read('config')
    password = config.get('SQL','password')


    con = psycopg2.connect(host="localhost",user='dssg',password=password,dbname="world_bank")

    return con

def snake_case(name):

    """Clean entity name strings"""
    remove_list = ['llc','ltd','llc','ltd','co','corporation','srl','nv','limited','pvtltd']
    remove = '|'.join(remove_list)
    regex = re.compile(r'\b('+remove+r')\b', flags=re.IGNORECASE)

    try:
        
        s1 = name.lower()
        s1 = s1.replace('.','')
        s1 = regex.sub("", s1)
        s1 = s1.strip()

        s1 = re.sub(' +','_',s1)
        s1 = re.sub('-','_',s1)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s1)
        s1 = s1.replace('*','')
        s1 = s1.replace('(','')
        s1 = s1.replace(')','')
        s1 = s1.replace('"','')
        s1 = s1.replace(',','')
        s1 = s1.replace('#','')
        s1 = s1.replace(':','_')
        s1 = s1.replace('&','_')
        s1 = s1.replace('\'','')
        s1 = s1.replace('/','_')
        s1 = re.sub('_+','_',s1)
    except:
        s1 = ''
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def reformat(data,column,inplace=False,shorten=False):
  
    if inplace:
        data[column] = data[column].map(lambda x: snake_case(x))
    else:
        data[column + '_reformat'] = data[column].map(lambda x: snake_case(x))

    if shorten:
        data[column] = [re.sub(r'and', '', x).replace('__','_') for x in data[column]]
        data[column] = [re.sub(r'[aeiou]', '', x) for x in data[column]]


    return data

def binarize(data,fields):

    dummies = pd.get_dummies(data[fields]).astype('int64')
    dummies.columns = ['_'.join(('is',fields,col,'ct')) for col in dummies.columns]
    data = data.merge(dummies,left_index=True,right_index=True,how='left')
 
    return data
    
def conditional_amounts(data):

    for col in data.columns:
        if 'is' in col and 'total' not in col and 'cum' not in col and 'percent' not in col and 'dominance' not in col:
            data[re.sub('_ct$','',col) + '_amt'] = data[col]*data['amount_standardized']

    return data

def distribution(data,field,amount=False):

    cols_to_use = []
    for col in data.columns:
        if 'is' in col and 'cum' in col and field in col and 'total' not in col and 'percent' not in col and 'dominance' not in col:
            if amount and 'amt' in col:
                cols_to_use.append(col)
            elif not amount and not 'amt' in col:
                cols_to_use.append(col)

    subset = data[cols_to_use]

    dist = subset.apply(lambda x: 100.0*x/x.sum(), axis=1)

    dist.columns = [col + '_percent' for col in dist.columns]

    return dist

def count_previous_contracts(data,days=0,amount = True, count = False):

    """Count number of data entries in the past n days from each entry"""

    def sum_func(column):
        def inner_func(t):
        
            if days == 0:
                min_date_lim = 0
            else:
                min_date_lim = t - timedelta(days)
            total = data.ix[(min_date_lim < data['contract_signing_date']) & (data['contract_signing_date'] <= t),[column,'amount_standardized']]

            if amount:
                total_sum = ((total[column] != 0)*total['amount_standardized']).cumsum()
            else:
                total_sum = total[column].cumsum()

            return total_sum

        return inner_func

    data = data.sort('contract_signing_date')


    
    count = 0
    for col in data.columns:
        if 'is' in col and 'total' not in col and 'cum' not in col and 'full' not in col and 'year' not in col:
            func = sum_func(col)
            result_temp = data[['contract_signing_date']].apply(func)
            result_temp = pd.DataFrame(result_temp)
            result_temp.columns = [col + '_cum']
            if count == 0:
                result = result_temp
            else:
                result = result.merge(result_temp,left_index=True,right_index=True,how='left')
            count += 1
   
    data = data.merge(result,left_index=True,right_index=True,how='left')

    return data

def dominance(data,field,not_field=[]):

    col_list = []
    for col in data.columns:
        if 'is' in col and 'cum' in col and field in col and 'total' not in col and 'percent' not in col and 'dominance' not in col:
            col_list.append(col+'_dominance')

            data[col + '_dominance'] = data[col]/data[col + '_total']

            data.replace([np.inf, -np.inf], np.nan,inplace=True)

            data[col + '_dominance'] = data[col + '_dominance'].fillna(0)

    return data

def rank(data,col_base,no=[]):

    """Rank the values in a set of fields to create anonymous ranking fields                                                                     
       e.g. first_major_sector_percent, second_major_sector_percent, ..."""

    #find matching columns                                                                                                                             
    col_list = []
    for col in data.columns:
        match = True
        for base in col_base:
            if base not in col:
                match = False
        if match:
            col_list.append(col)

    data_sub = data[col_list]

    #sort the columns by value                                                                                                                          
    data_array = np.array(data_sub)
    data_array.sort(axis=1)
    data_array = np.fliplr(data_array)

    #create data frame with column names                                                                                                                 
    df = pd.DataFrame(data_array,index=data.index,columns=['_'.join(('_'.join(col_base),str(i + 1))) for i in range(len(col_list))])

    return df

def get_engine():

    config = ConfigParser.RawConfigParser()
    config.read('config')
    password = config.get('SQL','password')

    engine = create_engine(r'postgresql://dssg:' + password + '@localhost/world_bank')

    return engine

def write_sql_query(fields,table_name,years=0,amount=False,total=False,table_name2=''):

    if table_name2 == '':
        table_name2 = table_name

    sql_base = 'SELECT st1.supplier_reformat,st1.contract_signing_date, st1.amount_standardized,st1.unique_id'
    
    for field in fields:
 
        if not total:
            sql_base += ',\nSUM(st2."' + field + '") AS "' + field + '_cum"'
        else:
            sql_base += ',\nSUM(st2."' + field + '") AS "' + field + '_cum_total"'

    sql_base += '\nFROM\n' 
    sql_base += table_name + ' AS st1\n'
    sql_base += 'INNER JOIN\n'
    sql_base += table_name2 + ' AS st2\n'
    sql_base += 'ON\n'
    sql_base += 'st2.contract_signing_date <= st1.contract_signing_date'
    if years != 0:
        sql_base += ' AND\n st2.contract_signing_date >= st1.contract_signing_date::date - ' + str(years*365)
    if not total:
        sql_base += ' AND\n st2.supplier_reformat = st1.supplier_reformat'

    sql_base += '\nGROUP BY st1.contract_signing_date, st1.amount_standardized, st1.supplier_reformat, st1.unique_id\n'
    sql_base += 'ORDER BY st1.contract_signing_date'

    sql_base += ';'

    return sql_base
    
def fix_duplicate_columns(data):

    cols_fixed = []
    for col in data.columns:
        pattern_y = re.compile('.*_y')
        pattern_x = re.compile('.*_x')
        if pattern_y.match(col):
            data.drop(col,axis=1,inplace=True)
        elif pattern_x.match(col):
            cols_fixed.append(col[:-2])
        else:
            cols_fixed.append(col)

    data.columns = cols_fixed

    return data


def setup_binary_fields(contracts,amounts,categories):
    
        print 'Generating binary fields...'
        start = time.time()
        boolean_fields = []
        for field in categories:
   #         boolean_fields.append([])
            print '   ' + field + '...',
            contracts = binarize(contracts,field)

            for col in contracts.columns:
                if 'is' in col and field in col and len(categories) != 2:
                    if not amounts:
                        boolean_fields.append(col)
                    else:
                        boolean_fields.append(re.sub('_ct$','',col) + '_amt')
            
                    
        print time.time() - start, 's elapsed'


        if len(categories) == 2:
            print 'Generating combined binary fields...'
            start = time.time()
          #  boolean_fields.append([])
            for cat1 in contracts[categories[0]].unique():
                for cat2 in contracts[categories[1]].unique():
                    if ( (contracts[categories[0]] == cat1) & (contracts[categories[1]] == cat2)).sum()  > 0:
                        col_name = '_'.join(('is',categories[0],categories[1],cat1,cat2 ,'ct'))
                        contracts[col_name] = (contracts[categories[0]] == cat1) & (contracts[categories[1]] == cat2)
                        contracts[col_name] = contracts[col_name].astype('int64')
                        if not amounts:
                            boolean_fields.append(col_name)
                        if amounts:
                            boolean_fields.append(re.sub('_ct$','',col_name) + '_amt')
        print time.time() - start, 's elapsed'


        print 'Boolean fields: ',len(boolean_fields)

        print 'Conditional amounts...'
        if amounts:
            contracts = conditional_amounts(contracts)
        print time.time() - start, 's elapsed'

        return contracts,boolean_fields


def drop_duplicate_cols(contracts):

    cols_fixed = []
    for col in contracts.columns:
        pattern_y = re.compile('.*_y')
        pattern_x = re.compile('.*_x')
        if pattern_y.match(col):
            print 'dropping ' + col
            contracts.drop(col,axis=1,inplace=True)
        elif pattern_x.match(col):
            print 'keeping ' + col,col[:-2]
            cols_fixed.append(col[:-2])
        else:
            cols_fixed.append(col)

    contracts.columns = cols_fixed


    col_list = []
    for i,col in enumerate(contracts.columns):
        if col not in col_list:
            col_list.append(col)
        else:
            col_list.append(col + '2')

    contracts.columns = col_list

    return contracts

def cleaning(contracts,categories):

    """Drop duplicate column names, reformat names, """

    drop_duplicate_cols(contracts)

    contracts = reformat(contracts,'supplier')
    contracts = reformat(contracts,'country',inplace=True)
    contracts = reformat(contracts,'region',inplace=True,shorten=True)


    contracts['major_sector'][contracts['major_sector'].str.contains("\(H\)")] = 'Other'
    contracts['major_sector'][contracts['major_sector'].str.contains("X")] = 'Other'
    contracts['major_sector'][contracts['major_sector'].str.contains("Not assigned")] = 'Other'


    contracts['prc_ctg'] = contracts['procurement_category'] 
    contracts['prc_typ'] = contracts['procurement_type'] 
       
    contracts = reformat(contracts,'major_sector',inplace=True,shorten=True)
    contracts = reformat(contracts,'prc_ctg',inplace=True,shorten=True)
    contracts = reformat(contracts,'prc_typ',inplace=True,shorten=True)

    contracts['ctry'] = contracts['country']
    contracts['rgn'] = contracts['region']
    contracts['sect'] = contracts['major_sector']

    #interesting columns
    contracts = contracts[['supplier_reformat','supplier','contract_signing_date',
                           'amount_standardized','wb_contract_number','unique_id'] + categories]


    contracts = contracts[contracts['amount_standardized'].notnull()]
     
    contracts['amount_standardized'] = contracts['amount_standardized'].astype('int64')

   
    #convert date to datetime
    contracts['contract_signing_date'] = pd.to_datetime(contracts['contract_signing_date'])

    return contracts


def main():

    print 'Connecting to database...',
    start = time.time()
    engine = get_engine()
    con = engine.connect()
    print time.time() - start,'s elapsed'
    

    print 'Reading data...',
    start = time.time()
    contracts = pd.read_csv(args.contract_file)
  #  contracts = pd.read_csv('/mnt/data/world-bank/joinedcontracts_features_phase4_resolved.csv')
#    labelled_contracts = pd.read_csv('/mnt/data/world-bank/joinedcontracts_features_phase4_supplier_features_labelled_resolved.csv')
    labelled_contracts = pd.read_csv(args.invest_file)
    print time.time() - start, 's elapsed'

    print labelled_contracts.shape
    if len(labelled_contracts.index) > args.contract_num_lim:
        labelled_contracts.sort(['contract_signing_date'],inplace=True)
        labelled_contracts = labelled_contracts.head(args.contract_num_lim)
    print labelled_contracts.shape

    contracts['unique_id'] = contracts.index
    labelled_contracts['unique_id'] = labelled_contracts.index

    labelled_contracts.to_sql(args.invest_file.split('/')[-1].split('.')[0] + '_' + args.table_id,engine,if_exists='replace')

    #drop duplicate column names
    contracts = drop_duplicate_cols(contracts)
    labelled_contracts = drop_duplicate_cols(labelled_contracts)

    #make sure labelled contracts are included in contracts (Should be true anyway)
    contracts = pd.concat([contracts,labelled_contracts[contracts.columns]])
    contracts.drop_duplicates(inplace=True,cols=['supplier','wb_contract_number','major_sector','amount_standardized'])
    
    amounts = args.amounts
    dist_bool = args.dist
    dom_bool = args.dom
    categories = args.categ
    dt = args.num_years

    supplier_list = labelled_contracts['supplier'].unique()

    if dist_bool:
        #we don't care about the overall distribution so limit ourselves to labelled suppliers
        print len(contracts.index)
        contracts = contracts[contracts['supplier'].isin(supplier_list)]
        print len(contracts.index)

    if dom_bool:
        #only need total counts for fields present in labelled data
        for categ in categories:
            print len(contracts.index)
            categ_list = labelled_contracts[categ].unique()
            contracts = contracts[contracts[categ].isin(categ_list)]
            print len(contracts.index)

 
    categs_temp = []
    for categ in categories:
        if categ == 'major_sector':
            categ = 'sect'
        if categ == 'country':
            categ = 'ctry'
        if categ == 'region':
            categ = 'rgn'
        if categ == 'procurement_category':
            categ = 'prc_ctg'
        if categ == 'procurement_type':
            categ = 'prc_typ'
        categs_temp.append(categ)
    categories = categs_temp

    #clean data and create dummy boolean fields
    contracts = cleaning(contracts,categories)
    labelled_contracts = cleaning(labelled_contracts,categories)
    
    contracts,boolean_fields = setup_binary_fields(contracts,amounts,categories)
    labelled_contracts,boolean_fields_labelled = setup_binary_fields(labelled_contracts,amounts,categories)



    start_cols = labelled_contracts.columns

    print 'Num years: ', dt
    field = '_'.join(categories)


    field_list = boolean_fields
    field_list_labelled = boolean_fields_labelled

    field_list = [val for val in boolean_fields_labelled if val in set(boolean_fields)]

    if True:
#    for field_list,field_list_labelled in zip(boolean_fields,boolean_fields_labelled):


        table_name = 'contracts_w_booleans_' + args.table_id
        if amounts:
            table_name = '_'.join((table_name,'amt',field))
        else:
            table_name = '_'.join((table_name,field))


        result = con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name;")
        result = list(result.fetchall())
        tables = [r[0] for r in result]
        if True: 

            print 'Running full table'

            print 'Writing to database...'
            start = time.time()
            contracts_boolean_fields = contracts[['supplier_reformat','contract_signing_date',
                                                  'amount_standardized','unique_id'] + field_list]
            con.execute('DROP TABLE IF EXISTS ' + table_name + ';')
            print len(contracts_boolean_fields.index)
            for q in range((len(contracts_boolean_fields.index) / 5000) + 1):
                subset = contracts_boolean_fields.iloc[q*5000:min((q+1)*5000,len(contracts_boolean_fields.index))] 
                print q, subset.shape
                if (q==0):
		    subset.to_sql(table_name,engine,if_exists='replace')
		else:
		    subset.to_sql(table_name,engine,if_exists='append')


        print 'Writing to database...',
        table_name2 = 'contracts_w_booleans_lab_' + args.table_id
        if amounts:
            table_name2 = '_'.join((table_name2,'amt',field))
        else:
            table_name2 = '_'.join((table_name2,field))

        start = time.time()
        contracts_boolean_fields_labelled = labelled_contracts[['supplier_reformat','contract_signing_date',
                                                                'amount_standardized','unique_id'] 
                                                               + field_list]
        con.execute('DROP TABLE IF EXISTS ' + table_name2 + ';')
        contracts_boolean_fields_labelled.to_sql(table_name2, engine)
        print time.time() - start,'s elapsed'


        total_agg = [False]

        if dom_bool:
            total_agg.append(True)

        for tagg in total_agg:
            print 'Running SQL statement...',tagg,
            start = time.time()
            sql_statement = write_sql_query(field_list,
                                            table_name2,
                                            total=tagg,
                                            table_name2=table_name)
            result = con.execute(sql_statement)
            print result
	    sql_results = pd.DataFrame(result.fetchall())
            sql_results.columns = result.keys()
             
            for col in sql_results.columns:
                if 'ct_cum' in col or 'amt_cum' in col:
                    sql_results[col] = sql_results[col].astype(float)

            print labelled_contracts.shape
            labelled_contracts = labelled_contracts.merge(sql_results,
                                                          on=['supplier_reformat', 
                                                              'contract_signing_date', 
                                                              'amount_standardized',
                                                              'unique_id'],
                                                          how='left')
            print labelled_contracts.shape
            print time.time() - start,'s elapsed'


    print 'Generating supplier specific counts...'
    start = time.time()

    
    print '   ' + field + '...'
             
    labelled_contracts = labelled_contracts.sort(['supplier','contract_signing_date'])
                    
    if dist_bool:
        print '         distribution...',
        start = time.time()
        dist = distribution(labelled_contracts,field,amount=amounts)        
        labelled_contracts = labelled_contracts.merge(dist,left_index=True,right_index=True,how='left')
        print time.time() - start, 's elapsed'

    if dom_bool:
        print '         dominance...',
        start = time.time()
        labelled_contracts = dominance(labelled_contracts,field)
        print time.time() - start, 's elapsed'


    #drop temperary fields
    for col in labelled_contracts.columns:
        if '_total' in col:
            labelled_contracts.drop(col,axis=1,inplace=True)
      

    print 'Creating anonymous ranking features...'
    start = time.time()
    if dist_bool:
        if not amounts:
            print field
            anonymous_dist = rank(labelled_contracts,col_base=[field,'percent','ct'])
        else:
            anonymous_dist = rank(labelled_contracts,col_base=[field,'percent','amt'])
        labelled_contracts = labelled_contracts.merge(anonymous_dist,left_index=True,right_index=True)
    
        print time.time() - start, 's elapsed'

    cols_added = labelled_contracts.columns.difference(start_cols).tolist()
        
    dt_name = 'full'
    if int(dt) != 0:
        dt_name = str(dt) + 'years'

    cols_renamed = []
    for col in cols_added:
        cols_renamed.append(col + '_' + dt_name)

    dictionary = dict(zip(cols_added, cols_renamed))

    labelled_contracts.rename(columns=dictionary,inplace=True)

    labelled_contracts = labelled_contracts.sort(['supplier','contract_signing_date'])

    booleans = [inner for outer in boolean_fields_labelled for inner in outer]

    contracts_to_write = labelled_contracts[labelled_contracts.columns - booleans]

    contracts_to_write.columns = [col.replace('country','cty') for col in contracts_to_write.columns]
    contracts_to_write.columns = [col.replace('percent','pct') for col in contracts_to_write.columns]
    contracts_to_write.columns = [col.replace('major_sector','sect') for col in contracts_to_write.columns]
    contracts_to_write.columns = [col.replace('dominance','dom') for col in contracts_to_write.columns]
    contracts_to_write.columns = [col.replace('amount','amt') for col in contracts_to_write.columns]
    contracts_to_write.columns = [col.replace('years','yr') for col in contracts_to_write.columns]

    contracts_to_write.columns = [col.lower() for col in contracts_to_write.columns]

    contracts_to_write = contracts_to_write.fillna(0)
    
    zero_cols = contracts_to_write.apply(lambda x: np.all(x==0))


    for col,value in zip(zero_cols.index,zero_cols):
        if value:
            contracts_to_write.drop(col,axis=1,inplace=True)
            
    if amounts:
        agg_types = ['amt_cum_pct','pct_amt']
    else:
        agg_types = ['ct_cum_pct','pct_ct']

  
    already_used = ['unique_id','supplier_reformat','supplier',
                    'wb_contract_number','sect','region','ctry',
                    'contract_signing_date','amt_standardized']

    for agg_type in agg_types:
        
        final_cols = ['unique_id','supplier_reformat','supplier',
                      'wb_contract_number','contract_signing_date',
                      'amt_standardized'] + categories
        for col in contracts_to_write.columns:
            if agg_type in col and col not in already_used:
                already_used.append(col)
                final_cols.append(col)


        to_write_subset = contracts_to_write[final_cols]

        output_name = '_'.join(('cntrcts_splr_ftr_set_' + args.table_id,field,agg_type))
        if dist_bool:
            output_name += '_dist'
        if dom_bool:
            output_name += '_dominance'
        output_name += '_' + dt_name

#        output_name += '_test2'

        con.execute('DROP TABLE IF EXISTS ' + output_name + ';')
        to_write_subset.to_sql(output_name,engine)


    print labelled_contracts.shape
    print contracts.shape


if __name__ == "__main__":

    main()
