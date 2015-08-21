import pandas as pd
import sql
import re

def fix_country_names(data):

    """Match the contract country names with the format used in the map data"""

    replace_dict = { 'Yemen, Republic' : 'Yemen' ,
                     'Venezuela, Repu' : 'Venezuela',
                     'Syrian Arab Rep' : 'Syria',
                     'Slovak Republic' : 'Slovakia',
                     'Congo, Democrat' : 'Democratic Republic of Congo',
                     'Congo, Republic' : 'Republic of Congo',
                     'Egypt, Arab Rep' : 'Egypt',
                     'Gambia, The' : 'The Gambia',
                     'Iran, Islamic R' : 'Iran',
                     'Kyrgyz Republic' : 'Kyrgyzstan',
                     'Korea, Republic' : 'Republic of Korea',
                     "Lao People's De" : "Lao PDR",
                     "Macedonia, form" : "Macedonia",
                     "Central African" : "Central African Republic"
    }


    data['country_name_standardized'] = data['country'].replace(replace_dict)

    return data


def snake_case(name):

    """Clean entity name strings"""
    try:
        s1 = re.sub(' +','_',name)
        s1 = re.sub('-','_',s1)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s1)
        s1 = s1.replace('.','')
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

def reformat(data,column):

    data[column + '_reformat'] = data[column].map(lambda x: snake_case(x))

    return data

def main():
    
    data = sql.read_sql('investigation_data')
    data = reformat(data,'subject')
#    sql.write_sql(data,'investigation_data2')
    data.to_csv('investigation_data2.csv')

    data = sql.read_sql('joinedcontracts_features_phase2')
    data = reformat(data,'supplier')
#    sql.write_sql(data,'joinedcontracts_features_phase2_2')
    data.to_csv('joinedcontracts_features_phase2_2.csv')



if __name__ == "__main__":
    main()