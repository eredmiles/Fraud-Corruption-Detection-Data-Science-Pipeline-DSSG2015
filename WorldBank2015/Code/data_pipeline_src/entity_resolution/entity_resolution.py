import pandas as pd
import argparse
#import sql

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--contracts_file_name',help='Input file')
    parser.add_argument('-e', '--canonincal_entity_file_name', help='Input file')
    parser.add_argument('-o', '--output_file', type=str, default="out_cleaned.csv", help='Name of output file with full path')
    args = parser.parse_args()


    contracts = pd.read_csv(args.contracts_file_name)
    entity_resolver = pd.read_csv(args.canonincal_entity_file_name, delimiter='\t')

    entity_resolver = entity_resolver[['Entity Name','Canonical Name (Semantic)']]

#    entity_resolver['Entity Name'] =  entity_resolver['Entity Name'].map(lambda x: x.replace('.', ''))
#    entity_resolver['Entity Name'] =  entity_resolver['Entity Name'].map(lambda x: x.replace(',', ''))
#    entity_resolver['Entity Name'] =  entity_resolver['Entity Name'].map(lambda x: x.replace('"', ''))
#    entity_resolver['Entity Name'] = entity_resolver['Entity Name'].map(lambda x: x.replace(')', ''))
#    entity_resolver['Entity Name'] = entity_resolver['Entity Name'].map(lambda x: x.replace('(', ''))
#    entity_resolver['Canonical Name (Semantic)'] =  entity_resolver['Canonical Name (Semantic)'].map(lambda x: str(x).replace('"', ''))
#    entity_resolver['Canonical Name (Semantic)'] =  entity_resolver['Canonical Name (Semantic)'].map(lambda x: str(x).replace('.', ''))
#    entity_resolver['Canonical Name (Semantic)'] =  entity_resolver['Canonical Name (Semantic)'].map(lambda x: str(x).replace('"', ''))
#    entity_resolver['Canonical Name (Semantic)'] =  entity_resolver['Canonical Name (Semantic)'].map(lambda x: str(x).replace(')', ''))
#    entity_resolver['Canonical Name (Semantic)'] =  entity_resolver['Canonical Name (Semantic)'].map(lambda x: str(x).replace('(', ''))
#    print contracts['supplier']
#    print type(contracts['supplier'][0])
#    contracts['supplier'] = contracts['supplier'].map(lambda x: str(x).replace('.', ''))
#    contracts['supplier'] = contracts['supplier'].map(lambda x: str(x).replace(',', ''))
#    contracts['supplier'] = contracts['supplier'].map(lambda x: str(x).replace('"', ''))
#    contracts['supplier'] = contracts['supplier'].map(lambda x: str(x).replace(')', ''))
#    contracts['supplier'] = contracts['supplier'].map(lambda x: str(x).replace('(', ''))
    df_new = pd.DataFrame({'Entity Name':[], 'Canonical Name (Semantic)':[]})
    df_new['Entity Name']=entity_resolver['Canonical Name (Semantic)']
   # df_new['Canonical Name (Syntactic)'] = entity_resolver['Canonical Name (Syntactic)']
    df_new['Canonical Name (Semantic)'] = entity_resolver['Canonical Name (Semantic)']
    entity_resolver = entity_resolver.append(df_new)
    def get_unique(x):
        
        try:
            return x.unique()[0]
        except:
            return np.nan

    entities = pd.DataFrame(entity_resolver.groupby('Entity Name')['Canonical Name (Semantic)'].apply(get_unique))


    entity_dict = entities.to_dict()['Canonical Name (Semantic)']


    contracts['resolved_supplier'] = contracts['supplier'].map(entity_dict)
#    print contracts['resolved_supplier']
  #  contracts['resolved_supplier'] = contracts['resolved_supplier'].map(lambda x: str(x).replace('.', ''))
 #   contracts['resolved_supplier'] = contracts['resolved_supplier'].map(lambda x: str(x).replace(',', ''))
#    contracts['resolved_supplier'] = contracts['resolved_supplier'].map(lambda x: str(x).replace('"', ''))

   # contracts['resolved_supplier']=contracts['resolved_supplier'].str.replace('\s', '')
    contracts['orig_supplier_name'] = contracts['supplier']
    print "Number of unique entities before resolution: ",contracts['orig_supplier_name'].nunique()
    print "Number of unique resolved entities: ", contracts['resolved_supplier'].nunique()
    contracts['supplier'] = contracts['resolved_supplier']
    contracts.to_csv(args.output_file)
  #  entity_resolver.to_csv('./entity_resolver.csv')

if __name__ == '__main__':
    main()
