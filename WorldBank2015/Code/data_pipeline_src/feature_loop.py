from subprocess import call
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-cf','--contract_file',help='Contract data file')
parser.add_argument('-if','--invest_file',help='Labelled data file')
parser.add_argument('-id','--table_id',default='default',help='ID for database tables')
args = parser.parse_args()

for i in [0,1,3,5]:
	for s in ["region","country","procurement_category","major_sector","procurement_type"]:
		for a in [' -a','']:
			command = 'python -W ignore supplier_feature_gen.py -cf '+args.contract_file+ ' -if '+ args.invest_file +' -y '+ str(i)+ ' -cat '+ s + a + ' -id ' + args.table_id
			os.system(command)
