from subprocess import call
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-cf','--contract_file',help='Contract data file')
parser.add_argument('-if','--invest_file',help='Labelled data file')
parser.add_argument('-id','--table_id',default='default',help='ID for database tables')
parser.add_argument('-path','--path',default='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015/WorldBank2015/Code/data_pipeline_src/supplier_feature_gen.py',help='path to supplier feature gen')
args = parser.parse_args()

for i in [0,1,3,5]:
	for s in ["region","country","procurement_category","major_sector","procurement_type"]:
		for a in [' -a','']:
			command = 'python -W ignore '+args.path+' -cf '+args.contract_file+ ' -if '+ args.invest_file +' -y '+ str(i)+ ' -cat '+ s + a + ' -id ' + args.table_id
			os.system(command)
