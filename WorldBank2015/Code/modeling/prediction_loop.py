from subprocess import call
import os
import argparse
import yaml
import model_pipeline_script
import pandas as pd

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('-tf','--training_table',help='Contract data file')
parser.add_argument('-pf','--prediction_table',help='File for prediction',default='')
parser.add_argument('-fl','--feature_log_file',help='Feature Log file',default='feature_sets_log.yaml')
parser.add_argument('-pred_id','--predict_table_id',help='Identifier for reading feature tables',default='')
parser.add_argument('-train_id','--train_table_id',help='Identifier for reading feature tables')
parser.add_argument('-fp','--file_path',help='Path and file name for output files', default='prioritized_list_of_contracts_to_investigate_for_')
parser.add_argument('-path','--predict_path',help="full path for predict.py")
args = parser.parse_args()

engine = model_pipeline_script.get_engine()
con = engine.connect()
contracts_data = pd.read_sql(args.training_table,engine)
count =0
#print contracts_data['allegation_category']
for allegation_type in contracts_data['allegation_category']:
	count = count+1
        allegation_type_for_file_name=allegation_type.replace(" ","_").lower()	
	allegation_type_for_file_name = allegation_type_for_file_name.replace('\\','').replace('.','').replace('/','')
	output_file_name=args.file_path+allegation_type_for_file_name+'.csv'
	
	if (args.prediction_table!=''):
		command = 'python -W ignore '+ args.predict_path+ ' -tf '+args.training_table + ' -pf '+ args.prediction_table +' -ac '+'\''+allegation_type+'\'' + ' -fl '+args.feature_log_file + ' -pred_id '+args.predict_table_id+' -train_id ' + args.train_table_id +' -wf '+ output_file_name
		print command
		os.system(command)
	else:
		command = 'python -W ignore '+ args.predict_path+ ' -tf '+args.training_table +' -ac '+ '\''+allegation_type+'\'' + ' -wf '+ output_file_name                 
                print command
		os.system(command)
