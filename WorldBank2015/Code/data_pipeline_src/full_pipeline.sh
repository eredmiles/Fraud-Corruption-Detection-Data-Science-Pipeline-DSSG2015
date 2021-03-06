#!/bin/bash  
#Elissa Redmiles DSSG2015
#Script and Error logs are located in the log_files directory within this directory
 
LOCALPATH='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015'
DATA_STORAGE='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015/pipeline_data'
CURRENCY_FILE_PPP='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015/pipeline_data/ppp.csv'
CURRENCY_FILE_FCRF='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015/pipeline_data/fcrf.csv'
PATH_TO_SUPPLIER_FEATURE_GEN=$LOCALPATH'/WorldBank2015/Code/data_pipeline_src/supplier_feature_gen.py'


DATE=`date +"%m%d%Y_%H%M"`
STDOUT_LOG='./log_files/pipeline_'$DATE'.log'
ERROR_LOG='./log_files/pipeline_error_'$DATE'.log'
echo $STDOUT_LOG
echo $ERROR_LOG

echo Starting DSSG2015 World Bank Complaint Ranking Pipeline

#Setting Python Path - specific to allow desktop icon to run
#create an environment setup file if necessary for setting your system PATH and PYTHONPATH variables
###comment out if not needed
export PATH="/home/dssg/anaconda/bin":$PATH
export PYTHONPATH="/home/dssg/anaconda":$PYTHONPATH
echo $PYTHONPATH

echo ============================================
echo Ranked files will be output to $DATA_STORAGE
echo ============================================

echo Starting DSSG2015 World Bank Complaint Ranking Pipeline

### Data Loading ###
## Contracts ##
#Download a new set of contracts from the World Bank Website 
echo Downloading newest contracts data set from finances.worldbank.org.
echo ================================================================= | tee $STDOUT_LOG $ERROR_LOG >/dev/null
echo Downloading newest contracts data set from finances.worldbank.org. | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download.csv'
wget --output-document=$CONTRACTS_FILE https://finances.worldbank.org/api/views/kdui-wcs3/rows.csv?accessType=DOWNLOAD >> $STDOUT_LOG 2>> $ERROR_LOG

#Cleaning the contracts: formating dates, renaming columns, properly coding null values
echo Cleaning contracts data set
echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo Cleaning contracts data set | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

CLEANED_CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/data_cleaning_script.py' -f $CONTRACTS_FILE -n '#' -r "Borrower Country,country,Total Contract Amount (USD),amount" -o $CLEANED_CONTRACTS_FILE >> $STDOUT_LOG 2>> $ERROR_LOG

#Entity Resolution Application
echo Resolving entities in contracts data set.

echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo Resolving entities in contracts data set. | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

CANONICAL_FILE=$DATA_STORAGE'/canonicalNamesV2.csv'
ENTITY_RESOLVED_CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned_resolved.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/entity_resolution/entity_resolution.py' -c $CLEANED_CONTRACTS_FILE -e $CANONICAL_FILE -o $ENTITY_RESOLVED_CONTRACTS_FILE >> $STDOUT_LOG 2>> $ERROR_LOG

#Loading contract data into PostgreSQL database
echo Loading contracts data set into database.

echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo Loading contracts data set into database. | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

CONTRACTS_TABLE='latest_contract_web_download_cleaned_resolved'
echo drop table \"$CONTRACTS_TABLE\"';' > droptable.sql
psql world_bank -f droptable.sql >> $STDOUT_LOG  2>> $ERROR_LOG
csvsql -i postgresql $ENTITY_RESOLVED_CONTRACTS_FILE > createtable.sql
#';'>>createtable.sql
psql world_bank -f createtable.sql >> $STDOUT_LOG 2>> $ERROR_LOG 
echo \\copy \"$CONTRACTS_TABLE\" FROM \'$ENTITY_RESOLVED_CONTRACTS_FILE\' CSV HEADER\; > copydata.sql
psql world_bank -f copydata.sql >> $STDOUT_LOG 2>> $ERROR_LOG

## Projects ##
#Getting project data from World Bank Website which will later be joined with the contracts data
echo Downloading newest projectts data set from finances.worldbank.org.
echo ================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo Downloading newest projects data set from finances.worldbank.org.  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

PROJECTS_FILE=$DATA_STORAGE'/project_data.csv'
wget http://search.worldbank.org/api/projects/all.csv -O $PROJECTS_FILE >> $STDOUT_LOG 2>> $ERROR_LOG

#Cleaning the project data
echo Cleaning projects data set.


echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo Cleaning projects data set   | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/data_cleaning_generic.py' -f $PROJECTS_FILE -o $PROJECTS_FILE >> $STDOUT_LOG 2>> $ERROR_LOG
#Loading the project data to the PostgreSQL database
echo Loading projects data set into database

echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo Loading projects data set  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null


PROJECTS_TABLE='project_data'
echo drop table \"$PROJECTS_TABLE\"';' > droptable.sql
psql world_bank -f droptable.sql >> $STDOUT_LOG 2>> $ERROR_LOG
csvsql -i postgresql $PROJECTS_FILE > createtable.sql
psql world_bank -f createtable.sql >> $STDOUT_LOG 2>> $ERROR_LOG
echo \\copy \"$PROJECTS_TABLE\" FROM \'$PROJECTS_FILE\' CSV HEADER';' > copydata.sql
psql world_bank -f copydata.sql >> $STDOUT_LOG 2>> $ERROR_LOG

#Joining Project data and Contracts data

echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo joining project and contract data  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

JOINED_TABLE='contracts_wprojects'
echo DROP TABLE $JOINED_TABLE';'>droptable.sql
psql world_bank -f droptable.sql >> $STDOUT_LOG 2>> $ERROR_LOG
echo "ALTER TABLE $CONTRACTS_TABLE DROP COLUMN _unnamed;">dropcolumn.sql
psql world_bank -f dropcolumn.sql >> $STDOUT_LOG 2>> $ERROR_LOG
echo "ALTER TABLE $PROJECTS_TABLE DROP COLUMN _unnamed;">dropcolumn.sql
psql world_bank -f dropcolumn.sql >> $STDOUT_LOG 2>> $ERROR_LOG
echo "AlTER TABLE $PROJECTS_TABLE DROP COLUMN project_name;">dropcolumn.sql
psql world_bank -f dropcolumn.sql >> $STDOUT_LOG 2>> $ERROR_LOG
echo "AlTER TABLE $PROJECTS_TABLE DROP COLUMN country;">dropcolumn.sql
psql world_bank -f dropcolumn.sql >> $STDOUT_LOG 2>> $ERROR_LOG
echo "CREATE TABLE $JOINED_TABLE AS SELECT * FROM $CONTRACTS_TABLE LEFT JOIN $PROJECTS_TABLE ON ($CONTRACTS_TABLE.project_id = $PROJECTS_TABLE.id);" > joinprojectdata.sql
psql world_bank -f joinprojectdata.sql>> $STDOUT_LOG 2>> $ERROR_LOG

## Feature Generation ##
#Generate contracts features
echo Generating data abstractions \(features\) for at the contracts level.
echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo Generating contract features  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

FEATURE_GEN_1_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned_feature_gen1.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/contracts_feature_gen.py' -f $ENTITY_RESOLVED_CONTRACTS_FILE -p procurement_method -wf $FEATURE_GEN_1_FILE -ppp $CURRENCY_FILE_PPP -fcrf $CURRENCY_FILE_FCRF >> $STDOUT_LOG 2>> $ERROR_LOG

#Adding Labels Data Set
###Cleaning the investigation data
echo Merging contracts, projects and investigations data sets. 
echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo Merging contracts,projects and investigations data sets. | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null


LABELS_FILE=$DATA_STORAGE'/investigations.csv'
echo Investigations file being used: $LABELS_FILE
echo Last Modification Date for Investigations File:
stat -c %y $LABELS_FILE

UTF8_LABELS_FILE=$DATA_STORAGE'/utf8_investigations.csv'
iconv -f utf-8 -t ascii -c $LABELS_FILE -o $UTF8_LABELS_FILE >> $STDOUT_LOG 2>> $ERROR_LOG
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/data_cleaning_generic.py' -f $UTF8_LABELS_FILE -o $UTF8_LABELS_FILE >> $STDOUT_LOG 2>> $ERROR_LOG

LABELED_CONTRACTS_FEATURE_GEN_1=$DATA_STORAGE'/labeled_contracts_cleaned_resolved_feature_gen_1.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/label.py' -c $FEATURE_GEN_1_FILE -i $UTF8_LABELS_FILE -wf $LABELED_CONTRACTS_FEATURE_GEN_1 >> $STDOUT_LOG 2>> $ERROR_LOG

#Generating supplier features for allegations
echo Generating supplier features for allegations data set.
echo =================================================================  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo generating supplier features for allegations | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

TABLE_ID_FOR_STORING_ALLEGATION_FEATURES='alleg'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/feature_loop.py' -cf $FEATURE_GEN_1_FILE -if $LABELED_CONTRACTS_FEATURE_GEN_1 -id $TABLE_ID_FOR_STORING_ALLEGATION_FEATURES -path $PATH_TO_SUPPLIER_FEATURE_GEN >> $STDOUT_LOG 2>> $ERROR_LOG

#Generating ranked list of complaints based on model.
echo Generating ranked list of allegations.
echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo generating allegations ranked list | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
TRAINING_TABLE_NAME='labeled_contracts_cleaned_resolved_feature_gen_1_alleg'
RANKED_LIST_OF_ALLEGATIONS_FILE_NAME=$DATA_STORAGE'/prioritzed_allegations_to_investigate.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/modeling/predict.py' -tf $TRAINING_TABLE_NAME -fl $LOCALPATH'/WorldBank2015/Code/modeling/feature_sets_log.yaml' -train_id $TABLE_ID_FOR_STORING_ALLEGATION_FEATURES -wf $RANKED_LIST_OF_ALLEGATIONS_FILE_NAME>>$STDOUT_LOG 2>> $ERROR_LOG

#Creating supplier features for contracts set
echo Generating supplier features for contracts data set.
echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo generating supplier features for contracts | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

TABLE_ID_FOR_STORING_CONTRACTS_FEATURES='cntrcts'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/feature_loop.py' -cf $FEATURE_GEN_1_FILE -if $FEATURE_GEN_1_FILE -id $TABLE_ID_FOR_STORING_CONTRACTS_FEATURES -path $PATH_TO_SUPPLIER_FEATURE_GEN >> $STDOUT_LOG 2>> $ERROR_LOG

#Generating ranked list of contracts in the last N years based on model.
echo Generating ranked list of contracts.
echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
echo generating contracts ranked lists  | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null

echo ================================================================= | tee -a $STDOUT_LOG $ERROR_LOG > /dev/null
TRAINING_TABLE_NAME='labeled_contracts_cleaned_resolved_feature_gen_1_alleg'
PREDICTING_TABLE_NAME='latest_contract_web_download_cleaned_feature_gen1_cntrcts'
OUTPUT_FILE_NAME=$DATA_STORAGE'/ranked_contracts_to_investigate_for_'
python -W ignore $LOCALPATH'/WorldBank2015/Code/modeling/prediction_loop.py' -tf $TRAINING_TABLE_NAME -pf $PREDICTING_TABLE_NAME -fl $LOCALPATH'/WorldBank2015/Code/modeling/feature_sets_log.yaml' -train_id $TABLE_ID_FOR_STORING_ALLEGATION_FEATURES -pred_id $TABLE_ID_FOR_STORING_CONTRACTS_FEATURES -path $LOCALPATH'/WorldBank2015/Code/modeling/predict.py' -fp $OUTPUT_FILE_NAME >>$STDOUT_LOG 2>> $ERROR_LOG


