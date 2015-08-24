#!/bin/bash  
#Elissa Redmiles DSSG2015
#Command outputs are located in pipeline.log

LOCALPATH='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015'
DATA_STORAGE='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015/pipeline_data'
CURRENCY_FILE_PPP='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015/pipeline_data/ppp.csv'
CURRENCY_FILE_FCRF='/home/dssg/dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015/pipeline_data/fcrf.csv'
echo Starting DSSG2015 World Bank Complaint Ranking Pipeline

### Data Loading ###
## Contracts ##
#Download a new set of contracts from the World Bank Website 
echo Downloading newest contracts data set from finances.worldbank.org.
echo ================================================================= >pipeline.log
echo Downloading newest contracts data set from finances.worldbank.org. >>pipeline.log
echo ================================================================= >>pipeline.log
CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download.csv'
wget --output-document=$CONTRACTS_FILE https://finances.worldbank.org/api/views/kdui-wcs3/rows.csv?accessType=DOWNLOAD >> pipeline.log

#Cleaning the contracts: formating dates, renaming columns, properly coding null values
echo Cleaning contracts data set
echo ================================================================= >>pipeline.log
echo Cleaning contracts data set >>pipeline.log
echo ================================================================= >>pipeline.log
CLEANED_CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/data_cleaning_script.py' -f $CONTRACTS_FILE -n '#' -r "Borrower Country,country,Total Contract Amount (USD),amount" -o $CLEANED_CONTRACTS_FILE >> pipeline.log

#Entity Resolution Application
echo Resolving entities in contracts data set.

echo ================================================================= >>pipeline.log
echo Resolving entities in contracts data set. >>pipeline.log
echo ================================================================= >>pipeline.log
CANONICAL_FILE=$DATA_STORAGE'/canonicalNamesV2.csv'
ENTITY_RESOLVED_CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned_resolved.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/entity_resolution/entity_resolution.py' -c $CLEANED_CONTRACTS_FILE -e $CANONICAL_FILE -o $ENTITY_RESOLVED_CONTRACTS_FILE >> pipeline.log

#Loading contract data into PostgreSQL database
echo Loading contracts data set into database.

echo ================================================================= >>pipeline.log
echo Loading contracts data set into database. >>pipeline.log
echo ================================================================= >>pipeline.log

CONTRACTS_TABLE='latest_contract_web_download_cleaned_resolved'
echo drop table \"$CONTRACTS_TABLE\"';' > droptable.sql
psql world_bank -f droptable.sql >> pipeline.log 2>> pipeline.log
csvsql -i postgresql $ENTITY_RESOLVED_CONTRACTS_FILE > createtable.sql
#';'>>createtable.sql
psql world_bank -f createtable.sql >> pipeline.log 2>> pipeline.log 
echo \\copy \"$CONTRACTS_TABLE\" FROM \'$ENTITY_RESOLVED_CONTRACTS_FILE\' CSV HEADER\; > copydata.sql
psql world_bank -f copydata.sql >> pipeline.log 2>> pipeline.log

## Projects ##
#Getting project data from World Bank Website which will later be joined with the contracts data
echo Downloading newest projects data set from worldbank.org

echo ================================================================= >>pipeline.log
echo Loading contracts data set into database. >>pipeline.log
echo ================================================================= >>pipeline.log

PROJECTS_FILE=$DATA_STORAGE'/project_data.csv'
wget http://search.worldbank.org/api/projects/all.csv -O $PROJECTS_FILE >> pipeline.log

#Cleaning the project data
echo Cleaning projects data set.


echo ================================================================= >>pipeline.log
echo Cleaning projects data set  >>pipeline.log
echo ================================================================= >>pipeline.log



#sudo chmod a+x+r+w $PROJECTS_FILE
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/data_cleaning_generic.py' -f $PROJECTS_FILE -o $PROJECTS_FILE >> pipeline.log

#Loading the project data to the PostgreSQL database
echo Loading projects data set into database

echo ================================================================= >>pipeline.log
echo Loading projects data set  >>pipeline.log
echo ================================================================= >>pipeline.log

PROJECTS_TABLE='project_data'
echo drop table \"$PROJECTS_TABLE\"';' > droptable.sql
psql world_bank -f droptable.sql >> pipeline.log 2>> pipeline.log
csvsql -i postgresql $PROJECTS_FILE > createtable.sql
psql world_bank -f createtable.sql >> pipeline.log 2>> pipeline.log
echo \\copy \"$PROJECTS_TABLE\" FROM \'$PROJECTS_FILE\' CSV HEADER';' > copydata.sql
psql world_bank -f copydata.sql >> pipeline.log 2>> pipeline.log

#Joining Project data and Contracts data

echo ================================================================= >>pipeline.log
echo joining project and contract data  >>pipeline.log
echo ================================================================= >>pipeline.log


JOINED_TABLE='contracts_wprojects'
echo DROP TABLE $JOINED_TABLE';'>droptable.sql
psql world_bank -f droptable.sql >> pipeline.log 2>> pipeline.log
echo "ALTER TABLE $CONTRACTS_TABLE DROP COLUMN _unnamed;">dropcolumn.sql
psql world_bank -f dropcolumn.sql >> pipeline.log 2>> pipeline.log
echo "ALTER TABLE $PROJECTS_TABLE DROP COLUMN _unnamed;">dropcolumn.sql
psql world_bank -f dropcolumn.sql >> pipeline.log 2>> pipeline.log
echo "AlTER TABLE $PROJECTS_TABLE DROP COLUMN project_name;">dropcolumn.sql
psql world_bank -f dropcolumn.sql >> pipeline.log 2>> pipeline.log
echo "AlTER TABLE $PROJECTS_TABLE DROP COLUMN country;">dropcolumn.sql
psql world_bank -f dropcolumn.sql >> pipeline.log 2>> pipeline.log
echo "CREATE TABLE $JOINED_TABLE AS SELECT * FROM $CONTRACTS_TABLE LEFT JOIN $PROJECTS_TABLE ON ($CONTRACTS_TABLE.project_id = $PROJECTS_TABLE.id);" > joinprojectdata.sql
psql world_bank -f joinprojectdata.sql>> pipeline.log 2>> pipeline.log

## Feature Generation ##
#Generate contracts features
echo Generating data abstractions \(features\) for at the contracts level.
echo ================================================================= >>pipeline.log
echo Generating contract features  >>pipeline.log
echo ================================================================= >>pipeline.log
FEATURE_GEN_1_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned_feature_gen1.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/contracts_feature_gen.py' -f $ENTITY_RESOLVED_CONTRACTS_FILE -p procurement_method -wf $FEATURE_GEN_1_FILE -ppp $CURRENCY_FILE_PPP -fcrf $CURRENCY_FILE_FCRF >> pipeline.log

#Adding Labels Data Set
###Cleaning the investigation data
echo Merging contracts, projects and investigations data sets. 
echo ================================================================= >>pipeline.log
echo labeleing data  >>pipeline.log
echo ================================================================= >>pipeline.log
LABELS_FILE=$DATA_STORAGE'/investigations.csv'
echo Investigations file being used: $LABELS_FILE
echo Last Modification Date for Investigations File:
stat -c %y $LABELS_FILE

UTF8_LABELS_FILE=$DATA_STORAGE'/utf8_investigations.csv'
iconv -f utf-8 -t ascii -c $LABELS_FILE -o $UTF8_LABELS_FILE >> pipeline.log
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/data_cleaning_generic.py' -f $UTF8_LABELS_FILE -o $UTF8_LABELS_FILE >> pipeline.log

###Merging Contracts W Features and Labels
echo ================================================================= >>pipeline.log
echo merging contracts, features labels  >>pipeline.log
echo ================================================================= >>pipeline.log
LABELED_CONTRACTS_FEATURE_GEN_1=$DATA_STORAGE'/labeled_contracts_cleaned_resolved_feature_gen_1.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/label.py' -c $FEATURE_GEN_1_FILE -i $UTF8_LABELS_FILE -wf $LABELED_CONTRACTS_FEATURE_GEN_1 >> pipeline.log

#importing labeled feature gen 1 file to database
echo ================================================================= >>pipeline.log
echo Loading labeled feature gen 1 data set  >>pipeline.log
echo ================================================================= >>pipeline.log


#Generating supplier features for allegations
echo Generating supplier features for allegations data set.
echo ================================================================= >>pipeline.log
echo generating supplier features for allegations  >>pipeline.log
echo ================================================================= >>pipeline.log

TABLE_ID_FOR_STORING_ALLEGATION_FEATURES='alleg'
PATH_TO_SUPPLIER_FEATURE_GEN=$LOCALPATH'/WorldBank2015/Code/data_pipeline_src/supplier_feature_gen.py'

python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/feature_loop.py' -cf $FEATURE_GEN_1_FILE -if $LABELED_CONTRACTS_FEATURE_GEN_1 -id $TABLE_ID_FOR_STORING_ALLEGATION_FEATURES -path $PATH_TO_SUPPLIER_FEATURE_GEN >> pipeline.log

#Generating ranked list of complaints based on model.
echo Generating ranked list of allegations.
echo ================================================================= >>pipeline.log
echo generating allegations ranked list  >>pipeline.log
echo ================================================================= >>pipeline.log
TRAINING_TABLE_NAME='labeled_contracts_cleaned_resolved_feature_gen_1_alleg'
RANKED_LIST_OF_ALLEGATIONS_FILE_NAME=$DATA_STORAGE'/prioritzed_allegations_to_investigate.csv'
python -W ignore $LOCALPATH'/WorldBank2015/Code/modeling/predict.py' -tf $TRAINING_TABLE_NAME -fl $LOCALPATH'/WorldBank2015/Code/modeling/feature_sets_log.yaml' -train_id $TABLE_ID_FOR_STORING_ALLEGATION_FEATURES -wf $RANKED_LIST_OF_ALLEGATIONS_FILE_NAME>>pipeline.log

#Creating supplier features for contracts set
echo Generating supplier features for contracts data set.
echo ================================================================= >>pipeline.log
echo generating supplier features for contracts  >>pipeline.log
echo ================================================================= >>pipeline.log

TABLE_ID_FOR_STORING_CONTRACTS_FEATURES='cntrcts'
python -W ignore $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/feature_loop.py' -cf $FEATURE_GEN_1_FILE -if $FEATURE_GEN_1_FILE -id $TABLE_ID_FOR_STORING_CONTRACTS_FEATURES>> pipeline.log

#Generating ranked list of contracts in the last N years based on model.
echo Generating ranked list of contracts.
echo ================================================================= >>pipeline.log
echo generating contracts ranked lists  >>pipeline.log
echo ================================================================= >>pipeline.log
TRAINING_TABLE_NAME='labeled_contracts_cleaned_resolved_feature_gen_1'
PREDICTING_TABLE_NAME='latest_contract_web_download_cleaned_feature_gen1_cntrcts'
OUTPUT_FILE_NAME=$DATA_STORAGE'/prioritized_contracts_to_investigate_for_'
python -W ignore $LOCALPATH'/WorldBank2015/Code/modeling/prediction_loop.py' -tf $TRAINING_TABLE_NAME -pf $PREDICTING_TABLE_NAME -fl $LOCALPATH'/WorldBank2015/Code/modeling/feature_sets_log.yaml' -train_id $TABLE_ID_FOR_STORING_ALLEGATION_FEATURES -pred_id $TABLE_ID_FOR_STORING_CONTRACTS_FEATURES -path $LOCALPATH'/WorldBank2015/Code/modeling/predict.py' -fp $OUTPUT_FILE_NAME >>pipeline.log


