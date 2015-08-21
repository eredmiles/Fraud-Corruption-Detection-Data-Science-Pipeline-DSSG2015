#!/bin/bash

LOCALPATH='/home/eredmiles'
DATA_STORAGE='/mnt/data/world-bank/pipeline_data'
CLEANED_CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned.csv'
CANONICAL_FILE=$DATA_STORAGE'/canonicalNamesV2.csv'
ENTITY_RESOLVED_CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned_resolved.csv'
#Entity Resolution Application
echo "---Semantic Matching V2 file---"
python $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/entity_resolution/entity_resolution.py' -c $CLEANED_CONTRACTS_FILE -e $CANONICAL_FILE -o $ENTITY_RESOLVED_CONTRACTS_FILE
echo "---Syntactic Matching V2 file 8/3/15---"
python $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/entity_resolution/entity_resolution_syntactic.py' -c $CLEANED_CONTRACTS_FILE -e $CANONICAL_FILE -o $ENTITY_RESOLVED_CONTRACTS_FILE

#CANONICAL_FILE=$DATA_STORAGE'/canonicalNamesV2.csv'
CANONICAL_FILE=$DATA_STORAGE'/canonicalNamesV3.csv'
echo "---Semantic Matching V3 file 8/3/15---"
ENTITY_RESOLVED_CONTRACTS_FILE=$DATA_STORAGE'/latest_contract_web_download_cleaned_resolved.csv'
python $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/entity_resolution/entity_resolution.py' -c $CLEANED_CONTRACTS_FILE -e $CANONICAL_FILE -o $ENTITY_RESOLVED_CONTRACTS_FILE
echo "---Syntactic Matching V3 file 8/3/15---"
python $LOCALPATH'/WorldBank2015/Code/data_pipeline_src/entity_resolution/entity_resolution_syntactic.py' -c $CLEANED_CONTRACTS_FILE -e $CANONICAL_FILE -o $ENTITY_RESOLVED_CONTRACTS_FILE
