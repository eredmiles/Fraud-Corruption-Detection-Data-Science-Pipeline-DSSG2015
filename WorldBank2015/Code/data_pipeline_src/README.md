#Data Pipeline for World Bank 2015

<img src='pipeline_diagram-2.jpg' width=80% height=80%>

This process is automated with BASH. 
See [full_pipeline.sh](./full_pipeline.sh) for more information.
######Example Call
```bash
bash full_pipeline.sh
```

##Directory Contents
This directory contains the following files:
* config_example: an example of the config file needed for logging into your PostgreSQL database.
* data_cleaning_script.py: this script takes a CSV or XLS file and addresses the following data issues. data_cleaning_generic.py is a version of this file that can be used on non-world bank data.
  * Dates: Dates within and across data files are specified in different formats (e.g. '09-10-2000', '9/10/2000'). We created a series of nested try/except statements to account for these formats and standardize all dates to the python datetime format. 

  * Missing Values: users can specify additional strings to parse as 'nan' values. That is, if the file denotes missing entries with 'NC', the user can specify this information in the data cleaning script run command in order to properly parse the file. The data cleaning script removes also any identical rows 

  * Column names: the data cleaning script standardizes all column names to lower case and snake case (e.g. project\_name). Users can specify specific columns to rename and provide new names, as needed. Users can also specify columns to remove, as desired.

  * De-duplification: the Major Contracts data files that we received from the World Bank contained some instances of sets of rows which contained identical entries in each column except amount. We chose to remove these rows and create one row with an amount that is the sum of the amounts in the previous nearly-identical rows. 

######Example Call
```bash
python [path to script]/data_cleaning_script.py -f [CONTRACTS_FILE] -n '#' [value to consider as NAN] -r "Borrower Country,country,Total Contract Amount (USD),amount" [columsn to rename in the format "original name,new name" -o [OUTPUT FILE NAME]
```
* contracts_feature_gen.py: this script generates contract level features from a sql table that contains contracts and projects data which has been joined in table (see bash script for more info) (typically the cleaned contracts data) and outputs a new csv file with field for these features. 
 * The features generated are listed below.
![table of features]
(table_of_contract_features.png)
 * To learn more about these features, please see Section 2 of our [lab notebook](./WorldBankLabNotebook.pdf).

######Example Call
```bash
python [PATH TO FILE]/contracts_feature_gen.py -f [entity resolved contracts file] -p procurement_method -wf [OUTPUT FILE NAME]
```
* supplier_feature_gen.py and feature_loop.py: The purpose of these scripts is to generate features that are related to the historical behavior of a supplier prior to a contract of interest.  For example, this generates features such as the percent of a supplier's contracts in the past year that were in the agriculatural sector or the percent of supplier's previous contracts that were in Asia.  
 * feature_loop.py is a wrapper for supplier_feature_gen.py which loops over different categoricals (sector, country, region, procurement type, and procurement category) and different aggregation time periods (1 year, 3 years, 5 years, and the entire history) 
 * To learn more about these features, please see Section 2 of our [lab notebook](./WorldBankLabNotebook.pdf).

######Example Call
```bash
python [PATH TO FILE]/feature_loop.py -cf [contracts file with contract level features[ -if [labeled contracts file, typically output of label.py]
```

* label.py: this script merges the investigations file (which contains our modeling 'labels' - e.g. allegation outcomes (substantiated, unsubstantiated, unfounded) with our contracts file. The merging is done on the unique id: wd_id (investigations) wb contract id (contracts). A final CSV file is output which contains all contracts for which there was an allegation which either has no outcome (these are the allegations on which we will predict) or a substantiated, unfounded or unsubstantiated outcome (these will be our training data).

######Example Call
```bash
python [PATH TO FILE]/label.py' -c [contracts file with contract level features] -i [investigations file] -wf [OUTPUT FILE NAME]
```
* predict.py: this script uses our model to generate a ranked list of allegations prioritized by likelihood to be substantiated. The ranked list is output as a CSV file.

* See the entity resolution directory for scripts related to resolving company names (e.g. PWC and Price Waterhouse Coopers) using a list of canonical names from work done by the University of Cincinnati.
* data_cleaning_util.py, currency.py and sql.py: util scripts imported into the scripts above and used to clean country names, standardize currency amounts to 2015 USD and connect to a PostgreSQL database, respectively.

##Authors
Emily Grace (emily.grace.eg@gmail.com), Ankit Rai (rai5@illinois.edu), and Elissa Redmiles (eredmiles@cs.umd.edu). DSSG2015.
