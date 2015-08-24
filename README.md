#SETUP INSTRUCTIONS
- Git clone https://github.com/eredmiles/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015.git

You will run full_pipeline.sh to do EVERYTHING!

- Install Anaconda python (see instructions [here](http://docs.continuum.io/anaconda/install#linux-install)
NOTE: Install Python 2.7 NOT Python 3.X
- Setup PostgreSQL database:
	1. sudo apt-get install postgresql postgresql-contrib
	2. Install according to [these instructions](https://help.ubuntu.com/community/PostgreSQL#Alternative_Server_Setup)

- Install pandas - conda install pandas
- Install seaborn - conda install seaborn
- Install csvkit - conda install csvkit
- Install psycopg2 - conda install psycopg2

###Required Files
- The investigations file must be saved as a .csv file. Further, it must be named investigations.csv
- Download the zip file from here: http://data.worldbank.org/indicator/PA.NUS.PPP, save the ppp file within as ppp.csv in pipeline_data (or your DATA_STORAGE variable location, see below). Do the same for the zip file from here: http://data.worldbank.org/indicator/PA.NUS.FCRF, save the fcrf file as fcrf.csv


###Modify full_pipeline.sh as follows:
- The LOCALPATH variable should be the path to the WorldBank2015 directory that you cloned from git (e.g. /dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015)
- The DATA_STORAGE variable should be the path where your data files AND THE INVESTIGATION FILE will live. there is already a directory called pipeline data in the github repo that contains the entity resolution canonical. You should use this directory (e.g. /dssg/Fraud-Corruption-Detection-Data-Science-Pipeline-DSSG2015/pipeline_data)

NOTE: This is also where all output files will be stored.
- The CURRENCY_FILE_PPP variable should contain the full path where you want ppp.csv (a currency conversion file) to live.
- THE CURRENCY_FILE_FCRF variable should contain the full path where you want fcrf.csv (a currency conversion file) to live.

- If you are no longer using the local host database:
    - you must change the database connection in sql.py (Line 54), supplier_feature_gen (line 42)
   Example:    host="localhost",user="dssg",password=password,dbname="world_bank"
   copy example_config to config (e.g. cp example_config config), modify config to have the password for the database (e.g. the system password for your user, in our example the user password for dssg)
    - you must also modify model_pipeline_script.py (line 86) and supplier_feature_gen.py (line 217): create_engine(r'postgresql://[USER_NAME]:'+password+'localhost/DATABASE'. e.g. create_engine(r'postgresql://dssg:'+password+'localhost/world_bank'
    - you must create a config file in the directory from which you will run the script.



##Author
Elissa Redmiles (eredmiles@cs.umd.edu). DSSG2015.
