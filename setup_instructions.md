#SETUP INSTRUCTIONS
- Git clone [url]
You will run full_pipeline.sh to do EVERYTHING!

- Install Anaconda python (see instructions [here](http://docs.continuum.io/anaconda/install#linux-install)
- Setup PostgreSQL database:
	1. sudo apt-get install postgresql postgresql-contrib
	2. Install according to [these instructions](https://help.ubuntu.com/community/PostgreSQL#Alternative_Server_Setup)

- Install pandas - conda install pandas
- Install seaborn - conda install seaborn

- If you are no longer using the local host database:
    - you must change the database connection in sql.py and model_pipeline_script.py
    - you must create a config file in the directory from which you will run the script.

##Author
Elissa Redmiles (eredmiles@cs.umd.edu). DSSG2015.
