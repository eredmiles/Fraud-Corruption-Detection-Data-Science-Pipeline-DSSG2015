import psycopg2
import ConfigParser
import pandas as pd


def read_sql(table_name):

    """Read table_name into a pandas dataframe"""

    con = connect()

    #read table from database
    data = pd.read_sql_query('select * from "' + table_name + '"',con=con)

    con.close()

    return data


def write_sql(data,table_name):

    """Write a pandas data frame to the world-bank SQL database"""

    con = connect()

    #write table to database
    data.to_sql('"' + table_name + '"', con)

    con.close()

def list_tables():

    """List tables in database"""

    con = connect()

    tables = pd.read_sql_query("select tablename from pg_catalog.pg_tables where schemaname != 'pg_catalog' AND schemaname != 'information_schema'",con=con)

    con.close()

    return tables


def connect():

    """Connect to database"""

    #read password from config file
    config = ConfigParser.RawConfigParser()
    config.read('config')
    password = config.get('SQL','password')

    #open connection with database
    con = psycopg2.connect(host="dssgsummer2014postgres.c5faqozfo86k.us-west-2.rds.amazonaws.com",user='world_bank',password=password,dbname="world_bank")

    return con

def main():

    tables = list_tables()

    print tables

    data = read_sql('consolidated_cleaned')
    
    print data.columns

if __name__ == '__main__':
    main()
