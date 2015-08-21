#Elissa Redmiles
#DSSG2015

import pandas as pd
import argparse
import datetime as dt
import numpy
#import data_cleaning_util as util
#arguments section
# inputs are:
# -f name of file to clean
# -s to skip that number of rows for all files
# -d column numbers of columns that contain dates
# -n values that should be considered nans
# -x columns to drop
# -r columns to rename, give list [oldname],[newname],[oldname],[newname]
# for example Unnamed: 5,Project Name to rename the column Unnamed: 5 to Project Name
#-o file to output, NO extension, FULL path

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file_name',help='Input file')
parser.add_argument('-s','--skiprows',type=int,default=0,help='Number of rows to skip')
parser.add_argument('-d','--datecolumns', type=str,default="", help='Indices of columns from which to parse dates')
parser.add_argument('-n', '--nan_additions', type=str, default="", help='List of nans to add to exclusions')
parser.add_argument('-x', '--exclude', type=str, default="", help ='List of columns to drop')
parser.add_argument('-r', '--rename', type=str, default="", help='pairs of columns to rename and new name')
parser.add_argument('-o', '--output_file', type=str, default="out_cleaned.csv", help='Name of output file (without extension) with full path')
args = parser.parse_args()

#parse argument lists
if args.datecolumns is not "":
    dates = [int(item) for item in args.datecolumns.split(',')]
else:
    dates=[]
if args.nan_additions is not "":
    nans = [str(item) for item in args.nan_additions.split(',')]
else:
    nans=[]
if args.exclude is not "":
    excols=[int(item) for item in args.exclude.split(',')]
else:
    excols=[]
if args.rename is not "":
    renames=[str(item) for item in args.rename.split(',')]
else:
    renames=[]
input_file = args.file_name

# read the file
# take in the input file with full path, skip given number of rows
# add additional values to consider as nan
# give particular columns to consider as dates
# reduce memory usage by not guessing column types
if '.csv' in input_file:
    print 'csv'
    try:
        df=pd.read_csv(input_file, skiprows=args.skiprows, na_values=nans, 
			parse_dates=dates, infer_datetime_format=True, 
			low_memory=False)
    except:
        df=pd.read_csv(input_file, skiprows=args.skiprows, na_values=nans,
                        parse_dates=dates, infer_datetime_format=True)
if '.xls' in input_file:
    try:
        df = pd.read_excel(input_file, skiprows=args.skiprows, na_values=nans, 
			parse_dates=dates, infer_datetime_format=True, 
			low_memory=False)
    except:
         df = pd.read_excel(input_file, skiprows=args.skiprows, na_values=nans,
                        parse_dates=dates, infer_datetime_format=True)
#print len(df.index)
#print df.shape
#remove duplicates
df = df.drop_duplicates()
#print len(df.index)
#print df.shape

#drop columns indicated
df.drop(df.columns[excols], axis=1, inplace=True) # Note: zero indexed
i=0
#print len(df.index)
#print df.shape
while i in range(len(renames)-1):

    oldname = renames[i]
    newname = renames[i+1]
    print "oldname: " + oldname + " newname: " + newname
    df = df.rename(columns={oldname: newname})
    i=i+2   
#print len(df.index)
#print df.shape
#change column names
df=df.rename(columns=lambda x: x.replace(" ", "_").lower())
print df.columns
print len(df.index)
#print df.shape
#change types
df.convert_objects(convert_numeric=True).dtypes
#print len(df.index)
#print df.shape

#process dates
if dates is not "":
    for index in dates:
      # df = df[df.ix[:,index].str.contains("/|-")==True]    
       if not(df.dtypes[index] in ['datetime64','datetime64[ns]']):
              try:
                  df.ix[:,index] = pd.to_datetime(df.ix[:,index], 
					errors='raise',coerce=True,
					format='%m/%d/%Y %H:%M:%S %p')
              #print len(df.index)
	      #print df.shape
	      except:
                  try:
                      df.ix[:,index] = pd.to_datetime(df.ix[:,index], 
					errors='raise', coerce=True, 
					format='%m/%d/%Y')                
                  except:
                      try:
                          df.ix[:,index]=pd.to_datetime(df.ix[:,index], 
					errors='raise', coerce=True, 
					format='%Y-%m-%d')
                      except:
	                  df.ix[:,index]=pd.to_datetime(df.ix[:,index],
					errors='raise', coerce=True,
					 format='%m-%d-%Y')
#df=util.fix_country_names(df)   
filename=args.output_file
df.to_csv(filename)#, encoding='utf-8') 


