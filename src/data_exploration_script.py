#Emily Grace & Elissa Redmiles
#DSSG 2015

import pandas as pd
import argparse
import datetime as dt

# arguments:
# -f file name - the full path and name for the file to preview
# -s the number of rows to skip at the top of the file
# -e the number of example entries of each column you want to output
# -d the columns which contain dates; provided as a list e.g. -d 17,19
# -n the additional values that should be parsed as nan; provided as a list 
# e.g. -n "#,NC" where both values will become nan when output
parser = argparse.ArgumentParser()
parser.add_argument('-f','--file_name',help='Input file with full path')
parser.add_argument('-s','--skiprows',type=int,default=0,
			help='Number of rows to skip')
parser.add_argument('-e','--num_examples',type=int,default=2,
			help='Number of examples')
parser.add_argument('-d','--datecolumns', type=str,default="",  
			help='List of columns from which to parse dates')
parser.add_argument('-n', '--nan_additions', type=str, default="", 
			help='List of nans to add to exclusions')
args = parser.parse_args()

# parse argument lists for dates and nans
if args.datecolumns is not "":
    dates = [int(item) for item in args.datecolumns.split(',')]
else:
    dates=[]
if args.nan_additions is not "":
    nans = [str(item) for item in args.nan_additions.split(',')]
else:
    nans=[]

input_file = args.file_name

# read the file
# take in the input file with full path, skip given number of rows
# add additional values to consider as nan
# give particular columns to consider as dates
# reduce memory usage by not guessing column types
if '.csv' in input_file:
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
# process dates
if dates is not "":
    for index in dates:
        if not(df.dtypes[index] in ['datetime64','datetime64[ns]']):
             try:
                 df.ix[:,index] = pd.to_datetime(df.ix[:,index], 
						format='%m/%d/%Y %H:%M:%S %p')
             except:
                 try:
                     df.ix[:,index] = pd.to_datetime(df.ix[:,index], 
						format='%m/%d/%Y')
                 except:
                     print "error " + str(index)

# output section
print 'Columns: ',df.columns
print 'type: ', df.dtypes
print 'length: ',len(df.index)

# Exploring each column
for idx,col in enumerate(df.columns):

   print 'COL ',col
   try: 
       print 'Unique: ',df[col].nunique()
       print 'Example: ',df[col].unique()[0:min(args.num_examples,
						len(df[col].unique()))]
       print 'NaNs:',float(len(df[col][df[col].isnull()]))/float(len(df[col]))
   except:
       print ("Wrong type, cannot calculate number of unique values, output an example" 
		" or print nans. Type is: %s",type(df[col]))
   if df.dtypes[idx] in ['int16', 'int32', 'int64', 'float16', 'float32',
			 'float64']:
      print 'Min:',min(df[col])
      print 'Max: ',max(df[col])
      print 'Mean: ',df[col].mean()
      print 'Std: ',df[col].std()
      print 'Median: ',df[col].median()

   if df.dtypes[idx] in ['datetime64','datetime64[ns]']:
      print 'Range:',min(df[col]),'-',max(df[col])	
   
