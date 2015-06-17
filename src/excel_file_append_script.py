#Elissa Redmiles
#DSSG2015
#script for merging excel files of same type of data together

import os
import pandas as pd
import argparse
import re
#by default new, appended, file is output as _consolidated.csv
def main():
    # arguments section
    # inputs are:
    # -d directory name
    # -sl list of rows to skip for each file in order
    # e.g. if you have file a.xls and b.xls in the directory, you would do 
    # -sl 1,2 to skip 1 row for a.xls and 2 rows for b.xls
    # -s to skip that number of rows for all files
 
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_name', help='directory name')
    parser.add_argument('-sl', '--skiprowlist', type=str,default="", help='List of rows to skip for each file')
    parser.add_argument('-s', '--skiprows', type=int, default=0, help='Number of rows to skip')
    args = parser.parse_args()
    # parse argument list
    if args.skiprowlist is not "":
        rows = [int(item) for item in args.skiprowlist.split(',')]
    
    df = pd.DataFrame()
    # get the files in the directory and sort them alphabetically
    # including numerical order
    files_in_dir = os.listdir(args.dir_name+"/")
    files_in_dir.sort(key=natural_keys)
    print files_in_dir
    
    #read in and append files
    cnt=0
    for file_in_dir in files_in_dir:
        print "in loop"
        if '.xlsx' in file_in_dir:
            if (not('consolidated' in file_in_dir)):
	        print file_in_dir
                print rows[cnt]
                data = pd.read_excel(args.dir_name + "/"+ file_in_dir, skiprows=rows[cnt])
	        cnt=cnt+1
                print data.columns
                df = df.append(data)
                
    df.to_csv(dir_name+"/_consolidated.csv", encoding='utf-8')

#natural human string sorting from stack overflow
# http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
main()
