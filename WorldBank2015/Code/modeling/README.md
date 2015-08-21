#Modeling pipeline
This directory contains the python script for running the entire modeling pipeline.

## Directory Contents
This directory contains all the required python script to build and test predictive models (multiple machine learning models):


**model_pipeline_script.py** - This is source code for building model on temporal train/test split

**compare_models.py** - Plots the AUC, precision, recall for top x% cases - different models for their performance evaluation.

**decision_surface_plot.ipynb** - Ipython notebook with functionality to plot decision boundary of model and feature response curves.

**modelpipeline.ipynb** - Ipython notebook with some initial modeling pipeline

**predict.py** - Script to generate final ranked list of cases based on their probability score.

**prediction_with_communicate.py** - script to generate list of pair of features with their relative score of being occuring as root node and first child in building decision tree.

**prediction_loop.py** - Script to generate prediction score for different allegation_category type.


##Authors 
Emily Grace (emily.grace.eg@gmail.com), Ankit Rai (rai5@illinois.edu), and Elissa Redmiles (eredmiles@cs.umd.edu). DSSG2015.
