#Entity Resolution
The world bank data contains 74000 unique entity (supplier/company) names. Of these, many are actually the same supplier (e.g. PWC and Price Waterhouse Coopers). We used a canonical entity resolution list to resolve these entities to the same name.

The canonical list was provided by the entity resolution team at the University of Cincinatti lead by Eric Rozier.

##Directory Contents
This directory contains the following files:
- entity_resolution.py and entity_resolution_syntactic.py: scripts used to resolve entities in the contracts data by using canonical list to match and replace the entities in our contracts file. The script generates a new contracts file/data frame which has the resolved entities in the 'supplier' field and the original entity names in the 'orig_supplier_name' field.
- entity_resolution.sh: script for testing different versions of the canonical list and different matching techniques (syntactic vs. semantic) and comparing number of entities resolved after each test.

######Example Call
```bash
python [path to data_pipeline_src directory]/entity_resolution/entity_resolution.py -c [contracts data file] -e [entity canonical file] -o [name of contracts file with resolved entities]
```

###Authors
Emily Grace (emily.grace.eg@gmail.com) and Elissa Redmiles (eredmiles@cs.umd.edu). DSSG2015.
