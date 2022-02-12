"""This file simply read in a mongodb and exports it as a csv

Auhor: Logan Blue
"""

import pymongo
import pandas as pd

collection_name = 'mp_test1'

#connect to mongoDB
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["exploration"]

#get table
table = db[collection_name]
cursor = table.find()

#convert to df
df = pd.DataFrame(list(cursor))

df.to_pickle(collection_name + '.pkl')
