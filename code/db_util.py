"""
Author: Logan Blue
Date: March 3, 2020

This class will encapsilate the database of choice (currently mongodb) for the acoustic
work. It will support loading, querying, and insertion into the database.
"""
#pylint: disable=trailing-whitespace, invalid-name, dangerous-default-value

import pymongo

class DBObj:
    """This object wraps the interface to our database. Used for future flexibility is 
    we need to change the underlying database. """

    def __init__(self, db_name='exploration', collection_name='timit_train'):
        """Constructor for db obj, will connect to default port and to a 
        specified db and collection. By default the db = 'exploration' and 
        collection = 'timit_train'
        """
        db_client = pymongo.MongoClient()               #connect to mongo
        self.db = db_client[db_name]                    #connect to correct db
        self.table = self.db[collection_name]           #access correct collection
        #print("DB successfully connected to...")

    def insert(self, data):
        """This function will insert records into the table. 

        data - is assumed to be a pandas dataframe. 
        """
        if isinstance(data, dict):
            self.__insert_single(data)
        else:
            self.__insert_multi(data)

    def __insert_single(self, data):
        """This function will insert records into the table. 

        data - is assumed to be a pandas dataframe. 
        """
        #insert into database
        self.table.insert_one(data)

    def __insert_multi(self, data):
        """This function will insert records into the table. 

        data - is assumed to be a pandas dataframe. 
        """
        #convert data into a list of dictionaries
        insertable_data = []
        for _, row in data.iterrows():
            insertable_data.append(row.to_dict())

        #insert into database
        self.table.insertMany(insertable_data)

    def query(self, filters={}):
        """This function wraps the query/search functionality of the db. 
        The filters will be a dictionary of the column name, the condition, and 
        the operation that relates the column and the condition. By default, if 
        filters are not provided, this function will return the whole collection. 

        We expect the filters input to be in the mongo db style of querying. 
        This is done to simply the early development of the tool, if later versions
        of the tool require a different DB backend we will translate mongoDB style
        queries to the new databases standards here. 
        """
        return self.table.find(filters) 
