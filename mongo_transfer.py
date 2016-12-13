import pymongo
# open tunnel using command #
""" 
ssh jiasheng@191.234.186.42 -L 27018:localhost:27017
"""

db_name = "dataSet"
collection_names = []

client = pymongo.MongoClient()
db = client[db_name]
for collection_name in collection_names:
    db.command("cloneCollection", **{"cloneCollection":db_name+collection_name,"collection":db_name+"."+collection_name, "from":"localhost:27018"})
    print db[collection_name].count()

