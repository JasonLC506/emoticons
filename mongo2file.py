import pymongo

newspage = "wsj"
filename = "./data/"+newspage+"_raw"

client = pymongo.MongoClient()
db = client["dataSet"]
collection= db["reactions_collection_"+newspage]
print collection.count()

for post in collection.find():
    del post["_id"]
    with open(filename,"a") as f:
        f.write(str(post)+"\n")

    
