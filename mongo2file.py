import pymongo

newspage = "nytimes_final"
filename = "./data/"+newspage+"_raw"

client = pymongo.MongoClient()
db = client["dataSet"]
collection= db["reactions_collection_"+newspage]
print collection.count()

with open(filename, "w") as f:
    for post in collection.find():
        del post["_id"]
        f.write(str(post)+"\n")

    
