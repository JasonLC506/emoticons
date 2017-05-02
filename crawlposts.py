import facebook
import json
import datetime
import csv
import urllib2
import time
import requests
from pprint import pprint
import unicodedata
from pymongo import MongoClient

#setting up DataBase
client = MongoClient("mongodb://localhost:27017")
db = client.dataSet
my_col = db.reactions_collection_nytimes_early_id # dependent on page name #
next_col = db.next_collection
user_col = db.test_collection
comm_col = db.trump_collection

#connecting to FB graph API
APP_ID = '1522702467762176'
APP_SECRET = '39846d105e1a79091bdecb16b888d28d'
graph = facebook.GraphAPI(APP_ID + "|" + APP_SECRET)

#setting up initial/basic query
to_scrape_lists = ["nytimes"]
idquery = "/posts/?fields=id,created_time"
query = "/posts/?fields=id,reactions,created_time,message,link&limit=26"
comment_query = "/posts/?fields=id,message,comments&limit=26"

#setting up parameters for inserting to mongo db (just for ease of use)
ID = "id"
LIKE = "LIKE"
LOVE = "LOVE"
SAD = "SAD"
HAHA = "HAHA"
ANGRY = "ANGRY"
WOW = "WOW"
THANKFUL = "THANKFUL"
NAME = "NAME"
REACTION = "TYPE"
POSTID = "POSTID"
COMMENT = "COMMENT"
TIME = "TIME"
MESSAGE = "MESSAGE"
LINK = "LINK"

datetimeformat = "%Y-%m-%dT%H:%M:%S+0000"
timeUntil = time.strptime("2016-01-01T00:00:00+0000", datetimeformat)

log_file = "logs/log_crawl_early_id"

def getPostID(page_name, max_stored = 100000):
    new_query = page_name + idquery
    profile = graph.get_object(new_query)
    cnt = 0
    cnt_stored = 0
    while cnt_stored < max_stored:
        posts = profile["data"]
        for post in posts:
            ctime = time.strptime(post["created_time"], datetimeformat)
            if ctime < timeUntil:
                my_col.insert_one(
                    {
                        ID: post["id"],
                        TIME: post["created_time"]
                    }
                )
                cnt_stored += 1
            cnt += 1
        with open(log_file, "a") as f:
            f.write("%d in %d\n" % (cnt_stored, cnt))
        if "paging" in profile.keys():
            if "next" in profile["paging"].keys():
                profile = requests.get(profile["paging"]["next"])
            else:
                with open(log_file, "a") as f:
                    f.write("\n! no next page !\n")
                break
        else:
            print profile
            raise IndexError("no paging in profile")


if __name__ == "__main__":
    for page_name in to_scrape_lists:
        getPostID(page_name)
