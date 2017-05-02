import facebook
import requests
import pymongo
import ast
from datetime import datetime
from datetime import timedelta
import time

#connecting to FB graph API
APP_ID = '1667170370247678'
APP_SECRET = '6829d69135dab88c93930f1a3f80b3ab'
graph = facebook.GraphAPI(APP_ID + "|" + APP_SECRET)

#setting up initial/basic query
to_scrape_lists = ["FoxNews"]
reaction_query = "/?fields=id,reactions,created_time,message,link&limit=26"
comment_query = "/?fields=id,comments"
comment_query_single = "/?fields=id,comment_count,like_count,parent"

#setting up Database
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client.dataSet
col_user_reaction = db.user_reaction
col_user_comment = db.user_comment

#parameters 
THRESHOLD_REACTIONS = 1000

def test(post_id):
    new_query = post_id + reacquery
    profile = graph.get_object(new_query)
    # print profile # test #
    comments_profile = graph.get_object(post_id + comment_query)
    # print comments_profile
    comment_profile = graph.get_object("10151000206999999_10151000462854999"+comment_query_single)
    print comment_profile   
    post = {}


def getReactions(post_id):
    new_query = post_id + reaction_query
    profile = graph.get_object(new_query)["reactions"]
    cnt = 0
    while cnt <= 1000000:
        if "data" not in profile:
            print profile
            break
        for item in profile["data"]:
            reaction = {}
            reaction["POSTID"] = post_id
            reaction["READERID"] = item["id"]
            reaction["EMOTICON"] = item["type"]
            #print reaction ### test
            col_user_reaction.insert_one(reaction)
            cnt += 1
        if "paging" not in profile:
            break
        if "next" not in profile["paging"]:
            break # end of pages
        try:
            profile = requests.get(profile["paging"]["next"]).json()
        except requests.exceptions.ConnectionError, e:
            print e.message
            time.sleep(10)
            profile = requests.get(profile["paging"]["next"]).json()
    return cnt


def getComments(post_id):
    new_query = post_id + comment_query
    profile = graph.get_object(new_query)["comments"]
    cnt = 0
    while cnt <= 100000:
        if "data" not in profile:
            print profile
            break
        for item in profile["data"]:
            comment = {}
            comment["POSTID"] = post_id
            comment["READERID"] = item["from"]["id"]
            comment["TIME"] = item["created_time"]
            comment["MESSAGE"] = item["message"]
            comment["ID"] = item["id"] # comment id
            
            try:
                comment_profile = graph.get_object(comment["ID"]+comment_query_single)
            except requests.exceptions.ConnectionError, e:
                time.sleep(100)
                comment_profile = graph.get_object(comment["ID"]+comment_query_single)
            comment["LIKES"] = comment_profile["like_count"]
            comment["COMMENTS"] = comment_profile["comment_count"]
            #print comment ### test
            col_user_comment.insert_one(comment)
            cnt += 1
            time.sleep(1)
        if "paging" not in profile:
            break
        if "next" not in profile["paging"]:
            break # end of pages
        try:
            profile = requests.get(profile["paging"]["next"]).json()
        except requests.exceptions.ConnectionError, e:
            print e.message
            time.sleep(10)
            profile = requests.get(profile["paging"]["next"]).json()
    return cnt

def postsFilter(file_name, threshold = THRESHOLD_REACTIONS):
    file = open(file_name, "r")
    posts_list=[]
    for item in file.readlines():
       try:
          post = ast.literal_eval(item.rstrip())
       except SyntaxError, e:
          print e.message
          continue
       if post["feature_emotion"][0] < 0:
          continue # only valid content post
       reactions_cnt = 0
       for key in post["emoticons"]:
           reactions_cnt += post["emoticons"][key]
       if reactions_cnt < threshold: 
           continue
       if post["id"] not in posts_list:
           posts_list.append(post["id"])
       else:
          print "duplicate", post["id"]
    return posts_list

if __name__ == "__main__":
    posts_list = postsFilter("data/foxnews_Feature_linkemotion.txt")
    print "# filtered posts: ", len(posts_list)
    start_id = 240
    reaction_done = True
    for i in range(len(posts_list)):
        if i < start_id:
           continue
        post_id = posts_list[i]
        print "start collecting post", i, post_id
        print "at ", datetime.now()
        if i == start_id and reaction_done:
            pass
        else:
            start = datetime.now()
            cnt_reactions = getReactions(post_id)
            end = datetime.now()
            print "time for reactions", cnt_reactions,(end-start).total_seconds()
        start = datetime.now()
        cnt_comments = getComments(post_id)
        end = datetime.now()
        print "time for comments", cnt_comments, (end-start).total_seconds()
        
