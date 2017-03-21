import facebook
import requests

#connecting to FB graph API
APP_ID = '1667170370247678'
APP_SECRET = '6829d69135dab88c93930f1a3f80b3ab'
graph = facebook.GraphAPI(APP_ID + "|" + APP_SECRET)

#setting up initial/basic query
to_scrape_lists = ["FoxNews"]
query = "/posts/?fields=id,reactions,created_time,message,link"
query_single = "/?fields=id,reactions,created_time,message,link&limit=26"
comment_query = "/posts/?fields=id,message,comments&limit=26"

def getReactionData(post_id):
    new_query = post_id + query
    profile = graph.get_object(new_query)
    print profile # test #
    post = {}
    post["ID"] = profile["data"]["id"]

if __name__ == "__main__":
    getReactionData("5281959998_10151000206999999")