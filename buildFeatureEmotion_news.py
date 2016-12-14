import queryAlchemy
import ast
from logRegFeatureEmotion import emoticon_list

def buildFeatureEmotion(postfile, resultfile):
    
    Nposts = 0
    N_posts_valid = 0
    Ntext_posts_token = 0
    postfile = open(postfile,"r")

    file_result = open(resultfile,"r")
    posts_collected = file_result.readlines()
    file_result.close()

    for item in postfile.readlines():
        post = ast.literal_eval(item.rstrip())
        post_item = {}
        post_item["text"] = post["MESSAGE"]
        post_item["link"] = post["LINK"]
        post_item["emoticons"] = {}

        # emoticons #

