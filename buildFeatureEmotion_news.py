import queryAlchemy
import ast
from logRegFeatureEmotion import emoticon_list

logfile = "log_buildFeatureEmotion_new"

def buildFeatureEmotion(postfile, resultfile):

    Nposts_token = 1004

    Nposts = 0
    N_posts_valid = 0
    postfile = open(postfile,"r")

    # file_result = open(resultfile,"r")
    # posts_collected = file_result.readlines()
    # file_result.close()

    for item in postfile.readlines():
        post = ast.literal_eval(item.rstrip())
        Nposts += 1
        if Nposts <= Nposts_token:
            continue
        if not validCheck(post):
            continue
        N_posts_valid += 1

        post_item = {}
        post_item["id"] = post["id"]
        post_item["emoticons"] = {}

        # emoticons #
        keys = post.keys()
        for emoticon in emoticon_list:
            emoti_upper = emoticon.upper()
            if emoti_upper in keys:
                post_item["emoticons"][emoticon]= post[emoti_upper]
            else:
                post_item["emoticons"][emoticon] = 0

        # emotion feature #
        try:
            post_item["feature_emotion"] = queryAlchemy.queryAlchemy(url=post["LINK"])
            with open(logfile,"a") as log:
                log.write("%d post succeed\n" % Nposts)
        except:
            with open(logfile,"a") as log:
                log.write("%d post fail\n" % Nposts)
            break

        # write to file #
        with open(resultfile,"a") as f:
            f.write(str(post_item) + "\n")

    postfile.close()

def validCheck(post):
    # currently only take posts with valid url link #
    if len(post["LINK"])>1:
        return True
    else:
        return False

if __name__ == "__main__":
    postfile = "data/atlantic_raw"
    resultfile = "data/atlantic_Feature_linkemotion.txt"
    buildFeatureEmotion(postfile=postfile, resultfile=resultfile)