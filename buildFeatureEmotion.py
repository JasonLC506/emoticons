import queryAlchemy
import ast

def buildFeatureEmotion(postfile, resultfile):

    Ntext_posts = 0
    Ntext_posts_eng = 2909
    Ntext_posts_token = 5291
    postfile = open(postfile,"r")


    for item in postfile.readlines():
        poster_post = ast.literal_eval(item)
        poster = poster_post.keys()[0]
        posts = poster_post[poster]
        for post in posts:
            post_item = {}
            post_item["text"] = post["text"]
            post_item["emoticons"] = post["emoticons"]
            post_item["author"] = poster
            if validCheck(post_item):
                Ntext_posts += 1
                if Ntext_posts<=Ntext_posts_token:
                    continue
                post_item["feature_emotion"]=queryAlchemy.queryAlchemy(post_item["text"])
                if post_item["feature_emotion"][0]>=0:
                    Ntext_posts_eng += 1
                print("Ntext_posts: ", Ntext_posts,)
                print("Ntext_posts_eng: ", Ntext_posts_eng)

                #post_list.append(post_item)
                file = open(resultfile, "a")
                file.write(str(post_item) + "\n")
                file.close()


    postfile.close()


def validCheck(post_item):
    if len(post_item["text"])>=1 and len(post_item["emoticons"])>=1:
        return True
    else:
        return False

if __name__ == "__main__":
    postfile = "C:/Users/zjs50/OneDrive/Documents/data/posts_0.txt"
    resultfile = "C:/Users/zjs50/OneDrive/Documents/data/posts_Feature_Emotion.txt"
    buildFeatureEmotion(postfile=postfile, resultfile= resultfile)