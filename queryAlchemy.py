import json
from watson_developer_cloud import AlchemyLanguageV1
from watson_developer_cloud.watson_developer_cloud_service import WatsonException

emotion_list = ["anger", "joy", "fear", "sadness", "disgust"]
# api_key_standard = "39e7693b1e263c52f69d8f7e5e3aadb638f4aea7"
# api_key_standard = "b4210866f733cfc63081cbdc768070ab20e0282f"
# api_key_standard = "06de8298744390593ddca1ca921336abcb7b64c6"
# api_key_standard = "333e563f905a56564e28d55f53b09309353ef1a2"
# api_key_standard = "a594d0764fa642fb6cd61188e7425e72434932cd"
api_key_standard = "d5f021ed0a280cf7d1da1c83202f237f225e53ce"

def queryAlchemy(text="", url=""):

    alchemy_language = AlchemyLanguageV1(api_key=api_key_standard)
    try:
        if len(text)>0:
            result = alchemy_language.combined(
                text = text,
                extract = "doc-emotion"
            )
        elif len(url)>0:
            result = alchemy_language.combined(
                url = url,
                extract = "doc-emotion"
            )
        else:
            print "empty query"
            return [-1.0 for i in range(len(emotion_list))]
    except WatsonException, e:
        print(e.message)
        if "unsupported" not in e.message and "cannot-retrieve" not in e.message:
            print text, url
            raise e
        return [-1.0 for i in range(len(emotion_list))]
    result_emotion = result["docEmotions"]
    emotions = [0 for i in range(len(emotion_list))]
    for i in range(len(emotion_list)):
        emotions[i]= result_emotion[emotion_list[i]]
    print(emotions)
    print("-------------------------------------------------------")
    print(json.dumps(
        result,
        indent=2
    ))
    return emotions

if __name__ == "__main__":
    queryAlchemy("you are my sunshine.")

