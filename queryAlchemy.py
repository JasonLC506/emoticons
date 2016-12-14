import json
from watson_developer_cloud import AlchemyLanguageV1
from watson_developer_cloud.watson_developer_cloud_service import WatsonException

emotion_list = ["anger", "joy", "fear", "sadness", "disgust"]
api_key_standard = "39e7693b1e263c52f69d8f7e5e3aadb638f4aea7"

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
        if "unsupported" not in e.message:
            print text
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

