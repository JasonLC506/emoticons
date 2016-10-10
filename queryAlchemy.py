import json
from watson_developer_cloud import AlchemyLanguageV1
from watson_developer_cloud.watson_developer_cloud_service import WatsonException

emotion_list = ["anger", "joy", "fear", "sadness", "disgust"]

def queryAlchemy(text):

    alchemy_language = AlchemyLanguageV1(api_key="39e7693b1e263c52f69d8f7e5e3aadb638f4aea7")
    try:
        result = alchemy_language.combined(
            text = text,
            extract = "doc-emotion"
        )
    except WatsonException, e:
        print(e.message)
        if "unsupported" not in e.message:
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

