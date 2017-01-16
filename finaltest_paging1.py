"""
author: Pratik Agarwal
"""
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
my_col = db.reactions_collection_foxnews
next_col = db.next_collection
user_col = db.test_collection
comm_col = db.trump_collection

#connecting to FB graph API
APP_ID = '1667170370247678'
APP_SECRET = '6829d69135dab88c93930f1a3f80b3ab'
graph = facebook.GraphAPI(APP_ID + "|" + APP_SECRET)

#setting up initial/basic query
to_scrape_lists = ["FoxNews"]
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

#individually iterating over each post to get type count
def get_reactions_data_for_post(incoming_post):
	like_count = 0
	love_count = 0
	sad_count = 0
	haha_count = 0
	angry_count = 0
	wow_count = 0
	thankful_count = 0
	j=0
	while True:
		try:
			if like_count + love_count + sad_count + haha_count + angry_count + wow_count + thankful_count > 100000:
				break
			if incoming_post["data"][j]["type"] == "LIKE":
				like_count = like_count + 1
			elif incoming_post["data"][j]["type"] == "LOVE":
				love_count = love_count + 1
			elif incoming_post["data"][j]["type"] == "SAD":
				sad_count = sad_count + 1
			elif incoming_post["data"][j]["type"] == "HAHA":
				haha_count = haha_count + 1
			elif incoming_post["data"][j]["type"] == "ANGRY":
				angry_count = angry_count + 1
			elif incoming_post["data"][j]["type"] == "WOW":
				wow_count = wow_count + 1
			elif incoming_post["data"][j]["type"] == "THANKFUL":
				thankful_count = thankful_count + 1
			j=j+1
		except IndexError:
			try:
				incoming_post = requests.get(incoming_post["paging"]["next"]).json()
				j=0
				continue;
			except KeyError:
				break;
			except ValueError:
				break;
	return (like_count,love_count,sad_count,haha_count,angry_count,wow_count,thankful_count)



#individually iterating over specific post to get user reaction data
def get_user_reaction_data_for_post(incoming_post,post_id):
	j=0
	count=0
	while True:
		try:
			id = incoming_post["data"][j]["id"]
			name = incoming_post["data"][j]["name"]
			reaction = incoming_post["data"][j]["type"]
			insert_row = user_col.insert_one(
				{
					POSTID: post_id,
					ID: id,
					NAME: name,
					REACTION: reaction,
				}
			)
			j=j+1
			count=count+1
		except IndexError:
			try:
				incoming_post = requests.get(incoming_post["paging"]["next"]).json()
				j=0
				continue;
			except KeyError:
				break;
		except KeyError:
			break;
	return count

#individually iterating over specific post to get user comment data
def get_user_comment_data_for_post(incoming_post, post_id, message):
	j=0
	comments = {}
	comments["message"] = message
	comments["id"] = post_id
	comments["comment_data"] = []
	while True:
		try:
			comments["comment_data"].append({"comment":incoming_post["data"][j]["message"], "time":incoming_post["data"][j]["created_time"], "name":incoming_post["data"][j]["from"]["name"]})
			j=j+1
		except IndexError:
			try:
				incoming_post = requests.get(incoming_post["paging"]["next"]).json()
				j=0
				continue;
			except KeyError:
				break;
		except KeyError:
			break;
	return comments

#get user id, name and corresponding reaction type for each post
def getUserReactionData(page_name):
	new_query = page_name + query

	profile = graph.get_object(new_query)
	i=0
	while True:
		try:
			get_user_reaction_data_for_post(profile["data"][i]["reactions"], profile["data"][i]["id"])
			i=i+1
		except IndexError:
			try:
				profile = requests.get(profile["paging"]["next"]).json()
				i=0
				continue
			except KeyError:
				break
		except KeyError:
			break
		except:
			break

#get userid,name,comment,created_time for each post
def getUserCommentData(page_name):
	new_query = page_name + comment_query
	profile = graph.get_object(new_query)
	i=0
	comment = {}
	index=0
	while True:
		try:
			comment[index] = get_user_comment_data_for_post(profile["data"][i]["comments"], profile["data"][i]["id"], profile["data"][i]["message"])
			comm_col.insert(comment[index])
			index+=1
			i=i+1
		except IndexError:
			try:
				profile = requests.get(profile["paging"]["next"]).json()
				i=0
				continue
			except KeyError:
				break
		except KeyError:
			i=i+1
			continue
		
#get post reactions count
def getReactionData(page_name):
	new_query = page_name + query
	new_query = new_query + "&since=2016-03-01" + "&until=2016-10-18"
	profile = graph.get_object(new_query)
	i=0
        cnt = 0
	while True:
		try:
                        print "current post %d, id:" % cnt, profile["data"][i]["id"]
			like_count, love_count, sad_count, haha_count, angry_count, wow_count, thankful_count = get_reactions_data_for_post(profile["data"][i]["reactions"])
			created_time = profile["data"][i]["created_time"]
			insert_row = my_col.insert_one(
				{
					ID: profile["data"][i]["id"],
					MESSAGE: profile["data"][i]["message"],
					LINK: profile["data"][i]["link"],
					TIME: created_time,
					LIKE: like_count,
					LOVE: love_count,
					SAD: sad_count,
					HAHA: haha_count,
					ANGRY: angry_count,
					WOW: wow_count,
					THANKFUL: thankful_count
				}
			)
			i=i+1
                        cnt += 1
		except IndexError:
			try:
				profile = requests.get(profile["paging"]["next"]).json()
				i=0
				continue
			except KeyError:
				break
		except KeyError:
			i=i+1
			continue;
	
			
def calculate_no_of_posts(incoming_post):
	new_query = "nytimes/posts/?fields=id&limit=100"
	profile = graph.get_object(new_query)
	i=0
	count = 0
	while True:
		try:
			count = count + 100
			i=i+1
		except IndexError:
			profile = requests.get(profile["paging"]["next"]).json()
			i=0
			continue
		except KeyError:
			break
	return count

#iterating through the pages supplied in to_scrape_lists
for l in range(0, len(to_scrape_lists)):
	getReactionData(to_scrape_lists[l])
	

client.close()

