import json
from datetime import datetime
import time
import pandas
import googlemaps
YOUR_API_KEY = 'AIzaSyAiFpFd85eMtfbvmVNEYuNds5TEF9FjIPI'

gmaps = googlemaps.Client(key='AIzaSyCEJPcVVU9bBh_f757noMSJbz0WsF7joc4')


def predict(types):
    for type in types:
        if type=="library":
            return "Studying in library"
        elif type=="convenience_store":
            return "Convenience Store Shoping"
        elif type=="restaurant":
            return "Having Lunch "
        elif type=="book_store":
            return "Buying Books"


def nearby(location):
    nearby_places = gmaps.places_nearby(location, radius=55)
    #print(nearby_places)
    names=[]
    for p in nearby_places["results"]:
        if "Louisana" in p["name"]:
            pass
        else:
            names.append(p["name"])
            if(len(names)==2):
                pred=predict(p["types"])
        if(len(names)==3):
            break
    return names,pred

def pathURL(locations,Action):
    if Action=='running':
        output = "https://maps.googleapis.com/maps/api/staticmap?size=600x300&key=AIzaSyCEJPcVVU9bBh_f757noMSJbz0WsF7joc4&language=en&path=color:0xd48431%7Cweight:6"
    else:
        output = "https://maps.googleapis.com/maps/api/staticmap?size=600x300&key=AIzaSyCEJPcVVU9bBh_f757noMSJbz0WsF7joc4&language=en&path=color:0x7931d4%7Cweight:6"
    for l in locations:
        output+="%7C"
        output=output+str(l["lat"])+","+str(l["lng"])
    return output

with open('../outputfileNov30', 'r') as f:
    lines=f.readlines()
    lines=json.loads(lines[0])
    print(len(lines))

    #d2 = {k: f(v) for k, v in lines.items()}
    #print(lines)
    d = [{"Timestamp": datetime.fromtimestamp(time.mktime(datetime.strptime(l['Timestamp'], "%Y-%m-%d %H:%M:%S -%f").timetuple())),
          'Action':l['Action'],
          'lng':float(l['Longtitude']),
          "lat":float(l['Latitude'])} for l in lines]

    d = sorted(d, key=lambda k:k['Timestamp'])
    location= [{'lat':item['lat'],'lng':item['lng']} for item in d]

    outputList=[]
    temp={}
    for i in range(len(d)):
        if temp=={}:
            temp={'Action':d[i]['Action'],
                  'Starttime':d[i]['Timestamp'],
                  'Endtime':d[i]['Timestamp'],
                  'Location':[{'lat':d[i]['lat'],
                               'lng':d[i]['lng']}]}
        else:
            #assume that action cannot different every second
            temp["Location"].append({'lat': d[i]['lat'], 'lng': d[i]['lng']})
            temp["Endtime"]=d[i]['Timestamp']
            if i==len(d)-1:
                outputList.append(temp)
                continue
            if d[i+1]['Action']!=temp['Action']:
                outputList.append(temp)
                temp={}

    # strftime("%Y-%m-%d %H:%M:%S")
    # strftime("%Y-%m-%d %H:%M:%S")
    #

    for i in range(len(outputList)):
        outputList[i]["duration"]=outputList[i]["Endtime"]-outputList[i]["Starttime"]
        outputList[i]["durationm"]=int(outputList[i]["duration"].seconds/60)
        outputList[i]["durations"] = int(outputList[i]["duration"].seconds % 60)
        outputList[i]["duration"]=0

        outputList[i]["Starttime"]=outputList[i]["Starttime"].strftime("%Y-%m-%d %H:%M:%S")
        outputList[i]["Endtime"]=outputList[i]["Endtime"].strftime("%Y-%m-%d %H:%M:%S")
        if outputList[i]["Action"]=="staying_still":
            outputList[i]["pred"]="   "
            outputList[i]["nearby"],outputList[i]["pred"]=nearby(outputList[i]["Location"][int(len(outputList[i]["Location"])/2)])
            if(outputList[i]["durationm"]<10):
                outputList[i]["pred"]="   "
            outputList[i]["url"]="https://maps.googleapis.com/maps/api/staticmap?zoom=17&size=600x300&language=en&maptype=roadmap&markers=color:blue%7Clabel:%7C"\
                                 +str(outputList[i]["Location"][int(len(outputList[i]["Location"])/2)]["lat"])+","+str(outputList[i]["Location"][int(len(outputList[i]["Location"])/2)]["lng"])+"&key=AIzaSyCEJPcVVU9bBh_f757noMSJbz0WsF7joc4"
            # outputList[i]["Location"]=[outputList[i]["Location"][0]]
        elif outputList[i]["Action"]=="walking" or outputList[i]["Action"]=="running":
            url="https://maps.googleapis.com/maps/api/staticmap?path=color:0x0000ff%7Cweight:5%7C40.737102,-73.990318%7C&size=600x300&key=AIzaSyCEJPcVVU9bBh_f757noMSJbz0WsF7joc4"
            #print(outputList[i])
            if(len(outputList[i]["Location"])>100):
                parts=[]
                for j in range(101):
                    proportion=int(j/100*(len(outputList[i]["Location"])-1))
                    parts.append(outputList[i]["Location"][proportion])
                outputList[i]["url"] = pathURL(parts,outputList[i]["Action"])
            else:
                outputList[i]["url"] = pathURL(outputList[i]["Location"],outputList[i]["Action"])

            print(outputList[i]["url"])

    # for i in range(len(outputList)):
    #


    # print(outputList)

    with open('../testfileNov30', 'w') as fout:
        json.dump(outputList, fout)


        #print(lines)
