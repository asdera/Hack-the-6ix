import pandas as pd
import boto3
from math import *
from tabulate import tabulate
import json
from pprint import pprint

flexibility = 0.23789994

def ppDF(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

def ppS(df):
    print(tabulate(df.to_frame(name="Values"), headers='keys', tablefmt='psql'))

with open('./people.json') as jsonFile:
    db = json.load(jsonFile)
P = db["People"]

with open('./entities.json') as data_file:    
    data = json.load(data_file)
E = pd.DataFrame(data['Entities']).drop(['BeginOffset', 'EndOffset'], axis=1)

with open('./keyphrases.json') as data_file:    
    data = json.load(data_file)
K = pd.DataFrame(data['KeyPhrases']).drop(['BeginOffset', 'EndOffset'], axis=1)

with open('./sentiment.json') as data_file:    
    data = json.load(data_file)
S = pd.Series(data['Sentiment']['SentimentScore'])

ppDF(E)

ppDF(K)

ppS(S)

def removeP(x):
    P.pop(x)

def clearP(x):
    del P[x]['Entities'][:]
    del P[x]['KeyPhrases'][:]
    P[x]['Sentiment'] = {k: 0.25 for k, v in P[x]['Sentiment'].items()}

def addP(name):
    P.append(
        {"Name": name, "Entities": [], "KeyPhrases": [], "Sentiment": {"Positive": 0.25, "Negative": 0.25, "Neutral": 0.25, "Mixed": 0.25}}
    )

def updateP(x):
    P[x]['Entities'] += E.to_dict('records')
    P[x]['KeyPhrases'] += K.to_dict('records')
    P[x]['Sentiment'] = {k: v*flexibility+S.get(k)*(1-flexibility) for k, v in P[x]['Sentiment'].items()}


def match(x, y):
    personality = 10-(sum([abs(v-P[y]['Sentiment'][k]) for k, v in P[x]['Sentiment'].items()])/2)*10
    score = 0
    for a in P[x]['Entities']:
        for b in P[y]['Entities']:
            if a['Text'] == b['Text']:
                score += a.Score * b.Score
    if len(P[x]['Entities'])*len(P[y]['Entities']) > 0 and score > 0:
        goals = log10(score/(len(P[x]['Entities'])*len(P[y]['Entities'])))+10
    else:
        goals = 0

    for a in P[x]['KeyPhrases']:
        for b in P[y]['KeyPhrases']:
            if a['Text'] == b['Text']:
                score += a.Score * b.Score
    if len(P[x]['KeyPhrases'])*len(P[y]['KeyPhrases']) > 0 and score > 0:
        interests = log10(score/(len(P[x]['KeyPhrases'])*len(P[y]['KeyPhrases'])))+10
    else:
        interests = 0


match(0, 1)
match(2, 3)

with open("./people.json", "w") as jsonFile:
    json.dump(db, jsonFile)

# comprehend = boto3.client(service_name='comprehend', region_name='region')
                
# text = "It is raining today in Seattle"

# print('Calling DetectKeyPhrases')
# print(json.dumps(comprehend.detect_key_phrases(Text=text, LanguageCode='en'), sort_keys=True, indent=4))
# print('End of DetectKeyPhrases\n')