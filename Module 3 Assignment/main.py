import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pydot
import scipy.spatial.distance 

fileName = 'links.csv'
df = pd.read_csv(fileName)
df.dropna()

searchQuery = ['finance','money', 'jobs','business']

targetRows = (df['target'].str.contains(searchQuery[0]) | 
              df['target'].str.contains(searchQuery[1]) |
              df['target'].str.contains(searchQuery[2]) |
              df['target'].str.contains(searchQuery[3]))

# targetRows = (df['source'].str.contains(searchQuery[0]) | df['target'].str.contains(searchQuery[0]) | 
#               df['source'].str.contains(searchQuery[1]) | df['target'].str.contains(searchQuery[1]) | 
#               df['source'].str.contains(searchQuery[2]) | df['target'].str.contains(searchQuery[2]) | 
#               df['source'].str.contains(searchQuery[3]) | df['target'].str.contains(searchQuery[3]))

# All rows related to finance, money, jobs, or business
targetdf = df[targetRows].reset_index().drop(columns=['index'])

#For each subreddit that was linked, store how they were saved
# sidebar, description, topbar, wiki, and count (total)
linkedSubreddits = {}
for index, row in targetdf.iterrows():
    if row['target'] not in linkedSubreddits:
        linkedSubreddits[row['target']] = {
            'sidebar': 0,
            'description': 0,
            'topbar': 0,
            'wiki': 0,
            'count': 0
        }

# Populating linkedSubreddits
for index, row in targetdf.iterrows():
    linkedSubreddits[row['target']][row['type']] += 1
    linkedSubreddits[row['target']]['count'] += 1

# DataFrames of each Feature (wiki, sidebar, topbar, description)
# Including Matrix for each
#Wiki
wikiRows = targetdf['type'].str.contains('wiki')
wikidf = targetdf[wikiRows].reset_index().drop(columns=['index'])
wikiMatrix = pd.crosstab(wikidf['target'], wikidf['source'])

#Sidebar
sidebarRows = targetdf['type'].str.contains('sidebar')
sidebardf = targetdf[sidebarRows].reset_index().drop(columns=['index'])
sidebarMatrix = pd.crosstab(sidebardf['target'], sidebardf['source'])

#Topbar
topbarRows = targetdf['type'].str.contains('topbar')
topbardf = targetdf[topbarRows].reset_index().drop(columns=['index'])
topbarMatrix = pd.crosstab(topbardf['target'], topbardf['source'])

#Description
descriptionRows = targetdf['type'].str.contains('description')
descriptiondf = targetdf[descriptionRows].reset_index().drop(columns=['index'])
descriptionMatrix = pd.crosstab(descriptiondf['target'], descriptiondf['source'])



#jaccard similarity measurements for Wiki
wiki_jaccard_distances = scipy.spatial.distance.cdist(wikiMatrix.values, wikiMatrix.values, metric='jaccard')
wiki_distances_df = pd.DataFrame(wiki_jaccard_distances, index=wikiMatrix.index.tolist())
wikiScores = dict()
for item in wiki_distances_df.index:
    wikiScores[item] = float(wiki_distances_df.loc[item].sum()/1000)

#Removing NSFW results from Top 10 Wiki
wikiScores.pop('blowjobsgonewild', None)
wikiScores.pop('blowjobsdaily', None)
wikiJaccardList = sorted(wikiScores.items(), key=lambda x: x[1], reverse=True)

sidebar_jaccard_distances = scipy.spatial.distance.cdist(sidebarMatrix.values, sidebarMatrix.values, metric='jaccard')
sidebar_distances_df = pd.DataFrame(sidebar_jaccard_distances, index=sidebarMatrix.index.tolist())
sidebarScores = dict()
for item in sidebar_distances_df.index:
    sidebarScores[item] = float(sidebar_distances_df.loc[item].sum()/1000)

#Removing NSFW results from Top 10 Sidebar
sidebarScores.pop('bestblowjobs', None)
sidebarScores.pop('better_blowjobs', None)
sidebarScores.pop('bikiniblowjobs', None)
sidebarScores.pop('blowjobsdaily', None)
sidebarScores.pop('blowjobsgonewild', None)
sidebarJaccardList = sorted(sidebarScores.items(), key=lambda x: x[1], reverse=True)

topbar_jaccard_distances = scipy.spatial.distance.cdist(topbarMatrix.values, topbarMatrix.values, metric='jaccard')
topbar_distances_df = pd.DataFrame(topbar_jaccard_distances, index=topbarMatrix.index.tolist())
topbarScores = dict()
for item in topbar_distances_df.index:
    topbarScores[item] = float(topbar_distances_df.loc[item].sum()/1000)

#Removing NSFW results from Top 10 Sidebar
topbarScores.pop('blowjobsdaily', None)
topbarJaccardList = sorted(topbarScores.items(), key=lambda x: x[1], reverse=True)

description_jaccard_distances = scipy.spatial.distance.cdist(descriptionMatrix.values, descriptionMatrix.values, metric='jaccard')
description_distances_df = pd.DataFrame(description_jaccard_distances, index=descriptionMatrix.index.tolist())
descriptionScores = dict()
for item in description_distances_df.index:
    descriptionScores[item] = float(description_distances_df.loc[item].sum()/1000)

#Removing NSFW results from Top 10 Description
descriptionScores.pop('blowjobs', None)
descriptionScores.pop('footjobs', None)
descriptionScores.pop('footjobsgifs', None)
descriptionScores.pop('blackgirlblowjobs', None)
descriptionScores.pop('suctionblowjobs', None)
descriptionJaccardList = sorted(descriptionScores.items(), key=lambda x: x[1], reverse=True)

print('Jaccard Similarity for Wiki')
print('Subreddit | Score')
print('-----------------')
i = 0
for subreddit, score in wikiJaccardList[:10]:
    i += 1
    print('[' + str(i) + ']',subreddit,'|', score)

print('\n\nJaccard Similarity for Sidebar')
print('Subreddit | Score')
print('-----------------')
i = 0
for subreddit, score in sidebarJaccardList[:10]:
    i += 1
    print('[' + str(i) + ']',subreddit,'|', score)

print('\n\nJaccard Similarity for Topbar')
print('Subreddit | Score')
print('-----------------')
i = 0
for subreddit, score in topbarJaccardList[:10]:
    i += 1
    print('[' + str(i) + ']',subreddit,'|', score)

print('\n\nJaccard Similarity for Description')
print('Subreddit | Score')
print('-----------------')
i = 0
for subreddit, score in descriptionJaccardList[:10]:
    i += 1
    print('[' + str(i) + ']',subreddit,'|', score)