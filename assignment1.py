# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import glob
import re
from os import listdir
import os
import seaborn as sns
from nltk.stem.porter import *
import nltk
import math
from sklearn.feature_extraction.text import TfidfTransformer 
from numpy import dot
from numpy.linalg import norm
import csv

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

#Download dependencies
nltk.download('punkt')
nltk.download('stopwords')

def task1():
    f = open(datafilepath, "r")
    data_dic = json.loads(f.read())
    data_sort = sorted(data_dic["teams_codes"])
    return data_sort
    
def task2():
    #Open and read data.json
    f = open(datafilepath, "r")
    data_dic = json.loads(f.read())
    #Create panda series, "clubs" as data, "teams_codes" as index
    clubs = data_dic["clubs"]
    #Initialize two dictionaries for dataframe
    goals_by_team = {}
    goals_against_team = {}
    for i in clubs:
        goals_by_team[i["name"]] = i["goals_scored"]
        goals_against_team[i["name"]] = i["goals_conceded"]
    #Organize into one dataframe
    byteam_df = pd.Series(goals_by_team)
    againstteam_df = pd.Series(goals_against_team)
    team_df = pd.DataFrame({'goals_scored_byteam':byteam_df,'goals_scored_againstteam':againstteam_df})
    team_df.index = data_dic["teams_codes"]
    sort_team_df = team_df.sort_index()
    return sort_team_df.to_csv('task2.csv')
      
def task3():
    #Initializiton
    rep_ser = ()
    dic = {}
    #Extract total goals in each report
    for i in listdir(articlespath + "/"):
        if i.endswith('.txt'):
            f = open(articlespath + "/" + i, "r")
            rep = f.read()
            match = re.findall(" ([0-9]{1,2})-([0-9]{1,2})\D", rep)
            dic[i] = 0        
            if match != []:
                # Add up the two number catched by groups in each tuple, return the largest result
                for j in match:
                    temp = int(j[0]) + int(j[1])
                    dic[i] = max(temp, dic[i])
            else:
                continue
    # Construst a pandas series and sort
    rep_ser = pd.Series(dic)
    rep_ser.name = "total_goals"
    rep_ser_sort = rep_ser.sort_index()
    return rep_ser_sort.to_csv('task3.csv', index_label = "filename")

def task4():
    df = pd.read_csv('task3.csv')
    df_sorted = df.sort_values(by=['total_goals'])
    total_goals = df_sorted['total_goals']
    #Create the canvas for boxplot
    fig, ax = plt.subplots()
    ax.set_title('Max total goals in each report')
    ax.set_ylabel('Goals Num')
    #Highlight the outliers
    green_diamond = dict(markerfacecolor='g', marker='D')
    ax.boxplot(total_goals, flierprops=green_diamond)
    return fig.savefig('task4.png')

def task5():
    #Open and read data.json
    f = open(datafilepath, "r")
    data_dic = json.loads(f.read())
    club_name = data_dic["participating_clubs"]
    #Panda Dataframe initialization
    df = pd.DataFrame(index = club_name, data = [0] * len(club_name), columns = ['number_of_mentions'])
    #Add pattern matched times to the dataframe iterately
    for i in listdir(articlespath + "/"):
        if i.endswith('.txt'):
            f = open(articlespath + "/" + i, "r")
            rep = f.read()
            for j in club_name:
                if re.search(j, rep) != None:
                    df.at[j,'number_of_mentions'] += 1
    #Paint bar chart
    df_sorted = df.sort_index()
    fig, ax = plt.subplots()
    ax.set_title('Clubs mentioned by the media')
    ax.set_ylabel('Group name')
    ax.set_xlabel('number_of_mentions')
    ax.bar(x = df_sorted.index, height = df_sorted["number_of_mentions"])
    plt.xticks(rotation=90)
    return df.to_csv('task5.csv', index_label = "clubname"), fig.savefig('task5.png',bbox_inches='tight')
    
def task6():
    #Open and read data.json
    f = open(datafilepath, "r")
    data_dic = json.loads(f.read())
    club_name = data_dic["participating_clubs"]
    #Panda Dataframe initialization
    df = pd.DataFrame(index = club_name, columns = club_name, data = 0, dtype=np.dtype("float"))
    #Read dataframe from task5
    if os.path.exists('task5.csv'):
        df_task5 = pd.read_csv('task5.csv', index_col=[0])
    else:    
        task5()
        df_task5 = pd.read_csv('task5.csv', index_col=[0])
    #Fill in df with both mentioned reports number
    for i in range(0, len(club_name)):
        club_1 = df.index[i]
        for j in range(0, len(club_name)):
            club_2 = df.index[j]
            for k in listdir(articlespath + "/"):
                if k.endswith('.txt'):
                    f = open(articlespath + "/" + k, "r")
                    rep = f.read()
                    if re.search(club_1, rep) != None and re.search(club_2, rep) != None:
                        df.at[club_1,club_2] += 1
                    else:
                        pass
            #Calculate the similarity
            club_1_num = df_task5.at[club_1,'number_of_mentions']
            club_2_num = df_task5.at[club_2,'number_of_mentions']            
            if club_1_num == 0 and club_2_num == 0:
                sim = 1
            else:
                sim = 2 * df.at[club_1,club_2] / (club_1_num + club_2_num)
            df.at[club_1,club_2] = sim
    #Mask unrelevant part of the heatmap
    mask = df.to_numpy()
    mask = np.zeros_like(mask)
    mask[np.triu_indices_from(mask)] = True
    #Paint heatmap 
    fig, ax = plt.subplots()
    ax.set_title('Similarity score heatmap') 
    sns.heatmap(df,cmap='PuBu',mask=mask, xticklabels = True)
    return plt.savefig('task6.png',bbox_inches='tight')
    
def task7():
    #Read dataframe from former tasks
    if os.path.exists('task5.csv'):
        df_task5 = pd.read_csv('task5.csv', index_col=[0])
    else:    
        task5()
        df_task5 = pd.read_csv('task5.csv', index_col=[0])
    if os.path.exists('task2.csv'):
        df_task2 = pd.read_csv('task2.csv', index_col=[0])
    else:    
        task2()
        df_task2 = pd.read_csv('task2.csv', index_col=[0])
    total_goals = df_task2['goals_scored_byteam']
    #Open and read data.json
    f = open(datafilepath, "r")
    data_dic = json.loads(f.read())
    teams_codes = data_dic["teams_codes"]
    participating_clubs = data_dic["participating_clubs"]
    total_goals = total_goals.reindex(teams_codes)
    df_task5.insert(1, "total_goals",total_goals.values)
    df = df_task5.sort_values(by=['total_goals'])
    #Scatter Plot
    x = df["total_goals"]
    y = df["number_of_mentions"]
    colors = np.random.rand(20)
    fig, ax = plt.subplots()
    ax.set_xlabel('Total goals')
    ax.set_ylabel('number_of_mentions')
    plt.scatter(x, y, c=colors, alpha=0.5)
    return plt.savefig("task7.png",bbox_inches='tight')
    
def task8(filename):
    from nltk.corpus import stopwords
    stopWords = set(stopwords.words('english'))
    f = open(filename, "r")
    text = f.read()
    f.close()
    #Remove all non-alphabetic characters
    text = re.sub("[^A-Za-z\s\t\n]"," ",text)
    #Only one whitespace character exists between each word
    text = re.sub("[\n|\t]|[\s]{2,}"," ",text)
    #Change all uppercase characters to lower case
    text = text.lower()
    #Tokenlize the text
    text = nltk.word_tokenize(text)
    #Remove stop words and one character word
    text = [w for w in text if len(w) != 1 and w not in stopWords]
    return text
    
def task9():
    #Initialization
    result = []
    repname = []
    term = []
    vol = {}
    transformer = TfidfTransformer()
    rep_name = listdir(articlespath + "/")
    def cos_sim(v1,v2):
        return dot(v1,v2)/(norm(v1)*norm(v2)) 
    #Get the volcabulary from all the reports
    for i in rep_name:
        if i.endswith('.txt'):
            repname.append(i)
            rep = task8(articlespath + "/" + i)
            term = term + rep
            vol[i] = dict((x,rep.count(x)) for x in set(rep))
    term = list(set(term))
    repname = sorted(repname)
    term_counts = np.zeros((len(repname),len(term)),dtype = int)
    #Create Tf-idf vector for each report
    for i in range(0,len(term)):
        word = term[i]
        for j in range(0,len(repname)):
            rep = repname[j]
            try:
                term_counts[j][i] = vol[rep][word]
            except:
                pass
    tfidf = transformer.fit_transform(term_counts)
    array = tfidf.toarray()
    #Calculate the cosine similarity
    for i in range(0, len(repname)):
        rep1 = repname[i]
        for j in range(i + 1, len(repname)):
            rep2 = repname[j]
            sim = cos_sim(array[i], array[j])
            result.append([rep1, rep2, sim])
    # Generate list of top10 similartiy pairs
    top10 = sorted(result, key=lambda x:x[2],reverse=True)[:10]
    df = pd.DataFrame(top10)
    return df.to_csv('task9.csv',index=None)
