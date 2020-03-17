from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
from nltk.sentiment.util import *
from newsapi import NewsAPI
import datetime
import sys, csv, json
import requests
import unicodedata
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import pygal
from pygal.style import LightenStyle, LightColorizedStyle
import json
from flask import Flask, render_template, Response, request, redirect, url_for
app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index2.html')

@app.route('/DJIA/', methods=['POST'])
def move_DJIA():
    param = {
    'q': ".DJI", # Stock symbol (ex: "AAPL")
    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "INDEXDJX", # Stock exchange symbol on which stock is traded (ex: "NASD")
    'p': "3M" # Period (Ex: "1Y" = 1 year)
           }
    # get price data (return pandas dataframe)
    df = get_price_data(param)
    df.to_csv('C:/Users/ansha/Anaconda3/FlaskApp/djia.csv')
    line = pd.read_csv("djia.csv",index_col=False)
 
    df = pd.DataFrame(data=line)

    dates = df['Unnamed: 0']

    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    line_chart = pygal.Line(x_label_rotation=20, x_labels_major_every=6, show_minor_x_labels=False, human_readable=True)
    line_chart.title = 'Dow Jones: DJIA'
    line_chart.x_labels = map(str, dates)
    line_chart.add('Open', o)
    line_chart.add('High', h)
    line_chart.add('Low', l)
    line_chart.add('Close', c)
    graph_data = line_chart.render_data_uri()
    return render_template("an.html", graph_data=graph_data)

@app.route('/AAPL/', methods=['POST'])
def move_AAPL():
    param = {
    'q': ".IXIC", # Stock symbol (ex: "AAPL")
    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "INDEXNASDAQ", # Stock exchange symbol on which stock is traded (ex: "NASD")
    'p': "3M" # Period (Ex: "1Y" = 1 year)
           }
    df = get_price_data(param)
    df.to_csv('C:/Users/ansha/Anaconda3/FlaskApp/nasd1.csv')
    line = pd.read_csv("nasd1.csv",index_col=False)
    
 
    df = pd.DataFrame(data=line)

    dates = df['Unnamed: 0']

    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    line_chart = pygal.Line(x_label_rotation=20, x_labels_major_every=6, show_minor_x_labels=False, human_readable=True)
    line_chart.title = 'NASDAQ'
    line_chart.x_labels = map(str, dates)
    line_chart.add('Open', o)
    line_chart.add('High', h)
    line_chart.add('Low', l)
    line_chart.add('Close', c)
    graph_data = line_chart.render_data_uri()
    return render_template("an.html", graph_data=graph_data)
              
@app.route('/TWTR/', methods=['POST'])
def move_TETR():
    line = pd.read_csv("nasd.csv")
 
    df = pd.DataFrame(data=line)

    dates = df['Date']

    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    line_chart = pygal.Line(x_label_rotation=20, x_labels_major_every=6, show_minor_x_labels=False, human_readable=True)
    line_chart.title = 'NASDAQ-100: NDX'
    line_chart.x_labels = map(str, dates)
    line_chart.add('Open', o)
    line_chart.add('High', h)
    line_chart.add('Low', l)
    line_chart.add('Close', c)
    line_chart.value_formatter = lambda x: "%.2f" % x
    graph_data = line_chart.render_data_uri()
    return render_template("an.html", graph_data=graph_data)

@app.route("/forward/", methods=['POST'])
def move_forward():
    test_start_date = datetime.datetime.now() - datetime.timedelta(days = 3)
    test_end_date = datetime.datetime.now() - datetime.timedelta(days = 2)
    df_stocks = pd.read_pickle('pickled_para.pkl')
    df_stocks['prices'] = df_stocks['close'].apply(np.int64)
    # selecting the prices and articles
    df_stocks = df_stocks[['prices', 'articles']]
    df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))
    df_stocks
    df = df_stocks[['prices']].copy()
    # Adding new columns to the data frame
    df["compound"] = ''
    df["neg"] = ''
    df["neu"] = ''
    df["pos"] = ''
    sid = SentimentIntensityAnalyzer()
    for date, row in df_stocks.T.iteritems():
        try:
            sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
            ss = sid.polarity_scores(sentence)
            df.set_value(date, 'compound', ss['compound'])
            df.set_value(date, 'neg', ss['neg'])
            df.set_value(date, 'neu', ss['neu'])
            df.set_value(date, 'pos', ss['pos'])
        except TypeError:
            print (df_stocks.loc[date, 'articles'])
            print (date)
    
    test = df.ix[test_start_date:test_end_date]


    ##
    #test_start_date = '2018-03-09'
    #test_end_date = '2018-03-10'
    test = df.ix[test_start_date:test_end_date]
  
    ##    
    
    # Calculating the sentiment score
    sentiment_score_list = []
    for date, row in test.T.iteritems():
        sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
        #sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
        sentiment_score_list.append(sentiment_score)
        numpy_df_test = np.asarray(sentiment_score_list)
     
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    #loaded_model.fit(numpy_df_test,test['prices'])   
    result = loaded_model.predict(numpy_df_test)
    
    difference=result[1]-result[0]
    if(difference>0):
        sentence="Stock Price will rise"
    else:
        sentence="Stock Price will fall"
    
    param = {
    'q': ".DJI", # Stock symbol (ex: "AAPL")
    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "INDEXDJX", # Stock exchange symbol on which stock is traded (ex: "NASD")
    'p': "1M" # Period (Ex: "1Y" = 1 year)
           }
    # get price data (return pandas dataframe)
    df = get_price_data(param)
    df.to_csv('C:/Users/ansha/Anaconda3/FlaskApp/djia.csv')
    line = pd.read_csv("djia.csv",index_col=False)
 
    df = pd.DataFrame(data=line)

    dates = df['Unnamed: 0']

    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    line_chart = pygal.Line(x_label_rotation=20, x_labels_major_every=3, show_minor_x_labels=False, human_readable=True)
    line_chart.title = 'Dow Jones: DJIA'
    line_chart.x_labels = map(str, dates)
    line_chart.add('Open', o)
    line_chart.add('High', h)
    line_chart.add('Low', l)
    line_chart.add('Close', c)
    graph_data = line_chart.render_data_uri()



    #print(result) 
    
    
    
    
    return render_template("an1.html",data=sentence,graph_data=graph_data,data1=difference)

@app.route("/NASDAQ/", methods=['POST'])
def move_NASDAQ():
    test_start_date = datetime.datetime.now() - datetime.timedelta(days = 5)
    test_end_date = datetime.datetime.now() #- datetime.timedelta(days = 1)
    df_stocks = pd.read_pickle('pickled_para_NDAQ.pkl')
    df_stocks['prices'] = df_stocks['close'].apply(np.int64)
    # selecting the prices and articles
    df_stocks = df_stocks[['prices', 'articles']]
    df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))
    df_stocks
    df = df_stocks[['prices']].copy()
    # Adding new columns to the data frame
    df["compound"] = ''
    df["neg"] = ''
    df["neu"] = ''
    df["pos"] = ''
    sid = SentimentIntensityAnalyzer()
    for date, row in df_stocks.T.iteritems():
        try:
            sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
            ss = sid.polarity_scores(sentence)
            df.set_value(date, 'compound', ss['compound'])
            df.set_value(date, 'neg', ss['neg'])
            df.set_value(date, 'neu', ss['neu'])
            df.set_value(date, 'pos', ss['pos'])
        except TypeError:
            print (df_stocks.loc[date, 'articles'])
            print (date)

    test = df.ix[test_start_date:test_end_date]


    ##
    #test_start_date = '2018-03-09'
    #test_end_date = '2018-03-10'
    test = df.ix[test_start_date:test_end_date]

    ##    

    # Calculating the sentiment score
    sentiment_score_list = []
    for date, row in test.T.iteritems():
        sentiment_score = np.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
        #sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
        sentiment_score_list.append(sentiment_score)
        numpy_df_test = np.asarray(sentiment_score_list)
 
    filename = 'finalized_model_NDAQ.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    #loaded_model.fit(numpy_df_test,test['prices'])   
    result = loaded_model.predict(numpy_df_test)

    difference=result[1]-result[0]
    if(difference>0):
        sentence="Stock Price will rise"
    else:
        sentence="Stock Price will fall"
    param = {
    'q': ".IXIC", # Stock symbol (ex: "AAPL")
    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "INDEXNASDAQ", # Stock exchange symbol on which stock is traded (ex: "NASD")
    'p': "1M" # Period (Ex: "1Y" = 1 year)
           }
    df = get_price_data(param)
    df.to_csv('C:/Users/ansha/Anaconda3/FlaskApp/nasd1.csv')
    line = pd.read_csv("nasd1.csv",index_col=False)
    
 
    df = pd.DataFrame(data=line)

    dates = df['Unnamed: 0']

    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']
    line_chart = pygal.Line(x_label_rotation=20, x_labels_major_every=3, show_minor_x_labels=False, human_readable=True)
    line_chart.title = 'NASDAQ'
    line_chart.x_labels = map(str, dates)
    line_chart.add('Open', o)
    line_chart.add('High', h)
    line_chart.add('Low', l)
    line_chart.add('Close', c)
    graph_data = line_chart.render_data_uri()

    return render_template("an1.html",data=sentence,graph_data=graph_data,data1=difference)

if __name__ == "__main__":
    app.run(debug=True)