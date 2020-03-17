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
sentence