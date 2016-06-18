###### code bits n pieces for Scikit Learn based Topic Model ###### 

# Topic Model modules
from __future__ import print_function
from time import time
from itertools import chain

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


###### Cleaning and tokenising option 1 ###### 
###### using regex patters and nltk ###### 

import re
import nltk
# nltk.download()
from nltk.corpus import stopwords

problemchars = re.compile(r'[\[=\+/&<>;:!\\|*^\'"\?%$.@)°#(_\,\t\r\n0-9-—\]]')
url_finder = re.compile(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
emojis = re.compile(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]+', 
    re.UNICODE)
stop = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
username = re.compile(r'(@)\w+( )')
# hashtag = re.compile(r'#(\w+)')           #maintain hashtags as a 'common word'
redate = re.compile(r'^(?:(?:(?:0?[13578]|1[02])(\/|-|\.)31)\1|(?:(?:0?[1,3-9]|1[0-2])(\/|-|\.)(?:29|30)\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:0?2(\/|-|\.)29\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:(?:0?[1-9])|(?:1[0-2]))(\/|-|\.)(?:0?[1-9]|1\d|2[0-8])\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$')


#example call for cleaning a row - takes pandas dataframe 
df_gsr_pandas['eventDescription_clean'] = df_gsr_pandas['eventDescription'].map(lambda w: stop.sub('', problemchars.sub('', emojis.sub('', url_finder.sub('', username.sub('', w.lower().strip()))))))




###### Cleaning and tokenising option 2 ###### 
###### using Spark MLlib ###### 

# take Spark dataframe object

# from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import NGram
from pyspark.sql.functions import col
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

# from nltk.stem import WordNetLemmatizer
# LEMMER = WordNetLemmatizer()
# stemmed = [LEMMER.lemmatize(w) for w in no_stopwords]


# function to tokenise cleaned text
def token(dataframe, in_col, out_col):
    
    tokenizer = Tokenizer(inputCol=in_col, outputCol=out_col)
    dataframe = tokenizer.transform(dataframe)
    
    dataframe.printSchema()
    
    return dataframe


# function to remove stopwords - NOT IN USE - error thrown on importing 'StopWordsRemover'
def stop(dataframe, in_col, out_col):
    
    remover = StopWordsRemover(inputCol=in_col, outputCol=out_col)
    remover.transform(dataframe).show(truncate=False)
        
    return dataframe


# function to find bigrams and show most frequent - change n for n-gram size
def ngram(dataframe, in_col, out_col, n):
    
    ngram = NGram(n=n, inputCol=in_col, outputCol=out_col)
    dataframe = ngram.transform(dataframe)
    
    # summarise top n-grams
    dataframe\
    .groupBy(out_col)\
    .count()\
    .sort(col("count").desc())\
    .show()
    
    return dataframe


#function to TF-IDF - change n for number of features
def tfidf(dataframe, in_col1, out_col1, in_col2, out_col2, n):

    global idfModel
    
    hashingTF = HashingTF(inputCol=in_col1, outputCol=out_col1, numFeatures=n)
    featurizedData = hashingTF.transform(dataframe)
    idf = IDF(inputCol=in_col2, outputCol=out_col2)
    idfModel = idf.fit(featurizedData)
    dataframe = idfModel.transform(featurizedData)
    
    return dataframe



def word2vec(dataframe, in_col, out_col):
    
    word2Vec = Word2Vec(inputCol=in_col, outputCol=out_col)
    
    model = word2Vec.fit(dataframe)
    dataframe = model.transform(dataframe)
    
    dataframe.printSchema()

    return dataframe



# call functions
myDF_GSR = token(myDF_GSR, "eventDescription_clean", "eventDescription_token")
# myDF_GSR = ngram(myDF_GSR, "eventDescription_token", "ngram", 2)
# myDF_GSR = tfidf(myDF_GSR, "eventDescription_token", "rawFeatures", "rawFeatures", "features", 10)
# myDF_GSR = word2vec(myDF_GSR, "eventDescription_token", "word_vec")

# #inspect new features
# myDF_GSR.select(["eventDescription", "eventDescription_clean", "eventDescription_token", "rawFeatures", "features", "word_vec"]).toPandas().head(20)

 
 
 

###### Scikit Learn Topic Modelling ###### 
###### LDA and NMF ###### 

# CHANGE PARAMETERS AS REQUIRED
n_samples = 100
n_features = 1000
n_topics = 7
n_top_words = 20



# create sequence of strings to feed into SKlearn Topic Model
GSR_text = myDF_GSR.select(['eventDescription_clean']).toPandas()
GSR_text = list(chain.from_iterable(GSR_text.values.tolist()))


# output human readable topic content
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    


# create list of topic words for labelling
def list_top_words(model, feature_names, n_top_words):
    list_topic = []
    
    for topic_idx, topic in enumerate(model.components_):
        list_mini_topic = []

        for i in topic.argsort()[:-n_top_words - 1:-1]:
            list_mini_topic.append(feature_names[i])
        
        list_topic.append(list_mini_topic)
        
    return list_topic


    
# MODEL ONE - NMF
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(input='content',
                                   #max_df=0.95, 
                                   #min_df=0.1, 
                                   max_features=n_features,
                                   stop_words='english')


tfidf = tfidf_vectorizer.fit_transform(GSR_text)


print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(input='content',
                                #max_df=0.95, 
                                #min_df=2, 
                                max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(GSR_text)

print("Fitting the NMF model with tf-idf features,"
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))

nmf = NMF(n_components=n_topics, 
          random_state=1, 
          alpha=.1, 
          l1_ratio=.5).fit(tfidf)

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, 
                tfidf_feature_names, 
                n_top_words)


# MODEL 2: LDA
print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, 
                                max_iter=5,
                                learning_method='online', 
                                learning_offset=50.,
                                random_state=0).fit(tf)

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, 
                tf_feature_names, 
                n_top_words)



# build list of lists with top n topic words - used in labelling functions
topic_words = list_top_words(lda, 
                tf_feature_names, 
                n_top_words)

