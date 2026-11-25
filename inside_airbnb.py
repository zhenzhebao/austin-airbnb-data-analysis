import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from wordcloud import WordCloud

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import MWETokenizer

df=pd.read_csv('/content/drive/MyDrive/Data Analytics/insideairbnb_Austin/listings.csv')

"""## Data Cleaning"""

pd.set_option('display.max_columns', df.shape[1])
df.head()

df.info()

"""### Remove Duplicates"""

df.drop_duplicates(inplace=True)

"""### Remove column that is completedly empty

Following columns are removed from dataset:

1.  neighborhood_overview
2.  host_since
3.  host_response_time
4.  host_response_rate
5.  host_acceptance_rate
6.  host_thumbnail_url
7.  host_neighbourhood
8.  host_total_listings_count
9.  host_verifications
10. neighbourhood
11. neighbourhood_group_cleansed
12. bathrooms
13. calendar_updated
14. instant_bookable
"""

df.drop(columns=list(df.count()[df.count()==0].index),inplace=True)
#df.info()

"""### Drop Columns that are not relevant to Data Analysis

Follwing columns are dropped:
1.  ID
2.  listing_url
3.  scrape_id
4.  last_scraped
5.  source
6.  picture_url
7.  host_id
8.  host_url
9.  host_profile_id
10. host_profile_url
11. host_name
12. host_picture_url
13. calendar_last_scraped
"""

df.drop(columns=['id', 'listing_url','scrape_id','last_scraped','source','picture_url','host_id', 'host_url',
                 'host_profile_id', 'host_profile_url','host_name','host_picture_url','calendar_last_scraped'],inplace=True)

"""### Drop columns that contains same price information

The follwing columns are dropped:
1.  price_quote_price_per_night
2.  price_quote_total_price
3.  minimum_minimum_nights
4.  maximum_maximum_nights
5.  price_quote_raw
"""

### Remove dollar sign and comma from price and convert it to float
df['price']=df['price'].str.replace('$','').str.replace(',','').astype(float)

if abs((df['price'][(df['price'].isna()==False )&(df['price_quote_price_per_night'].isna()==False)]-
    df['price_quote_price_per_night'][(df['price'].isna()==False )&(df['price_quote_price_per_night'].isna()==False)])).sum()==0:
    print('Price and Price Quote Price Per Night is the same')
    # drop rows if both columns contains null value
    df.dropna(subset=['price','price_quote_price_per_night'],how='all',inplace=True)
if df['price'].isna().sum()==0:
   df.drop(columns=['price_quote_price_per_night'],inplace=True)

# conver both column into correct data format
if df[['price_quote_checkin_date','price_quote_checkout_date']].isna().sum().sum()==0:
   df['price_quote_checkin_date']=pd.to_datetime(df['price_quote_checkin_date'])
   df['price_quote_checkout_date']=pd.to_datetime(df['price_quote_checkout_date'])

df_test=df[['price','price_quote_checkin_date','price_quote_checkout_date','price_quote_total_price']][df['price_quote_total_price'].isna()==False]
df_test['calculated_total']=(df_test['price_quote_checkout_date']-df_test['price_quote_checkin_date']).dt.days*df_test['price']
#The dataset has 10005 records, after remove 16 records with nulls in price_quote_total_price column
# we have 9989 records, out these records 7782 records' quote price is the same as calculated total
print('Number of records:',len(df_test))
print('Number of records with same calulated and qouted total:',len(df_test[df_test['calculated_total']==df_test['price_quote_total_price']]))
# The sum of absolute differnce between calculated total and quote total is 140.72, mean is 0.06, maximum is 1.7
# Because the difference is small, therefore price_quote_total_price can be calculated by time period in days times price
#Therefore we will remove price_quote_total_price column
print('Sum of absolute difference:',round((abs(df_test['calculated_total']-df_test['price_quote_total_price'])).sum(),2))
print('Mean of absolute difference:',round((abs(df_test['calculated_total']-df_test['price_quote_total_price'])).sum()/(len(df_test)-len(df_test[df_test['calculated_total']==df_test['price_quote_total_price']])),2))
print('Maximum of absolute difference:',round((abs(df_test['calculated_total']-df_test['price_quote_total_price'])).max(),2))

del df_test
df.drop(columns=['price_quote_total_price'],inplace=True)

if abs((df['minimum_nights'][(df['minimum_nights'].isna()==False) &(df['minimum_minimum_nights'].isna()==False)]-df['minimum_minimum_nights'][(df['minimum_nights'].isna()==False) &(df['minimum_minimum_nights'].isna()==False)])).sum()==0:
   print('minimum_nights and minimum_minimum_nights is the same')
if len(df[(df['minimum_nights'].isna()==True) &(df['minimum_minimum_nights'].isna()==True)])==df['minimum_nights'].isna().sum():
   df.drop(columns=['minimum_minimum_nights'],inplace=True)

if abs((df['maximum_nights'][(df['maximum_nights'].isna()==False) & (df['maximum_maximum_nights'].isna()==False)]-df['maximum_maximum_nights'][(df['maximum_nights'].isna()==False) & (df['maximum_maximum_nights'].isna()==False)])).sum()==0:
   print('maximum_nights and maximum_maximum_nights is the same')
if len(df[(df['maximum_nights'].isna()==True) & (df['maximum_maximum_nights'].isna()==True)])==df['maximum_nights'].isna().sum():
   df.drop(columns=['maximum_maximum_nights'],inplace=True)

df.reset_index(drop=True,inplace=True)
df.drop(columns=['price_quote_raw'],inplace=True)

# # After explore the price_quote_raw column in detail, most information has been stored in price column
# # Therefore, the price_quote_raw column will be dropped
# def retrieve_fields(n):
#     result=json.loads(n)
#     fields=[]
#     for i in result.keys():
#         #print(i)
#          fields.append(i)
#          #print(result[i])
#          for key in result[i].keys():
#               #print(key)
#               #print(result[i][key])
#               fields.append(key)
#     return fields

# if len(df['price_quote_raw'].apply(retrieve_fields).value_counts())==1:
#    print('All value in price_quote_raw columns has the same fields')

# def flatten_result(n):
#     flattened_result={}
#     result=json.loads(n)
#     for outlier_key in result.keys():
#         if len(result.keys())==1 and type(result[outlier_key])==dict:
#             content=result[outlier_key]
#             #print(content)
#             for key in content.keys():
#                 #print(key)
#                 #print(content[key])
#                 flattened_result.update({key:str(content[key])})
#         else:
#             print(outlier_key)
#             print(result[outlier_key])
#     return flattened_result

# list_of_flatten_result=[]
# flattened_price_quote_raw=df['price_quote_raw'].apply(flatten_result)
# for i in range(len(df)):
#    list_of_flatten_result.append(flattened_price_quote_raw[i])

# price_quote=pd.DataFrame(list_of_flatten_result)
# price_quote.replace('None',np.nan,inplace=True)

# #remove columns that are completely empty
# price_quote.drop(columns=list(price_quote.isna().sum()[price_quote.isna().sum()==len(price_quote)].index),inplace=True)
# # remove currency column as it only have 16 nulls and rest of them are USD
# price_quote.drop(columns=['currency'],inplace=True)

# # covert the column into correct data type
# price_quote['requested_checkin_date']=pd.to_datetime(price_quote['requested_checkin_date'])
# price_quote['requested_checkout_date']=pd.to_datetime(price_quote['requested_checkout_date'])
# price_quote['taxes']=price_quote['taxes'].astype(float)
# price_quote['total_price']=price_quote['total_price'].astype(float)
# price_quote['price_per_night']=round(price_quote['price_per_night'].astype(float),2)

# print(price_quote.info())

# df=pd.merge(df,price_quote,left_index=True,right_index=True)
# del price_quote

# if len(df[df['requested_checkin_date']!=df['price_quote_checkin_date']])==0:
#     print('price_quote_checkin_date and requested_checkin_date is the same')
#     df.drop(columns=['requested_checkin_date'],inplace=True)
# if len(df[df['requested_checkout_date']!=df['price_quote_checkout_date']])==0:
#     print('requested_checkout_date andprice_quote_checkout_date is the same')
#     df.drop(columns=['requested_checkout_date'],inplace=True)

# print(abs(df['price']-df['price_per_night']).sum())
# print(df['raw_price_line_items'][0])
# print(df['raw_price_line_items'][1])
# print(round(abs((df['price_quote_checkout_date']-df['price_quote_checkin_date']).dt.days*df['price']-df['total_price']).sum(),2))

# df[['price','taxes','total_price','is_available','discount_amount','price_per_night','nightly_subtotal','is_monthly_priced',
#     'discounted_subtotal','raw_price_line_items']][(df['taxes'].isna()==False)& (df['discount_amount'].isna()==False)]

# correct data type for column host_is_superhost,host_has_profile_pic and host_identity_verified
if df[['host_is_superhost','host_has_profile_pic','host_identity_verified']].isna().sum().sum()==0:
   df['host_is_superhost']=df['host_is_superhost'].replace({'t':True,'f':False})
   df['host_has_profile_pic']=df['host_has_profile_pic'].replace({'t':True,'f':False})
   df['host_identity_verified']=df['host_identity_verified'].replace({'t':True,'f':False})

# clean up the amentities column
def clean_result(n):
  clean_result=json.loads(n)
  if len(clean_result)==0:
     return np.nan
  else:
     return clean_result

df['amenities']=df['amenities'].apply(clean_result)

# save all unique amenities from the dataset
unique_amenities=[]
for i in range(len(df)):
    if type(df['amenities'][i])==float:
       continue
    else:
      for item in df['amenities'][i]:
          if item not in unique_amenities:
             unique_amenities.append(item)
          else:
            continue

# tokenize all amenities
amenities_tokens=[]
for i in unique_amenities:
 tokens=nltk.word_tokenize(i)
 #print(tokens)
 lowercase_toekens=[]
 for item in tokens:
     lowercase_toekens.append(item.lower())
 #print(lowercase_toekens)
 amenities_tokens.append(lowercase_toekens)

 # clean up the tokens, only keep alphabets
token_for_analysis=[]
for i in amenities_tokens:
   #print(i)
   cleaned_amenities_tokens=[]
   for item in i:
     if item.isalpha()==True:
        cleaned_amenities_tokens.append(item)
   token_for_analysis.append(cleaned_amenities_tokens)

def display_term_frequency(n):
    flatten_list=[]
    for item in n:
        #print(item)
        for i in item:
          #print(i)
          flatten_list.append(i)
    term_frequency=pd.DataFrame(FreqDist(flatten_list).most_common(),columns=['Term','Frequency'])
    term_frequency.sort_values(by='Frequency',ascending=False,inplace=True)
    return term_frequency

# identify the most commmon terms (no stop word is applied)
# few observations at this stage, with, inch, mbps, fast should be remove from the term for analysis as it doesn't provide any meaning
term_frequency=display_term_frequency(token_for_analysis)
print(term_frequency.head(20))

# add stop word
stop_word=set(stopwords.words('english'))
stop_word.update(['inch','mbps','fast'])

tokens_stopword_removed=[]
for item in token_for_analysis:
    tokens=[]
    for token in item:
        if token not in stop_word:
           tokens.append(token)
    tokens_stopword_removed.append(tokens)

term_frequency=display_term_frequency(tokens_stopword_removed)
print(term_frequency.head(20))

# implement bigram, trigrams and four grams on original tokens to identify multi-word terms
# multi-word terms apple tv, fire tv,body soap, hbo max, amazon prime video,bluetooth sound system
# we know fast and mbps are related to WIFI, therefore we should add both term into stop word based on the result from bigram and trigram
#bigram
bigrams=[]
for i in token_for_analysis:
    #print(i)
    result=list(ngrams(i,2))
    words=[]
    for item in result:
        words.append(item[0]+' '+item[1])
    bigrams.append(words)

term_frequency=display_term_frequency(bigrams)
print(term_frequency.head(20))

#trigrams
trigrams=[]
for i in token_for_analysis:
    #print(i)
    result=list(ngrams(i,3))
    words=[]
    for item in result:
        words.append(item[0]+' '+item[1]+' '+item[2])
    trigrams.append(words)

term_frequency=display_term_frequency(trigrams)
print(term_frequency.head(20))

#fourgrams
fourgrams=[]
for i in token_for_analysis:
    #print(i)
    result=list(ngrams(i,4))
    words=[]
    for item in result:
        words.append(item[0]+' '+item[1]+' '+item[2]+' '+item[3])
    fourgrams.append(words)

term_frequency=display_term_frequency(fourgrams)
print(term_frequency.head(20))

# Identify the multi-words during tokenization
multi_word_terms=[('air','conditioning'),('apple','tv'),('fire','tv'),('roku','tv'),('body','soap'),('amazon','prime','video'),
                  ('bluetooth','sound','system'),('hbo','max'),('stainless','steel'),('outdoor','pool'),
                  ('available','at','extra','cost')]
mwe_tokenizer=MWETokenizer(multi_word_terms,separator=' ')

multi_word_tokens=[]
for i in token_for_analysis:
    #print(mwe_tokenizer.tokenize(i))
    multi_word_tokens.append(mwe_tokenizer.tokenize(i))

term_frequency=display_term_frequency(multi_word_tokens)
print(term_frequency.head(50))

# apply synonyms list on the multi-word tokens we have
synonyms={'hdtv':'tv','apple tv':'tv','fire tv':'tv','roku tv':'tv',
          'netflix':'streaming service','amazon prime video':'streaming service','hulu':'streaming service',
          'roku':'streaming service','hbo max':'streaming service'}

multi_word_tokens_with_synonyms=[]
for i in multi_word_tokens:
   #print(i)
   new_token_list=[]
   for item in i:
       #print(item)
       if item in synonyms.keys():
          #print(item)
          new_token_list.append(synonyms[item])
       else:
          new_token_list.append(item)
   multi_word_tokens_with_synonyms.append(new_token_list)

term_frequency=display_term_frequency(multi_word_tokens_with_synonyms)
#print(term_frequency.head(20))

# applied the stop word
# we should remove words such as available, standard, hours, open based on initial result
#initial result
# # stop_word=set(stopwords.words('english'))
# # stop_word.update(['inch','mbps','fast'])

# multi_word_tokens_with_synonyms_stopword_removed=[]
# for item in multi_word_tokens_with_synonyms:
#     tokens=[]
#     for token in item:
#         if token not in stop_word:
#            tokens.append(token)
#     multi_word_tokens_with_synonyms_stopword_removed.append(tokens)

# term_frequency=display_term_frequency(multi_word_tokens_with_synonyms_stopword_removed)
# print(term_frequency.head(20))

stop_word=set(stopwords.words('english'))
stop_word.update(['inch','mbps','fast','available','standard','hours','open'])

multi_word_tokens_with_synonyms_stopword_removed=[]
for item in multi_word_tokens_with_synonyms:
    tokens=[]
    for token in item:
        if token not in stop_word:
           tokens.append(token)
    multi_word_tokens_with_synonyms_stopword_removed.append(tokens)

term_frequency=display_term_frequency(multi_word_tokens_with_synonyms_stopword_removed)
print(term_frequency.head(30))

plt.figure(figsize=(12, 8))

wc = WordCloud(width=1000, height=600,
               background_color='white')
wc.generate_from_frequencies(term_frequency.set_index('Term')['Frequency'].to_dict())

plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

plt.figure(figsize=(10,6))
bars=plt.bar(term_frequency.head(10)['Term'],term_frequency.head(10)['Frequency'])
plt.title('Common Amentities')
plt.xlabel('Ammentities')
plt.ylabel("Frequency")
plt.xticks(rotation=340)
plt.bar_label(bars,padding=1)
plt.tight_layout()
plt.show()
