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

"""### Text Mining (Ammenities Column)"""

amenities=[]
for i in range(len(df)):
    if type(df['amenities'][i])==float:
       continue
    else:
      for item in df['amenities'][i]:
          amenities.append(item)

# tokenize all amenities and make it lowercase
amenities_tokens=[]
for i in amenities:
 tokens=nltk.word_tokenize(i)
 #print(tokens)
 lowercase_toekens=[]
 for item in tokens:
     lowercase_toekens.append(item.lower())
 #print(lowercase_toekens)
 amenities_tokens.append(lowercase_toekens)

 # clean up the tokens, only keep alphabets
cleaned_token_for_analysis=[]
for i in amenities_tokens:
   #print(i)
   cleaned_amenities_tokens=[]
   for item in i:
     if item.isalpha()==True:
        cleaned_amenities_tokens.append(item)
   cleaned_token_for_analysis.append(cleaned_amenities_tokens)

def display_term_frequency(n):
    flatten_list=[]
    for item in n:
        #print(item)
        for i in item:
          #print(i)
          flatten_list.append(i)
    term_frequency=pd.DataFrame(FreqDist(flatten_list).most_common(),columns=['Term','Term Frequency'])
    term_frequency.sort_values(by='Term Frequency',ascending=False,inplace=True)
    return term_frequency

# Remove rare terms
term_frequency=display_term_frequency(cleaned_token_for_analysis)
rare_terms_df=term_frequency['Term'][term_frequency['Term Frequency']<=10]
rare_terms_df.reset_index(drop=True,inplace=True)

rare_terms=[]
for i in range(len(rare_terms_df)):
   #print(rare_terms_df[i])
   rare_terms.append(rare_terms_df[i])
rare_terms=set(rare_terms)

def prepare_tokens_for_analysis(raw_tokens,filter_list):
    filtered_tokens=[]
    for item in raw_tokens:
        tokens=[]
        for token in item:
            if token not in filter_list:
               tokens.append(token)
        filtered_tokens.append(tokens)
    return filtered_tokens

token_for_analysis=prepare_tokens_for_analysis(cleaned_token_for_analysis,rare_terms)
# identify the most commmon terms (no stop word is applied)
term_frequency=display_term_frequency(token_for_analysis)
#print(term_frequency.head(40))

# # implement bigram, trigrams and four grams on original tokens to identify multi-word terms
# # Then go through the multi-words and identify the potential stop words

# multi_word_terms=[('carbon','monoxide','alarm'),('first','aid','kit'),('coffee','maker'),
#                   ('air','conditioning'),('hair','dryer'),('fire','extinguisher'),('bed','linens'),('body','soap'),
#                   ('extra','pillows','and','blankets'),('long','term','stays','allowed'),
#                   ('shower','gel'),('water ','kettle'),('smoke','alarm'),('dining','table'),('wine','glasses'),
#                   ('ceiling','fan'),('bbq','grill'),('cooking','basics'),('dedicated','workspace'),
#                   ('cleaning','products'),('private','entrance'),('outdoor','furniture'),
#                   ('baking','sheet'),('pets','allowed'),
#                   # words identified in round 2
#                   ('dishes','and','silverware'),('outdoor','dining','area'),('luggage','dropoff','allowed'),('board','games'),
#                   # free parking and on premises are used to be two words separately, I notice all four words appears together
#                    ('free','parking','on','premises'),('private','patio','or','balcony'),('hot','water','kettle'),
#                   ('central','heating'),('clothing','storage','closet'),('fire','pit'),
#                   ('central','air','conditioning'),('laundromat','nearby'),
#                   ('ethernet','connection'),('exercise','equipment'),('indoor','fireplace'),('gas','stove'),('parking','on','premises'),
#                   ('paid','parking'),('coffee','machine'),('free','dryer','in','unit'),('books','and','reading','material'),
#                   ('free','washer','in','unit'),('city','skyline','view'),('available','at','extra','cost'),('free','dryer','in','building'),
#                   ('outdoor','pool'),('mini','fridge'),('on','premises'),('hot','water'),('on','property'),('hot','tub'),
#                   ('security','cameras'),('clothing','storage'),('private','backyard','fully','fenced'),('sun','loungers'),('drying','rack'),
#                   ('electric','stove'),('ev','charger'),('stainless','steel','oven'),('paid','parking','on','premises'),('amazon','prime'),
#                   ('free','street','parking'),('patio','or','balcony'),('available','during','stay'),('available','upon','request'),
#                   ('available','all','year'),('amazon','prime','video'),('gym','in','building'),('free','driveway','parking','on','premises'),
#                   ('clothing','storage','closet','and','dresser'),('paid','parking','off','premises'),('ping','pong','table'),
#                   ('exterior','security','cameras','on','property'),('single','level','home'),('drying','rack','for','clothing'),
#                   ('stainless','steel','gas','stove'),('portable','fans'),('rice','maker'),('trash','compactor'),('barbecue','utensils'),
#                   ('stainless','steel'),('building','staff'),('drip','coffee','maker'),('standard','cable'),('single','oven'),
#                   ('stainless','steel', 'single','oven'),('free','parking','garage','on','premises'),
#                   ('private','outdoor','pool','available','all','year'),('shared','outdoor','pool','available','all','year'),
#                   ('shared','gym','in','building'),('lake','access'),('garden','view'),('books','and','toys'),('record','player'),
#                   ('bluetooth','sound','system'),('private','backyard'),('shared','pool'),('fully','fenced'),('in','building'),('hbo','max'),
#                   ('pack','n','crib'),('noise','decibel','monitors','on','property'),('host','greets','you'),('not','fully','fenced'),
#                   ('dryer','in','unit'),('split','type','ductless','system'),('free','washer','in','building'),('paid','dryer','in','building'),
#                   ('paid','washer','in','building'),('shared','backyard'),('apple','tv'),('private','hot','tub'),('outdoor','kitchen'),
#                   ('shared','patio','or','balcony'),('private','bbq','grill'),('resort','access'),('housekeeping','available','at','extra','cost'),
#                   ('sound','system'),('cleaning','available','during','stay'),('french','press'),('game','console'),('beach','essentials'),
#                   ('children','s','books','and','toys'),('children','s','dinnerware'),('baby','safety','gates'),('always','at','the','listing'),
#                   ('lock','on','bedroom','door'),('standalone','high','chair','available','upon','request'),('yoga','mat'),('arcade','games'),
#                   ('pack','n','crib','available','upon','request'),('pack','n','crib','always','at','the','listing'),('open','hours'),
#                   ('stainless','steel','electric','stove'),('shared','backyard','fully','fenced'),('outdoor','playground'),('fire','tv'),
#                   ('fast','wifi','mbps'),('open','specific','hours'),('tv','with','standard','cable'),('hdtv','with','amazon','prime','video'),
#                   ('closet','and','dresser'),('hdtv','with','roku'),('wardrobe','and','dresser'),('washer','in','unit'),('hdtv','with','netflix'),
#                   ('window','ac','unit'),('paid','pack','n','crib','available','upon','request'),('premium','cable'),('shared','outdoor','pool'),
#                   ('outdoor','shower'),('bread','maker'),('private','backyard','not','fully','fenced'),('smoking','allowed'),('private','pool'),
#                   ('private','hot','tub','available','all','year'),('courtyard','view'),('pocket','wifi'),('baby','bath'),('outlet','covers'),
#                   ('folding','or','convertible'),('free','carport','on','premises'),('french','press','keurig','coffee','machine'),
#                   ('stainless','steel','double','oven'),('coffee','maker','espresso','machine'),('drip','coffee','maker','keurig'),
#                   ('folding','or','convertible','high','chair'),('paid','street','parking'),('free','residential','garage','on','premises'),
#                   ('ginger','lily','farms'),('paid','street','parking','off','premises'),('available','seasonally','open','specific','hours')]

# Utilize the multi-words I constructed before and the result from bigram, trigram and four gram to determine which words should
# add into stop words list
stop_words=set(['on', 'and','premises', 'gas', 'in', 'unit', 'building', 'at', 'extra', 'cost', 'property', 'fully', 'fenced', 'private',
            'electric', 'stainless', 'steel', 'during', 'stay', 'upon', 'request', 'all', 'year', 'driveway', 'off', 'drip', 'single',
            'garage', 'shared', 'outdoor', 'pack', 'n', 'french', 'press', 's', 'the', 'listing', 'always', 'keurig', 'double', 'espresso',
            'machine', 'residential', 'ginger', 'lily', 'farms', 'folding', 'or', 'convertible', 'high','pocket','paid','free','for',
            'age','years','old','from','am','to','pm','split','type','ductless','system','open','hours','specific','every','day','seasonally',
            'nespresso','ages','standard','premium','street','central','nearby','indoor','dove','tesla','only','gilchrist','soames','london',
            'collection','charcoal','mbps','decibel','level','self','standalone','samsung','entrance','shades','space','spaces','recommendations',
            'ge','portable','a','week','days','whirlpool','induction','living','room','public','goods','heated','rooftop','inch','with','cable',
            'aux','sonos','bosch','available','not','furniture','essentials','bluetooth','frigidaire','lg','nintendo','switch','xbox','one','lot',
            'bose','other','terra','pure','pantene','kitchenaid','malin','goetz','thermador','by','trader','joe','various','brands','provided',
            'kenmore','dr','bronners','bergman','kelly','monday','friday','kirkland','series','x','organic','native','suave','molton','brown',
            'google','home','oil of olay','full size','smart','jbl','alexa','kitchen aid','travel size','galanze','brand','range','of',
            'clothing storage','bronner'])



tokens_stopword_removed=prepare_tokens_for_analysis(token_for_analysis,stop_words)
term_frequency=display_term_frequency(tokens_stopword_removed)
#print(term_frequency.head(50))


def produce_ngrams(token_for_analysis,ngram_value):
      if int(ngram_value) not in [2,3,4]:
         raise Exception("Unsupport ngram_value, ngram_value must between 2-4")
      ngrams_list=[]
      for i in token_for_analysis:
          #print(i)
          result=list(ngrams(i,int(ngram_value)))
          words=[]
          if int(ngram_value)==2:
             for item in result:
                 words.append(item[0]+' '+item[1])
             ngrams_list.append(words)
          elif int(ngram_value)==3:
              for item in result:
                  words.append(item[0]+' '+item[1]+' '+item[2])
              ngrams_list.append(words)
          elif int(ngram_value)==4:
              for item in result:
                  words.append(item[0]+' '+item[1]+' '+item[2]+' '+item[3])
              ngrams_list.append(words)
      return ngrams_list

#bigram
bigrams=produce_ngrams(tokens_stopword_removed,2)
term_frequency=display_term_frequency(bigrams)
#print(term_frequency.head(50))

#trigrams
trigrams=produce_ngrams(tokens_stopword_removed,3)
term_frequency=display_term_frequency(trigrams)
#print(term_frequency.head(50))

#fourgrams
fourgrams=produce_ngrams(tokens_stopword_removed,4)
term_frequency=display_term_frequency(fourgrams)
#print(term_frequency.head(50))

# updated version of the multi_word terms
multi_word_terms=[('body', 'soap'), ('children', 'books', 'toys'), ('dishes', 'silverware'), ('outlet', 'covers'), ('baking', 'sheet'),
('clothing', 'storage'), ('fire', 'tv'), ('dvd', 'player'), ('smart', 'lock'), ('pool', 'view'), ('ev', 'charger'), ('security', 'cameras'),
('noise', 'monitors'), ('cleaning', 'products'), ('valley', 'view'), ('lake', 'view'), ('clothing', 'storage', 'closet', 'dresser'),
('ethernet', 'connection'), ('patio', 'balcony'), ('hot', 'water', 'kettle'), ('smoke', 'alarm'), ('children', 'playroom'), ('wine', 'glasses'),
('cooking', 'basics'), ('closet', 'dresser'), ('drying', 'rack', 'clothing'), ('mini', 'fridge'), ('dining', 'table'), ('fire', 'extinguisher'),
('bed', 'linens'), ('fire', 'pit'), ('dining', 'area'), ('city', 'skyline', 'view'), ('books', 'reading', 'material'), ('board', 'games'),
('fireplace', 'guards'), ('children', 'dinnerware'), ('hdtv', 'netflix'), ('smoking', 'allowed'), ('hbo', 'max'), ('barbecue', 'utensils'),
('stationary', 'bike'), ('river', 'view'), ('lake', 'access'), ('trash', 'compactor'), ('hot', 'tub'), ('changing', 'table'), ('window', 'ac'),
('hdtv', 'roku'), ('amazon', 'prime'), ('drying', 'rack'), ('courtyard', 'view'), ('apple', 'tv'), ('hdtv', 'amazon', 'prime', 'video'),
('bbq', 'grill'), ('long', 'term', 'stays', 'allowed'), ('amazon', 'prime', 'video'), ('bread', 'maker'), ('record', 'player'),
('valet', 'parking'), ('hair', 'dryer'), ('sun', 'loungers'), ('ping', 'pong', 'table'), ('fast', 'wifi'), ('window', 'guards'),
('arcade', 'games'), ('exterior', 'security', 'cameras'), ('bowling', 'alley'), ('workout', 'bench'), ('game', 'console'),
('water', 'kettle'), ('garden', 'view'), ('resort', 'access'), ('mosquito', 'net'), ('shower', 'gel'), ('lock', 'bedroom', 'door'),
('pillows', 'blankets'), ('park', 'view'), ('baby', 'safety', 'gates'), ('yoga', 'mat'), ('table', 'corner', 'guards'), ('baby', 'bath'),
('pool', 'table'), ('mini', 'golf'), ('pets', 'allowed'), ('wardrobe', 'dresser'), ('clothing', 'storage', 'closet'),
('luggage', 'dropoff', 'allowed'), ('baby', 'monitor'), ('movie', 'theater'), ('dedicated', 'workspace'), ('air', 'conditioning'),
('ceiling', 'fan'), ('boat', 'slip'), ('first', 'aid', 'kit'), ('carbon', 'monoxide', 'alarm'), ('coffee', 'maker'), ('rice', 'maker'),
('mountain', 'view'), ('hot', 'water'), ('host', 'greets', 'you'), ('exercise', 'equipment'),('golf','course','view'),('children','bikes'),
('solar','panels'),('beach','access'),('ebike','charger'),('spa','services'),('climbing','wall'),('resort','view')
# catch the brandname and remove it later or repeated term and standardize it later
,('kitchen','aid'),('body','wash','body','soap'),('coffee','maker','coffee','maker','coffee','coffee'),
('coffee','maker','coffee','maker','coffee'),('coffee','maker','coffee'),('coffee','maker','coffee','maker'),('soap','body','soap'),
('ice','maker','refrigerator'),('convection','oven','oven'),('pool','infinity'),('full','size','refrigerator'),
('conditioner','conditioner'),('body','soap','body','soap'),('travel','size'),('pool','pool'),('oven','oven'),('full','size'),
('shampoo','shampoo')]

mwe_tokenizer=MWETokenizer(multi_word_terms,separator=' ')

multi_word_tokens=[]
for i in tokens_stopword_removed:
    #print(mwe_tokenizer.tokenize(i))
    multi_word_tokens.append(mwe_tokenizer.tokenize(i))

term_frequency=display_term_frequency(multi_word_tokens)
#print(term_frequency.head(50))

#bigram
bigrams=produce_ngrams(multi_word_tokens,2)
term_frequency=display_term_frequency(bigrams)
#print(term_frequency.head(50))

#trigrams
trigrams=produce_ngrams(multi_word_tokens,3)
term_frequency=display_term_frequency(trigrams)
#print(term_frequency.head(50))

#fourgrams
fourgrams=produce_ngrams(multi_word_tokens,4)
term_frequency=display_term_frequency(fourgrams)
#print(term_frequency.head(50))

# clean the tokens by standaridize them and remove stoped words that are multi-word terms
synonyms={'hdtv':'tv','roku tv':'tv','apple tv':'streaming service','fire tv':'streaming service',
          'netflix':'streaming service','amazon prime video':'streaming service','hulu':'streaming service',
          'roku':'streaming service','hbo max':'streaming service','chromecast':'streaming service',
          'body wash body soap':'body soap','coffee maker coffee maker coffee coffee':'coffee maker',
          'coffee maker coffee maker coffee':'coffee maker','coffee maker coffee':'coffee maker',
          'coffee maker coffee maker':'coffee maker','soap body soap':'body soap','convection oven oven':'oven',
          'ice maker refrigerator':'refrigerator','pool infinity':'pool',
          'full size refrigerator':'refrigerator','conditioner conditioner':'conditioner',
          'body soap body soap':'body soap','pool pool':'pool','oven oven':'oven','shampoo shampoo':'shampoo'}

multi_word_tokens_cleaned=[]
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
   multi_word_tokens_cleaned.append(new_token_list)

#term_frequency=display_term_frequency(multi_word_tokens_cleaned)
#print(term_frequency.head(20))

multi_word_tokens_stopword_removed=prepare_tokens_for_analysis(multi_word_tokens_cleaned,stop_words)
term_frequency=display_term_frequency(multi_word_tokens_stopword_removed)
print(term_frequency.head(50))

#bigram
bigrams=produce_ngrams(multi_word_tokens_stopword_removed,2)
term_frequency=display_term_frequency(bigrams)
print(term_frequency.head(50))

#trigrams
trigrams=produce_ngrams(multi_word_tokens_stopword_removed,3)
term_frequency=display_term_frequency(trigrams)
print(term_frequency.head(50))

#fourgrams
fourgrams=produce_ngrams(multi_word_tokens_stopword_removed,4)
term_frequency=display_term_frequency(fourgrams)
print(term_frequency.head(50))

"""### Plot the result"""

term_frequency=display_term_frequency(multi_word_tokens_stopword_removed)
plt.figure(figsize=(12, 8))

wc = WordCloud(width=1000, height=600,
               background_color='white')
wc.generate_from_frequencies(term_frequency.set_index('Term')['Term Frequency'].to_dict())

plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

plt.figure(figsize=(10,6))
bars=plt.bar(term_frequency.head(10)['Term'],term_frequency.head(10)['Term Frequency'])
plt.title('Common Amentities')
plt.xlabel('Ammentities')
plt.ylabel("Frequency")
plt.xticks(rotation=340)
plt.bar_label(bars,padding=1)
plt.tight_layout()
plt.show()
