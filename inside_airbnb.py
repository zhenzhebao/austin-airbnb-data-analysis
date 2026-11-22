import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import json

df=pd.read_csv('/content/drive/MyDrive/Data Analytics/insideairbnb_Austin/listings.csv')

"""## Data Cleaning"""

pd.set_option('display.max_columns', df.shape[1])
df.head()

df.info()

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

"""price_quote_raw column"""

def retrieve_fields(n):
    result=json.loads(n)
    fields=[]
    for i in result.keys():
        #print(i)
         fields.append(i)
         #print(result[i])
         for key in result[i].keys():
              #print(key)
              #print(result[i][key])
              fields.append(key)
    return fields

if len(df['price_quote_raw'].apply(retrieve_fields).value_counts())==1:
   print('All value in price_quote_raw columns has the same fields')

def flatten_result(n):
    flattened_result={}
    result=json.loads(n)
    for outlier_key in result.keys():
        if len(result.keys())==1 and type(result[outlier_key])==dict:
            content=result[outlier_key]
            #print(content)
            for key in content.keys():
                #print(key)
                #print(content[key])
                flattened_result.update({key:str(content[key])})
        else:
            print(outlier_key)
            print(result[outlier_key])
    return flattened_result

list_of_flatten_result=[]
flattened_price_quote_raw=df['price_quote_raw'].apply(flatten_result)
for i in range(len(df)):
   list_of_flatten_result.append(flattened_price_quote_raw[i])

price_quote=pd.DataFrame(list_of_flatten_result)
price_quote.replace('None',np.nan,inplace=True)

#remove columns that are completely empty
price_quote.drop(columns=list(price_quote.isna().sum()[price_quote.isna().sum()==len(price_quote)].index),inplace=True)
# remove currency column as it only have 16 nulls and rest of them are USD
price_quote.drop(columns=['currency'],inplace=True)

# covert the column into correct data type
price_quote['requested_checkin_date']=pd.to_datetime(price_quote['requested_checkin_date'])
price_quote['requested_checkout_date']=pd.to_datetime(price_quote['requested_checkout_date'])
price_quote['taxes']=price_quote['taxes'].astype(float)
price_quote['total_price']=price_quote['total_price'].astype(float)

df=pd.merge(df,price_quote,left_index=True,right_index=True)
del price_quote
