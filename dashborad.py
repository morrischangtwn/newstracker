import time

import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px
from pygooglenews import GoogleNews
import json
import time
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta


st.set_page_config(
    page_title="Taiwan News Tracker",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded",
)
def tokenize_title(title):
    # Tokenize the title into words
    tokens = word_tokenize(title)
    return tokens


print('-------')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Function to tokenize and filter out useless tokens
def tokenize_and_remove_useless(title):
    # Tokenize the title into words
    tokens = word_tokenize(title)
    
    # Remove stop words and short tokens
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and len(token) > 2 and token.lower() != 'taiwan']
    return filtered_tokens


st.title('International News Coverage on Taiwan Tracker')

# input_df = pd.read_csv('results.csv')


# days_option = st.sidebar.slider('Days Included until current time？',1, 100)
# print(days_option)
gn = GoogleNews()

current_date = datetime.now().date()
start_date = st.sidebar.date_input("Start Date", current_date - timedelta(days=1))  # Set default to a week ago
end_date = st.sidebar.date_input("End Date", current_date)
date_list = pd.date_range(start=start_date, end=end_date, freq='D').date.tolist()

user_input_text = st.sidebar.text_input("Enter keyword:",value='Taiwan',max_chars=15,)

@st.cache_data()
def get_news_in_time(date_list, user_input_text):
    result_df = pd.DataFrame()
    for i in range(len(date_list)-1):
        print('day', date_list[i])
        search = gn.search(user_input_text, from_ = date_list[i], to_ = date_list[i+1])
        entries = search["entries"]
        temp_df = pd.DataFrame(entries)
        result_df = pd.concat([result_df,temp_df])
    result_df['source_link'] = result_df['source'].apply(lambda x: x['href'])
    result_df['source_name'] = result_df['source'].apply(lambda x: x['title'])
    result_df['title_token'] = result_df['title'].apply(tokenize_and_remove_useless)
    result_df['published'] = pd.to_datetime(result_df['published'])
    return result_df.reset_index()
querry_df = get_news_in_time(date_list,user_input_text)   

# nltk.download('punkt')




# @st.cache_data()
# def get_news_from_google(days_option):
#     days_option_text = f'{days_option}d'
#     search = gn.search('Taiwan', when = days_option_text)
#     entries = search["entries"]
#     test_df = pd.DataFrame(entries)
#     test_df['source_link'] = test_df['source'].apply(lambda x: x['href'])
#     test_df['source_name'] = test_df['source'].apply(lambda x: x['title'])
#     test_df['title_token'] = test_df['title'].apply(tokenize_title)
#     test_df['published'] = pd.to_datetime(test_df['published'])
#     return test_df

# @st.cache_data()
# def get_news_from_google_range(day):
#     days_option_text = f'{days_option}d'
#     search = gn.search('Taiwan', when = days_option_text)
#     entries = search["entries"]
#     test_df = pd.DataFrame(entries)
#     test_df['source_link'] = test_df['source'].apply(lambda x: x['href'])
#     test_df['source_name'] = test_df['source'].apply(lambda x: x['title'])
#     test_df['title_token'] = test_df['title'].apply(tokenize_title)
#     test_df['published'] = pd.to_datetime(test_df['published'])
#     return test_df

# querry_df = get_news_from_google(days_option)
outlet_count = pd.DataFrame(querry_df['source_name'].value_counts()).reset_index()
date_count = pd.DataFrame(querry_df['published'].dt.date.value_counts()).reset_index()
# st.dataframe(querry_df)
# st.dataframe(date_count)
num_article = querry_df.shape[0]
st.write(f'Number of Articles: {num_article}')
st.write(f'Due to Google News Query Limit, Only 100 Articles from Each Day is included')
# Page Row 1 
col1, col2 = st.columns(2, gap="small")
fig_outlet = px.pie(outlet_count, values='count', names='source_name', title='News Source Proprotion')
fig_date = px.bar(date_count, y='count', x='published', title='Number of Articles per Day')
# Display the pie chart using Streamlit
with col1:
    st.plotly_chart(fig_outlet)

with col2:
    st.plotly_chart(fig_date)


# Page Row 2 
from collections import Counter

# Assuming 'title_token' is the column containing tokenized words
tokenized_words = querry_df['title_token']

# Combine all lists of tokenized words into a single list
all_words = [word for sublist in tokenized_words for word in sublist]

# Count the occurrences of each word
word_counts = Counter(all_words)

# Get the most common words and their counts
most_common_words = word_counts.most_common(20)  # Change 10 to the desired number of top words

# Print or use the most common words
for word, count in most_common_words:
    print(f'{word}: {count}')


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Create a string from the list of tokens
text = ' '.join(all_words)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black',include_numbers = True).generate(text)

fig = plt.figure(figsize=(10, 5))
# Plot the WordCloud image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig)