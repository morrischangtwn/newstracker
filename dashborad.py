import time

import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px
# from pygooglenews import GoogleNews
import json
import time
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta
from GoogleNews import GoogleNews
googlenews = GoogleNews()

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

nltk.download('punkt')
nltk.download('stopwords')


print('-------')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Function to tokenize and filter out useless tokens
def tokenize_and_remove_useless(title):
    # Tokenize the title into words
    tokens = word_tokenize(title)
    
    # Remove stop words and short tokens
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and len(token) > 2 and token.lower() != user_input_text.lower()]
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

# date_list = [date.strftime('%Y-%m-%d') for date in date_list]

user_input_text = st.sidebar.text_input("Enter keyword:",value='Taiwan',max_chars=15,)

def remove_items(row):
    return [token for token in row['title_token'] if token != row['media'].lower()]

@st.cache_data()
def get_news_in_time(date_list, user_input_text):
    result_df = pd.DataFrame()
    for i in range(len(date_list)-1):
        print('day', date_list[i],date_list[i+1])
        print('day', type(date_list[i]), type(date_list[i+1]))
        googlenews = GoogleNews(start=date_list[i],end= date_list[i+1])
        googlenews.get_news(user_input_text)
        temp_df = pd.DataFrame((googlenews.results()))
        result_df = pd.concat([result_df,temp_df])
    result_df['title'] = result_df['title'].str.replace(r'More', ' ')
    result_df['title_token'] = result_df['title'].apply(tokenize_and_remove_useless)
    result_df['datetime'] = pd.to_datetime(result_df['datetime'])
    result_df['title_token'] = result_df.apply(remove_items, axis=1)

    return result_df.reset_index()
querry_df = get_news_in_time(date_list,user_input_text)   

outlet_count = pd.DataFrame(querry_df['media'].value_counts()).reset_index()
date_count = pd.DataFrame(querry_df['datetime'].dt.date.value_counts()).reset_index()
# st.dataframe(querry_df)
# st.dataframe(date_count)
num_article = querry_df.shape[0]
st.write(f'Number of Articles: {num_article}')
st.write(f'Due to Google News Query Limit, Only 100 Articles from Each Day is included')
# Page Row 1 

fig_outlet = px.pie(outlet_count, values='count', names='media', title='News Source Proprotion')
fig_date = px.bar(date_count, y='count', x='datetime', title='Number of Articles per Day')
# Display the pie chart using Streamlit


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
font = r'TaipeiSansTCBeta-Regular.ttf'
# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black',include_numbers = True, font_path=font, margin=0).generate(text)

fig = plt.figure(figsize=(10, 5))
# Plot the WordCloud image
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis('off')



# Overall Sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()
def analyze_sentiment_vader(text):
    scores = sia.polarity_scores(text)
    return scores['compound'], scores['pos'], scores['neu'], scores['neg']
querry_df[['compound', 'positive', 'neutral', 'negative']] = querry_df['title'].apply(lambda x: pd.Series(analyze_sentiment_vader(x)))
def label_sentiment(input):
    if input < 0:
        return 'negative'
    elif input == 0.0:
        return 'netural'
    elif input >0:
        return 'positive'
    
querry_df['sent_label'] = querry_df['compound'].apply(label_sentiment)

fig_sent_date = px.histogram(querry_df, x='datetime', y='compound', template='plotly', title='Sentiment (Compound) Change Overtime',      labels={
        'datetime': 'Date', 
        'compound': 'Compound Sentiment Score'
    }
)

fig_sent = px.pie(querry_df, 
            #  values='sent_label', 
             names='sent_label',
            #  title='Sentiment Analysis Distribution',)
             color_discrete_sequence=['#00CC96','#636EFA','#EE553B'])
col1, col2, col3 = st.columns(3, gap="small")
with col1:
    st.plotly_chart(fig_outlet)

with col2:
    st.plotly_chart(fig_date)

with col3:
    st.plotly_chart(fig_sent)

col1_2, col2_2 = st.columns(2, gap="small")

with col1_2:
    st.pyplot(fig)
with col2_2:
    st.plotly_chart(fig_sent_date)


st.dataframe(querry_df)
