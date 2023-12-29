import re
import emoji
import numpy as np
import pandas as pd
from sklearn import preprocessing
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize  # You might need to install nltk: pip install nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('vader_lexicon')



def wordvec(word):
  tokenized_sentence = word_tokenize(word.lower())
  model = Word2Vec(sentences=[tokenized_sentence], vector_size=100, window=5, min_count=1, workers=4)
  word_vectors = [model.wv[word] for word in tokenized_sentence]

  return word_vectors[0]

label_encoder = preprocessing.LabelEncoder()

def remove_emojis(text):
    def emoji_to_text(match):
        emoji_str = match.group()
        try:
            # Convert emoji to text
            emoji_text = emoji.demojize(emoji_str)
            # Remove underscore and colons
            emoji_text = emoji_text.replace("_", " ").replace(":", " ")
            return emoji_text
        except:
            return ""

    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(emoji_to_text, text)

def get_counts(df):
  hyperlink = "<hyperlink>"
  df["hyperlink_count"] = df['content'].apply(lambda x: x.split().count(hyperlink))
  mention = "<mention>"
  df['mention_count'] = df['content'].apply(lambda x: x.split().count(mention))
  df['hashtag_count'] = df['content'].apply(lambda x: sum([1 for word in x.split() if word.startswith('#')]))
  df['content'] = df['content'].str.replace('<mention>', '')
  df['content'] = df['content'].str.replace('<hyperlink>', '')
  df['content'] = df['content'].str.replace('<mention>', '')
  df['content'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('#')]))
  return df

def get_media_urls(df):
  df['image_url'] = df['media'].str.extract(r"fullUrl='(.*?)'")
  df['video_url'] = df['media'].apply(lambda x: re.findall(r"url='(.*?)'", x)[1] if 'variants' in x and len(re.findall(r"url='(.*?)'", x)) > 1 else None)
  df['thumbnail_url'] = df['media'].str.extract(r"thumbnailUrl='(.*?)'")
  df['duration'] = df['media'].apply(lambda x: float(re.search(r"duration=([\d.]+)", x).group(1)) if 'duration' in x else 0)
  df['views'] = df['media'].apply(lambda x: int(re.search(r"views=(\d+)", x).group(1)) if re.search(r"views=(\d+)", x) else 0)
  df["word_count"] = df['content'].apply(lambda x: sum([1 for word in x.split()]))

  return df

def get_link(row):
  if isinstance(row['image_url'],str):
    return row['image_url']
  else:
    return row['thumbnail_url']

# Function to create sine and cosine embeddings for time and date
def create_time_embeddings(df, column_name):

    seconds_in_day = 24 * 60 * 60
    seconds_in_year = 365.25 * 24 * 60 * 60

    # Calculate normalized values
    seconds_of_day = (df[column_name].dt.hour * 3600 + df[column_name].dt.minute * 60 + df[column_name].dt.second) / seconds_in_day
    seconds_of_year = (df[column_name] - df[column_name].min()).dt.total_seconds() / seconds_in_year

    # Create embeddings
    df['day_sin'] = np.sin(2 * np.pi * seconds_of_day)
    df['day_cos'] = np.cos(2 * np.pi * seconds_of_day)
    df['year_sin'] = np.sin(2 * np.pi * seconds_of_year)
    df['year_cos'] = np.cos(2 * np.pi * seconds_of_year)
    df['week_sin'] = np.sin((df['day']*2*np.pi)/7)
    df['week_cos'] = np.cos((df['day']*2*np.pi)/7)
    return df



def uptime(date):
  ref = '2023-11-21 00:00:00'
  refdate = pd.to_datetime(ref)
  reftime = refdate.timestamp()
  return reftime - date.timestamp()
def get_uptime(df):
  df['uptime'] = df['datetime'].apply(uptime)
  return df

def getweekday(date):
  return date.weekday()

def get_weekday_df(df):
  df['datetime'] = pd.to_datetime(df['date'])
  df['day'] = df['datetime'].apply(getweekday)
  return df

def master_cleaner(df):
  # Initialize the VADER sentiment analyzer
  sid = SentimentIntensityAnalyzer()
  df['sentiment'] = df['content'].apply(lambda x: 1 if sid.polarity_scores(x)['compound'] > 0 else (0 if sid.polarity_scores(x)['compound'] == 0 else -1))
  df = df.dropna(subset=['content'])
  df = get_counts(df)
  df['content'] = df['content'].apply(remove_emojis)
  df = get_media_urls(df)
  df['url'] = df.apply(get_link,axis = 1)
  df['user_encoding']= label_encoder.fit_transform(df['username'])
  df = get_weekday_df(df)
  df['inferred_company'] = df['inferred company']
  df = create_time_embeddings(df,'datetime')
  df = get_uptime(df)
  df['company_embedd'] = df['inferred_company'].apply(wordvec)
  
  return df
  # return df[['id', 'content', 'username', 'inferred_company','mention_count',  'hashtag_count','url','image_url','thumbnail_url','video_url', 'day_sin', 'day_cos','year_sin', 'year_cos', 'uptime', 'week_sin', 'week_cos','views','duration']]