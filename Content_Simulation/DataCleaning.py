import pandas as pd

def get_media_urls(df):
  df['image_url'] = df['media'].str.extract(r"fullUrl='(.*?)'")
  import re
  df['video_url'] = df['media'].apply(lambda x: re.findall(r"url='(.*?)'", x)[1] if 'variants' in x and len(re.findall(r"url='(.*?)'", x)) > 1 else None)

  df['thumbnail_url'] = df['media'].str.extract(r"thumbnailUrl='(.*?)'")

  return df

def get_link(row):
  if isinstance(row['image_url'],str):
    return row['image_url']
  else:
    return row['thumbnail_url']

def get_date_time(df):
  df['datetime'] = pd.to_datetime(df['date'])

  # Create 'date' and 'time' columns
  df['date'] = df['datetime'].dt.date
  df['time'] = df['datetime'].dt.time

  return df

def getweekday(date):
  return date.weekday()

def get_weekday_df(df):
  df['datetime'] = pd.to_datetime(df['date'])
  df['day'] = df['datetime'].apply(getweekday)
  return df

def master_clean(df):
  df = get_date_time(df)
  df = get_media_urls(df)
  df['url'] = df.apply(get_link,axis = 1)
  df['inferred_company'] = df['inferred company']
  df = get_weekday_df(df)

  return df[['id','date','time','likes', 'username', 'inferred_company','video_url','image_url','thumbnail_url','url']]