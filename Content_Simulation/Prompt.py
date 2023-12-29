def get_prompt(row):
    date,time,username,inferred_company,likes,caption = row[['date','time','username','inferred company','likes','caption']]

    prompt = f"""
    Generate a tweet that might have been posted:
    📅 Date: {date} , Time: {time}
    👤 Username: {username}
    🏢 Company: {inferred_company}
    👍 Likes: {likes}
    🖼 Media: {caption}
    """
    return prompt