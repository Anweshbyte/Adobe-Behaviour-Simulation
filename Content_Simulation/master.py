import numpy as np
import time
from openai import OpenAI

OPENAI_API_KEY = "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"

client = OpenAI(
    api_key = OPENAI_API_KEY
)

def output(prompt):
    
    messages = [
        {
            "role": "system",
            ### Have to change this
            "content": "Meow is a descriptive agent who is very good at generating tweets given some data about that tweet"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
    )

    message_content = chat_completion.choices[0].message.content
    return message_content

def master_try(dataframe):
    prompt_array = np.array(dataframe['prompt'])
    output_array = []
    processed_rows = 0
    start_time = time.time()

    for i in range(len(prompt_array)):
        output_array.append(output(prompt_array[i]))
        processed_rows += 1

        # Check if 3 rows have been processed within a minute
        if processed_rows >= 3:
            elapsed_time = time.time() - start_time

            # If less than 60 seconds elapsed, sleep for the remaining time
            if elapsed_time < 60:
                time.sleep(60 - elapsed_time)
            
            # Reset processed_rows counter and start time for the next batch
            processed_rows = 0
            start_time = time.time()

    dataframe['content'] = output_array
    return dataframe