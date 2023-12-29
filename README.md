# Behaviour Simulation Challenge

**Behaviour Simulation** | **Content Simulation** <br>
&emsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GXOhJ7rDmb-6ijpZzxTroug7I_ACbiNE?usp=sharing) &emsp; &nbsp;&emsp;  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Zulk3BocFkqu1xTUbQwcvH7NqhYplHeZ?usp=sharing) 


## Description

### Behaviour Simulation

<p align="center">
    <img src="media/photo1.png" alt="Image 1" style="width:40%; ">    
</p>

The task at hand involved predicting user engagement, specifically the number of likes, for tweets. The prediction was based on various factors, including the content of the tweet (text), the associated company, the username of the account posting the tweet, any media URLs included, and the timestamp of the tweet. This problem required the development of a predictive model that can effectively analyze these features to estimate the likely number of likes a tweet will receive


### Content Simulation

<p align="center">
    <img src="media/photo2.png" alt="Image 2" style="width:40%;">   
</p>

The problem statement revolves around the task of reconstructing or generating the textual content of a tweet based solely on its metadata. The metadata includes information such as the associated company, username, media URL, and timestamp. In essence, the challenge is to infer the message or information conveyed in the tweet without having access to the actual text.

## Contents
This repository contains the following :-
1. Colab Notebooks for both the tasks. Instructions for which is mentioned in the notebook itself. 
2. [results](results/) folder containing result :-
    - [behaviour_simulation_test_company.xlsx](/results/behaviour_simulation_test_company.xlsx)
    - [behaviour_simulation_test_time.xlsx](/results/behaviour_simulation_test_time.xlsx)
    - [content_simulation_test_company.xlsx](/results/content_simulation_test_company.xlsx)
    - [content_simulation_test_time.xlsx](/results/content_simulation_test_time.xlsx)
3. [Report](report.pdf) for the problem statement. 
4. [Experiments](Experiments/) folder, containing notebooks for all the approaches undertaken by us.
5. [Behaviour_Simulation](Behaviour_Simulation/) folder, containing the code for running our Behaviour Simulation model on custom dataset.
6. [Content_Simulation](Content_Simulation/) folder, containing the code for running our Content Simulation model on custom dataset.
7. [Metadata_features_and_captions](Metadata_features_and_captions/) folder, containing the excel file with the features that we have extracted for our pipeline.

## Instructions
For the instructions on how to run the models, check out the ReadMe in the [Behaviour_Simulation](Behaviour_Simulation/) and [Content_Simulation](Content_Simulation/) folder. If for any reason the code does not run, please go to the colab instances :-
- [Colab instance for Behaviour Simulation Task](https://colab.research.google.com/drive/1GXOhJ7rDmb-6ijpZzxTroug7I_ACbiNE?usp=sharing) 
- [Colab instance for Content Simulation Task](https://colab.research.google.com/drive/1Zulk3BocFkqu1xTUbQwcvH7NqhYplHeZ?usp=sharing)

## Approach
### Task 1 : Behaviour Simulation

<p align="center">
    <img src = "media\Task1.jpg" height="300" alt="Image">
  <br>
  <em>Approach for Task 1</em>
</p>

Our approach for behaviour simulation starts by dividing the inputs into four categories :-
- Company
- Image
- Tweet Content
- Tweet Metadata like sentiment,date,time,mention_count,hyperlink_count and so on.

1. For the company we have have embedded the company name using Word2Vec
2. For the image and the tweet content we have used CLIP to get 768 dimnesional embedding for each. These are processed through a dense layer.
3. For the meta-deta we have a 13 dimensional array.

These three are than concatenated and given as input to a series of dense layers that give the final likes as output.

### Task 2 : Content Simulation

<p align="center">
    <img src = "media\Task2.jpg" height="300" alt="Image">
  <br>
  <em>Approach for Task 2</em>
</p>

For the content simulation task, we have followed the underlying procedure :-
1. For the tweet media we have used captioning models
2. Then using these captions and the tweet meta deta we have engineered a prompt and then feeded it to an LLM for tweet generation

After experimenting with various models we decided to go with fastdup-blip model for captions and open-ai gpt-3.5-turbo for the tweet generation.

<!-- ![Approach for Task 1](media\Task1.jpg)
![Approach for Task 2](media\Task2.jpg) -->
