# css_project_2022
 Code for the Computational Social Science project

## How to use the notebook
- install the requirements
- download the dataset from https://www.kaggle.com/datasets/lingshuhu/political-partisanship-tweets, extract `cong_politician_tweets_2020-3-12-2021-5-28_text_party_balanced_anonymous.csv` to the root of this repo and rename it to `political_tweets.csv`
- run the notebook

## How to use the prediction script
- download the preferred model from https://mega.nz/folder/FAtFmIgB#J1_FE9AxZ6HG1TrVjkfNRA and place them in the root folder of this repo, or use the notebook to fit the models and create them locally
- start the script with `python predict.py -mode [1, 2] -model_type [mnb, sgd]`
- both the arguments are optional, by default they are 1 for 'mode' and 'mnb' for 'model_type'
- follow the instructions and receive a prediction for the tweet's political orientation

## Notes
- for the 'mode' argument, type 1 if you want to paste in the script the text of a tweet (default) or type 2 to input the link to a tweet
- for the 'model_type' argument, type 'mnb' to use the Multinomial Naive Bayes model (default) or type 'sgd' to use the SGD-trained model