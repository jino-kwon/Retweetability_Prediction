# Retweetability Predictor
#### *This work was submitted in partial fulfillment of Columbia University Master's Program*
Original Thesis Title : The Use of Domain-Specific Sentiment Analysis on Predicting Information Diffusion in Online Social Networks

#### Short Summary
In this project, I built machine learning models (*logistic regression, k nearest neighbor, random forests, and XGBoost*) including domain-specific *sentiment analysis* that predict whether a tweet will be retweeted or not, with the highest AUC score of *84%*.

#### Previous Work
This project was inspired by previous research projects I was involved in.
If interested, here are some research findings that I presented at national social science conferences:
1. March 2018, [the Society for Personality and Social Psychology Annual Meeting](https://github.com/jino-kwon/Sentiment_Analysis_For_Predicting_Info_Diffusion/blob/master/Jino%202018%20SPSP%20poster.pdf)
2. May 2019, [the Association for Psychological Science Annual Convention](https://github.com/jino-kwon/Sentiment_Analysis_For_Predicting_Info_Diffusion/blob/master/Jino%202019%20APS%20poster.pdf)

#### Motivation
As online social media networks have become a major platform for sharing one’s opinions, there is a growing need for building an accurate predictive model for information diffusion. Taking insights from social science research (See the *previous work* for more details), I hoped to build a prediction model that incorporates a number of social factors associated with sharing *political content* in online social networks.

In addition, in continuation of my previous work regarding domain-specific sentiment analysis, I compared the performances between models employing different dictionaries:

1. General, binary sentiment-based model ( positive or negative words )
2. Moral-emotional sentiment-based model ( moral-emotional words such as *peace* and *punish* )
3. Outrage-fear sentiment-based model

#### Data
The scraped tweets were posted by all 100 U.S. Senators during the year leading up to the 2016 U.S. election: from November 2015 to October 2016. (n = 99,750)

#### Features
- The number of followers
- URL and Media Attachment
- Political ideology scores
- Sentiment Analysis
- Gender
- Social support

#### Target
The main task was to classify whether a tweet is retweetable or not.
In defining what a retweetable message is, I referred to the median retweet counts in the dataset, which was 6.
Thus, Twitter messages retweeted 6 times or more were classified as 'retweetable', and the other were categorized as 'not retweetable'.

#### Heatmap for the Correlation Matrix for All Features
![alt text](https://github.com/jino-kwon/Retweet_Prediction_Models/blob/master/images/heatmap.jpg)

#### Feature Importance Plot for the Best Model
*importance type='weight'*
![alt text](https://github.com/jino-kwon/Retweet_Prediction_Models/blob/master/images/feature_importance.jpg)

#### Performance
![alt text](https://github.com/jino-kwon/Retweet_Prediction_Models/blob/master/images/performance.jpg)
