# Retweet Predictor
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
As online social media networks have become a major platform for sharing one’s opinions, there is a growing need for building an accurate predictive model for information diffusion. For instance, companies/influencers/politicians hope to know how viral their new message will be.

#### Summary of this project
Previous models have included sentiment analysis as a part of features, yet most of them relied on general-purpose dictionaries that classify sentiment simply as positive or negative. The idea of utilizing a domain-specific dictionary for sentiment analysis has been accepted and widely used in other domains such as financial prediction. Empirical evidence helps specify a few candidates for domain-specific dictionaries. Firstly, moral-emotional expression is reported to be highly correlated with information dissemination of moralized content (moral contagion). Further studies identified specific types of moral-emotions relevant to the moral contagion effect: outrage and fear. 12 prediction models were created for performance comparison: models based on 4 machine learning algorithms (logistic regression, k nearest neighbor, random forests, and XGBoost) for each type of 3 sentiment scores (the positive-negative score, the moral-emotional score, and the outrage-fear score). The results showed that models with domain-specific emotion scores, moral-emotional or outrage-fear, showed stronger performance than a model with a positive-negative sentiment score. However, the difference was too small to make a definitive claim about the importance of domain-specific emotions. Feature importances highlight the significance of sentiment scores in each prediction model. Therefore, future work on this topic would be helpful in terms of advancing our knowledge on predicting virality of moralized content in online social contexts.
