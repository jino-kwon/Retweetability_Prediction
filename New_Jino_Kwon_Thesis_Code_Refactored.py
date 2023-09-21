# Standard libraries
import pandas as pd
import numpy as np
import nltk
import re
from numpy import inf

# Third-party libraries
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

# Constants
DATA_PATH = "/content/drive/My Drive/Colab Notebooks/thesis/"
PUNCTUATION = '''!"@#$%&'()*+,-./:;<=>?[\]^_`{|}~'''


def authenticate_google():
    """Authenticates the user with Google."""
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    return GoogleDrive(gauth)


def load_data(drive):
    """Loads the data from the Google Drive."""
    df = pd.read_csv(DATA_PATH + "senator_all.csv")
    return df


def preprocess_data(df):
    """Processes and cleans the data."""
    # Drop unnecessary columns and rename others
    columns_to_drop = ["rowname", "created_at", "favorite_count", "elite_idx", "dw_extr", 
                       "dwextr_rs", "AffectCount", "MoralCount", "neg_uniqueCount", 
                       "pos_uniqueCount", "shared_negCount", "shared_posCount"]
    df = df.drop(columns_to_drop, axis=1)
    df.rename(columns={"elite_new": "senator", "dwscore.y": "dwscore", 
                       "genderfx":"gender", "shared":"moral-emo"}, inplace=True)

    # Recode categorical values: '1' means 'yes' and '0' means 'no'
    for col in ['url', 'media', 'gender']:
        df[col] += 0.5

    # Clean text
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    df['clean_text'] = df['text'].str.lower().apply(clean_lemma, args=(lemmatizer, stop_words))

    # Add retweetable column
    df["retweetable"] = df["retweet_count"].apply(retweetable)
    
    return df


def clean_lemma(text, lemmatizer, stop_words):
    """Lemmatizes and cleans the text."""
    text = text.replace("\n", " ")
    for char in PUNCTUATION:
        text = text.replace(char, " ")
    
    while "  " in text:
        text = text.replace("  ", " ")
    
    words = [lemmatizer.lemmatize(w) for w in word_tokenize(text)]
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)


def retweetable(count):
    """Determines if a tweet is retweetable."""
    return 1 if count >= 6 else 0


def sentiment_analysis(df):
    """Performs sentiment analysis on the dataframe."""
    filepath = DATA_PATH + "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    nrc_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], sep='\t')
    nrc_df = nrc_df[nrc_df.association != 0]

    emotions = {
        'outrage': ['anger', 'disgust'],
        'fear': ['fear'],
        'positive': ['positive'],
        'negative': ['negative']
    }

    for emotion in emotions:
        df[emotion] = df.clean_text.apply(lambda x: sentiment_score(x, emotion, nrc_df, emotions))
    
    df['token_num'] = df.clean_text.apply(lambda x: len(word_tokenize(x)))
    df['moral-emo'] = df.apply(lambda x: x['moral-emo']/x['token_num'], axis=1)
    df = df.drop(["token_num"], axis=1)
    
    return df


def sentiment_score(txt, emotion, nrc_df, emotions):
    """Scores the sentiment of a text based on the given emotion."""
    token = set(word_tokenize(txt))
    emotion_words = set(nrc_df[nrc_df.emotion.isin(emotions[emotion])]['word'])
    emotion_count = len(token.intersection(emotion_words))
    return emotion_count / len(token)


def main():
    drive = authenticate_google()
    df = load_data(drive)
    df = preprocess_data(df)
    df = sentiment_analysis(df)
    
    # Additional preprocessing for exploratory analysis
    df['ln_followers'] = np.log(df['followers'])
    df['ln_retweet'] = np.log(df['retweet_count'])
    df = df.replace({'ln_retweet': {-inf: 0}})
    
    df.to_csv(DATA_PATH + 'preprocessed_data.csv')


if __name__ == "__main__":
    main()


---------------------------------------------------------------------------
# This would be in a separate file
"""# Building Prediction Models"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import xgboost as xgb
import warnings

# Supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Load the dataset
df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/thesis/preprocessed_data.csv")

# Visualize correlations using a heatmap
def visualize_correlations(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.2g', cbar_kws={'orientation': 'horizontal'})
    plt.show()

features = df[['retweetable', 'posneg', 'moral-emo', 'outrage', 'fear', 'followers', 'dwscore', 'url', 'media', 'gender']]
visualize_correlations(features)

# Print count of zero sentiment rows
print("POSITIVE-NEGATIVE SENTIMENT:", df[df['posneg'] == 0].shape[0])
print("MORAL-EMOTIONAL SENTIMENT:", df[df['moral-emo'] == 0].shape[0])
print("OUTRAGE AND FEAR SENTIMENT:", df[(df['outrage'] == 0) & (df['fear'] == 0)].shape[0])

# Plot histograms for certain columns
df.hist(column='retweet_count')
df.hist(column='followers')

# Display basic statistics
subset_df = df[['retweet_count', 'retweetable', 'posneg', 'moral-emo', 'fear', 'outrage', 'dwscore', 'url', 'media', 'gender', 'followers']]
descriptive_stats = subset_df.describe().transpose()
print(descriptive_stats)

"""## ML models with pos/neg"""

# train test split
X = df[['url', 'media', 'followers', 'dwscore', 'gender', 'posneg']]
y = df['retweetable']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""* Penalizaed Logistic Regression Model w/ L2 Penalty"""

logreg_pipe1 = make_pipeline(LogisticRegression(penalty='l2'))
logreg_param_grid1 = {'logisticregression__C': np.linspace(1, 100, 100)}
logreg_grid1 = GridSearchCV(logreg_pipe1, logreg_param_grid1, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("LOGISTIC REGRESSION W/ L2 Penalty (SCALED DATA)")
print("Best Parameter: {}".format(logreg_grid1.best_params_))
print("Test set Score: {:.4f}".format(logreg_grid1.score(X_test_scaled, y_test)))

y_pred = np.array(logreg_grid1.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* KNN Classifier"""

knn_pipe1 = make_pipeline(KNeighborsClassifier())
knn_param_grid1 = {'kneighborsclassifier__n_neighbors': range(1, 30)}
knn_grid1 = GridSearchCV(knn_pipe1, knn_param_grid1, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("KNN CLASSIFER (SCALED DATA)")
print("Best Parameter: {}".format(knn_grid1.best_params_))
print("Test set Score: {:.4f}".format(knn_grid1.score(X_test_scaled, y_test)))

y_pred = np.array(knn_grid1.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* Random Forest Classifier"""

rfc_pipe1 = make_pipeline(RandomForestClassifier(random_state=42))
rfc_param_grid1 = {'randomforestclassifier__n_estimators': [100, 500, 1000],
                  'randomforestclassifier__max_depth': [6, 7, 8]}
rfc_grid1 = GridSearchCV(rfc_pipe1, rfc_param_grid1, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("RANDOM FOREST CLASSIFIER")
print("Best Parameter: {}".format(rfc_grid1.best_params_))
print("Test set Score: {:.4f}".format(rfc_grid1.score(X_test_scaled, y_test)))

y_pred = np.array(rfc_grid1.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* XGBoost Model"""

xgb_pipe1 = make_pipeline(xgb.XGBClassifier())
xgb_param_grid1 = {'xgbclassifier__max_depth': [6, 7, 8]}
xgb_grid1 = GridSearchCV(xgb_pipe1, xgb_param_grid1, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("XGBOOST CLASSIFIER")
print("Best Parameter: {}".format(xgb_grid1.best_params_))
print("Test set Score: {:.4f}".format(xgb_grid1.score(X_test_scaled, y_test)))

y_pred = np.array(xgb_grid1.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

# And, the feature importance function:
def plot_feature_importance(importance, names):
  #Create arrays from feature importance and feature names
  feature_importance = np.array(importance)
  feature_names = np.array(names)

  #Create a DataFrame using a Dictionary
  data={'feature_names':feature_names,'feature_importance':feature_importance}
  fi_df = pd.DataFrame(data)

  #Sort the DataFrame in order decreasing feature importance
  fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
  print(fi_df)

  #Define size of bar plot
  plt.figure(figsize=(10,8))
  #Plot Searborn bar chart
  ax = sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], palette="Blues_r")

  for p in ax.patches:
    ax.annotate("%.4f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),
            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
  #Add chart labels
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')

# A feature importance plot for a XG Boost model
# fitting the model with best parameters
xgb1 = xgb.XGBClassifier(max_depth=7, importance_type='weight')
xgb1.fit(X_train_scaled, y_train)
print(xgb1.feature_importances_)
plot_feature_importance(xgb1.feature_importances_, X.columns)

"""## ML models with moral-emotions"""

# train test split
X = df[['url', 'media', 'followers', 'dwscore', 'gender', 'moral-emo']]
y = df['retweetable']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""* Penalizaed Logistic Regression Model w/ L2 penalty"""

logreg_pipe2 = make_pipeline(LogisticRegression(penalty='l2'))
logreg_param_grid2 = {'logisticregression__C': np.linspace(1, 100, 100)}
logreg_grid2 = GridSearchCV(logreg_pipe2, logreg_param_grid2, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("LOGISTIC REGRESSION W/ L2 Penalty (SCALED DATA)")
print("Best Parameter: {}".format(logreg_grid2.best_params_))
print("Test set Score: {:.4f}".format(logreg_grid2.score(X_test_scaled, y_test)))

y_pred = np.array(logreg_grid2.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* KNN Classifier"""

knn_pipe2 = make_pipeline(KNeighborsClassifier())
knn_param_grid2 = {'kneighborsclassifier__n_neighbors': range(1, 30)}
knn_grid2 = GridSearchCV(knn_pipe2, knn_param_grid2, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("KNN CLASSIFER (SCALED DATA)")
print("Best Parameter: {}".format(knn_grid2.best_params_))
print("Test set Score: {:.4f}".format(knn_grid2.score(X_test_scaled, y_test)))

y_pred = np.array(knn_grid2.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* Random Forest Classifier"""

rfc_pipe2 = make_pipeline(RandomForestClassifier(random_state=42))
rfc_param_grid2 = {'randomforestclassifier__n_estimators': [100, 500, 1000],
                  'randomforestclassifier__max_depth': [6, 7, 8]}
rfc_grid2 = GridSearchCV(rfc_pipe2, rfc_param_grid2, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("RANDOM FOREST CLASSIFIER")
print("Best Parameter: {}".format(rfc_grid2.best_params_))
print("Test set Score: {:.4f}".format(rfc_grid2.score(X_test_scaled, y_test)))

y_pred = np.array(rfc_grid2.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* XGBoost Model"""

xgb_pipe2 = make_pipeline(xgb.XGBClassifier())
xgb_param_grid2 = {'xgbclassifier__max_depth': [6, 7, 8]}
xgb_grid2 = GridSearchCV(xgb_pipe2, xgb_param_grid2, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("XGBOOST CLASSIFIER")
print("Best Parameter: {}".format(xgb_grid2.best_params_))
print("Test set Score: {:.4f}".format(xgb_grid2.score(X_test_scaled, y_test)))

y_pred = np.array(xgb_grid2.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""### Create a Feature Importance Plot for the Best Model : XGBoost"""

# A feature importance plot for a XG Boost model
# fitting the model with best parameters
xgb2 = xgb.XGBClassifier(max_depth=7, importance_type='weight')
xgb2.fit(X_train_scaled, y_train)

plot_feature_importance(xgb2.feature_importances_, X.columns)

"""## ML models with outrage and fear"""

# train test split
X = df[['url', 'media', 'followers', 'dwscore', 'gender', 'outrage', 'fear']]
y = df['retweetable']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""* Penalizaed Logistic Regression Model w/ L2 penalty"""

logreg_pipe3 = make_pipeline(LogisticRegression(penalty='l2'))
logreg_param_grid3 = {'logisticregression__C': np.linspace(1, 100, 100)}
logreg_grid3 = GridSearchCV(logreg_pipe3, logreg_param_grid3, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("LOGISTIC REGRESSION W/ L2 Penalty (SCALED DATA)")
print("Best Parameter: {}".format(logreg_grid3.best_params_))
print("Test set Score: {:.4f}".format(logreg_grid3.score(X_test_scaled, y_test)))

y_pred = np.array(logreg_grid3.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* KNN Classifier"""

knn_pipe3 = make_pipeline(KNeighborsClassifier())
knn_param_grid3 = {'kneighborsclassifier__n_neighbors': range(1, 30)}
knn_grid3 = GridSearchCV(knn_pipe3, knn_param_grid3, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("KNN CLASSIFER (SCALED DATA)")
print("Best Parameter: {}".format(knn_grid3.best_params_))
print("Test set Score: {:.4f}".format(knn_grid3.score(X_test_scaled, y_test)))

y_pred = np.array(knn_grid3.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* Random Forest Classifier"""

rfc_pipe3 = make_pipeline(RandomForestClassifier(random_state=42))
rfc_param_grid3 = {'randomforestclassifier__n_estimators': [100, 500, 1000],
                  'randomforestclassifier__max_depth': [6, 7, 8]}
rfc_grid3 = GridSearchCV(rfc_pipe3, rfc_param_grid3, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("RANDOM FOREST CLASSIFIER")
print("Best Parameter: {}".format(rfc_grid3.best_params_))
print("Test set Score: {:.4f}".format(rfc_grid3.score(X_test_scaled, y_test)))

y_pred = np.array(rfc_grid3.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""* XGBoost Model"""

xgb_pipe3 = make_pipeline(xgb.XGBClassifier())
xgb_param_grid3 = {'xgbclassifier__max_depth': [6, 7, 8]}
xgb_grid3 = GridSearchCV(xgb_pipe3, xgb_param_grid3, cv=kfold, scoring='roc_auc').fit(X_train_scaled, y_train)

print("XGBOOST CLASSIFIER")
print("Best Parameter: {}".format(xgb_grid3.best_params_))
print("Test set Score: {:.4f}".format(xgb_grid3.score(X_test_scaled, y_test)))

y_pred = np.array(xgb_grid3.predict(X_test_scaled))
print(classification_report(y_test, y_pred))

"""### Create a Feature Importance Plot for the Best Model"""

# A feature importance plot for a XG Boost model
# fitting the model with best parameters
xgb3 = xgb.XGBClassifier(max_depth=7, importance_type='weight')
xgb3.fit(X_train_scaled, y_train)

plot_feature_importance(xgb3.feature_importances_, X.columns)

"""# Visualization of General Performance Comparison"""

# peformance results
pos_neg = [0.6996, 0.8113, 0.8310, 0.8376]
moral_emotion = [0.7071, 0.8183, 0.8285, 0.8343]
outrage_fear = [0.7082, 0.8092, 0.8310, 0.8393]

labels = ['Logistic Regression', 'k-NN Classifier', 'Random Forest', 'XGBoost']
bar_width = 0.3 # set the bar width

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10,8))
r1 = np.arange(len(pos_neg))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# make the plot
row1 = ax.bar(r1, pos_neg, color='lightsteelblue', width=bar_width, edgecolor='white', label='pos-neg')
row2 = ax.bar(r2, moral_emotion, color='cornflowerblue', width=bar_width, edgecolor='white', label='moral-emotion')
row3 = ax.bar(r3, outrage_fear, color='midnightblue', width=bar_width, edgecolor='white', label='outrage-fear')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance (AUC)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rows):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for r in rows:
        height = r.get_height()
        ax.annotate('{}'.format(height),
                    xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(row1)
autolabel(row2)
autolabel(row3)

plt.show()
