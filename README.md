# Intro to NLP using SciKit Learn and Python
### Can we predict the points range, price range and grape variety of a wine from a wine experts description?
A project to introduce you to a simple Bag of Words NLP using SciKit Learn and Python. No previous machine learning knowledge required! You can use this same logic for document classification or any text classification problem you may be trying to solve.

## Prerequisites
There are different ways to follow along on this tutorial, however the most simple option would probably be option 2!

1. Create an [Azure account](https://azure.microsoft.com/en-us/free/) and [Create Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-run-cloud-notebook?WT.mc_id=github-blog-casiljan) to use the Notebook VMs. This gives you a LOT of functionality and I would highly recommend this for models you plan to put in production.
2. [Azure Notebooks](https://notebooks.azure.com/) - an online Jupyter notebook that makes it easy to share and access your notebook from anywhere.
3. [Download Jupyter](https://jupyter.org/) notebooks and run it locally. Additionally you will need [Anaconda](https://www.anaconda.com/distribution/) or Python installed to run the notebook locally.
4. Lastly you can run a Jupyter notebook kernal directly in VS Code. If you prefer to read through the tutorial and then run it in VS Code. [Download the source from GitHub](https://github.com/cassieview/intro-nlp-wine-reviews) and run the individual cells in the file `wine-prediction-train.py`. This python file does NOT have the detailed instructions that the Jupyter notebook has and this tutorial has. It has all the code in a script format vs notebook tutorial format.

Once you are set with one of the above notebook environment configurations its time to start building!

## Import packages and data
### 1. Import the Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load
```
NOTE: If you get an error "No module named" install it with the command `!pip install joblib`. Replace `joblib` with the module name in the error message.

### 2. We need data!
![data](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSwvLv12Qt9SOXvdGwlqQP0ORHhvO1OI7hAxqAvXbf3tpRl4t2Isw)
1. I used a dataset I found on Kaggle. Kaggle is an online community of data scientists. 
2. Download the dataset from this repo or kaggle.
* [Wine Dataset from Repo](dataset/winemag-review.csv)
* [Kaggle Dataset](https://www.kaggle.com/zynicide/wine-reviews)

3. Import the data as a [Pandas](https://pandas.pydata.org/pandas-docs/stable/) DataFrame
```python
#File path to the csv file
csv_file = r"C:\path-to-your-file\winemag-review.csv"

# Read csv file into dataframe
df = pd.read_csv(csv_file)

# Print first 5 rows in the dataframe
df.head()
```

![graph](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/dfhead.PNG)

## Visualize the data
Once we have the data then its time to analyze it and do some [Feature Selection and Engineering](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/create-features?WT.mc_id=github-blog-casiljan). We will visualize our data using [Seaborn](https://seaborn.pydata.org/). This will allow us to see if there is a strong correlation between different data points and help us answer questions about our data. Since our initial question was around predicting `price`, `points` or `variety` from the `description` we already know that our `feature` will be the `description` and our `label` will be `price`, `points`or `variety`. Each prediction label will be a separate model so there will be three models in total if you build one for each label.

For fun, lets ask some questions about the data and answer them by graphing it with Seaborn.

### 1. Is there a correlation between price and points?
```python
sns.barplot(x = 'points', y = 'price', data = df)
```
![graph](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/priceandpoints.PNG)

```python
sns.boxplot(x = 'points', y = 'price', data = df)
```
![graph](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/priceandpoints2.PNG)

### 2. Does one wine critic give higher ratings than the others?

```python
sns.catplot(x = 'points', y = 'taster_name', data = df)
```
![graph](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/tasterpoints.PNG)

### 3. Lets look at a WordCloud of the `description` Text

```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
text = df.description.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
```
![graph](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/wordcloud.PNG)
<small>I like to think of this WordCloud as a cheatsheet of discriptive words to use when tasting wine to make yourself sound like a wine expert :D</small>


### What other questions could you ask and answer by graphing this data?
![graph](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/dfhead.PNG)

## Create Calculated Columns for Labels

We are going to do a multi-classification for the price and points of the wines reviewed by the wine critics. Right now our points and price are a number features. We are going to create a couple functions to generate calculated columns based on the values in the points and price columns to use as are our labels.

If we wanted to predict a specific price or point value we would want to build a regression model not a multi-classification. It really just depends on what your goal is. We are looking at classification in this tutorial so we want to convert them to text features for classification.

### 1. Create `quality` column from points values to classes of bad, ok, good, great. Below is a function to return string `quality` based on the points value.

```python
def getQuality(points):
    if(points <= 85):
        return 'bad'
    elif(points<=90 ):
        return 'ok'
    elif(points<=95):
        return 'good'
    elif(points<=100):
        return 'great'
    else:
        return 'If this gets hit, we did something wrong!'
```

### 2. Next lets apply the function to the points column of the dataframe and add a new column named `quality`.

```python
df['quality'] = df['points'].apply(getQuality)
```
### 3. Lets visualize our new column against the price column like we did above.

```python
sns.catplot(x = 'quality', y = 'price', data = df)
```
![graph](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/pricequality.PNG)


```python
sns.barplot(x = 'quality', y = 'price', data = df)
```
![graph](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/pricequality2.PNG)

we now have quality buckets based on the points to use as a label class for our multi-classification model.

### 1. Create priceRange column from price column of `1-30`, `31-50`, `51-100`, `Above 100` and `0` for columns with NaN. Below is a function to return string priceRange based on price value.

```python
def getPriceRange(price):
    if(price <= 30):
        return '1-30'
    elif(price<=50):
        return '31-50'
    elif(price<=100): 
        return '51-100'
    elif(math.isnan(price)):
        return '0'
    else:
        return 'Above 100'
```
### 2. Next lets apply the function to the points column of the dataframe and add a new column named `priceRange`.

```python
df['priceRange'] = df['price'].apply(getPriceRange)
```

### 3. Print totals for each priceRange assigned to see how the labels are distributed

```python
df.groupby(df['priceRange']).size()

Output: priceRange
        0             8996
        1-30         73455
        31-50        27746
        51-100       16408
        Above 100     3366
        dtype: int64
```
## We now have our labels for  models to predict quality, priceRange and grape variety. Next we need to take our description text and process NLP with the library SciKitLearn to create a Bag-of-Words using the [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) functionality.

The docs do a great job of explaining the CountVectorizer. I recommend reading through them to get a full understanding of whats going on, however I will go over some of the basics here.

At a high level the CountVectorizer is taking the text of the description, removing stop words (such as “the”, “a”, “an”, “in”), creating a tokenization of the words and then creating a vector of numbers that represents the description. The text description is now represented as numbers with only the words we care about and can be processed by the computer to train a model. Remember the computer understand numbers and words can be represented as numbers so the computer can "understand".

Before we jump into the CountVectorizer code and functionality. I want to list out some terms and point out that CountVectorizer _does not_ do the Lemmatization or Stemming for you.
 
* StopWords:  A stopword can be a word with meaning in a specific language. For example, in the English language, words such as "a," "and," "is," and "the" are left out of the full-text index since they are known to be useless to a search. A stopword can also be a token that does not have linguistic meaning.
* [N-Gram](https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/glossary#n-gram?WT.mc_id=github-blog-casiljan): A feature extraction scheme for text data: any sequence of N words turns into a feature value.
![ngram](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/ngram.PNG)
* [Lemmatization](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/preprocess-text#module-overview?WT.mc_id=github-blog-casiljan): converts multiple related words to a single canonical form ("fruity", "fruitiness" and "fruits" would all become "fruit")
* Stemming: Similar to Lemmatization but a bit more aggressive and can leave words fragmented.

### Lets take a look at how we do this now.

These are all the properties that you can set within the CountVectorizer. Many of them are defaulted or if set override other parts of the CountVectorizer. We are going to leave most of the defaults and then play with changing some of them to get better results for our model.

```python
CountVectorizer(input=’content’, encoding=’utf-8’, decode_error=’strict’, strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>)
```
## Create the function to get the vector and vectorizer from the `description` feature.

### 1. There are different CountVectorizer configurations commented out so that we can play with different configs and see how it changes our result. Additionally this will help us look at one description and pick apart what is actually happening in the CountVectorizer.

```python
def get_vector_feature_matrix(description):
    vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=5)
    #vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    #vectorizer = CountVectorizer(lowercase=True, stop_words="english",ngram_range=(1, 2), max_features=20)

    #vectorizer = CountVectorizer(lowercase=True, stop_words="english", tokenizer=stemming_tokenizer) 
    vector = vectorizer.fit_transform(np.array(description))
    return vector, vectorizer
```

### 2. For the first run we are going to have the below config. What this is saying is that we want to convert the text to lowercase, remove the english stopwords and we only want 5 words as feature tokens.

```python
vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=5)
```
```python
#Optional: remove any rows with NaN values. 
#df = df.dropna()
```
### 3. Next lets call our function and pass in the description column from the dataframe. 

This returns the `vector` and the `vectorizer`. The `vectorizer` is what we apply to our text to create the number `vector` representation of our text so that the machine learning model can learn.

```python
vector, vectorizer = get_vector_feature_matrix(df['description'])
```
If we print the vectorizer we can see the current default parameters for it.

```python
print(vectorizer)

Output: CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=5, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words='english',
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)
```
### 4. Lets examine our variables and data to understand whats happening here.

```python
print(vectorizer.get_feature_names())
Output:
['aromas', 'flavors', 'fruit', 'palate', 'wine']
```
Here we are getting the features of the vectorizer. Because we told the CountVectorizer to have a `max_feature = 5` it will build a vocabulary that only consider the top feature words ordered by term frequency across the corpus. This means that our `description` vectors would _only_ include these words when they are tokenized, all the other words would be ignored.

Lets print out our first `description` and first `vector` to see this represented.

```python
print(vector.toarray()[0])
Output: [1 0 1 1 0]
```

```python
df['description'].iloc[0]
Output: "_Aromas_ include tropical _fruit_, broom, brimstone and dried herb. The _palate_ isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity."
```

The vector array (`[1 0 1 1 0]`) that represents the vectorization features (`['aromas', 'flavors', 'fruit', 'palate', 'wine']`) in first description in the corpus. 1 indicates its present and 0 indicates not present in the order of the vectorization features.

Play around with different indexes of the vector and description. You will notice that there isn't lemmatization so words like `fruity` and `fruits` are being ignored since only `fruit` is included in the vector and we didn't lemmatize the description to transform them into their root word.

## Time to Train the Model

## 1. Update the function so that the second vectorizer configuration is being used.

```python
def get_vector_feature_matrix(description):
   def get_vector_feature_matrix(description):
    #vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=5)
    vectorizer = CountVectorizer(lowercase=True, stop_words="english", max_features=5000)
    #vectorizer = CountVectorizer(lowercase=True, stop_words="english",ngram_range=(1, 2), max_features=5000)
    #vectorizer = CountVectorizer(lowercase=True, stop_words="english", tokenizer=stemming_tokenizer) 
    vector = vectorizer.fit_transform(np.array(description))
    return vector, vectorizer
```

And call the function to update the vectorizer

```python
vector, vectorizer = get_vector_feature_matrix(df['description'])
```

Now create our feature matrix

```python
features = vector.todense()
```
We have three different labels for three different models. Lets assign the label variable next and use the `quality` label first.

```python
label = df['quality'] 
#label = df['priceRange']
#label = df['variety']
```
## 2. We have the features and label variables created. Next we need to split the data to train and test. 

We are going to use 80% to train and 20% to test. This will allow us to get an accuracy estimation from the training to see how the model is performing.

```python
X, y = features, label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## 3. Train the model using a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) algorithm.

```python
lr = LogisticRegression(multi_class='ovr',solver='lbfgs')
model = lr.fit(X_train, y_train)
```
Lets check the accuracy!

```python
accuracy = model.score(X_test, y_test)
print ("Accuracy is {}".format(accuracy))
Output: "Accuracy is 0.7404885554914407"
```

### 4. Time to test the model

When you select a candidate model it should always be tested on unseen data. If a model is [overfitted](https://en.wikipedia.org/wiki/Overfitting) to its data it will perform really will on its own data and poorly on new data. This is why its very important to test on unseen data.

I grabbed this review from the wine mag site. Its a 96 points and $60 bottle of wine review.

```python
test = "This comes from the producer's coolest estate near the town of Freestone. White pepper jumps from the glass alongside accents of lavender, rose and spice. Compelling in every way, it offers juicy raspberry fruit that's focused, pure and undeniably delicious."


x = vectorizer.transform(np.array([test]))
proba = model.predict_proba(x)
classes = model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)
```

Lets take a look at the results
```python
resultdf
```
![testresult](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/testresult.PNG)

Another way to look at the result is to transpose, sort and then print the head resulting in a list of the top 5 predictions.

```python
topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
topPrediction.head()
```
# This is a correct prediction! 🎉 
However I am sure there are ways to improve this model and accuracy. Play around and see if you can get a better result!

## Other things to try
1. Change the label and run again for the price bucket prediction or grape variety.
2. Try to use different algorithms to see if you can get a better result
3. Add additional features to the description text to improve accuracy. There was a strong correlation between price and points. Maybe adding those would improve the accuracy score?
4. Add lemmatization to the text to improve score using the [NLTK](https://www.nltk.org/)
5. Try doing a text classification on a different dataset.

Remember: Data science is a trial and error process. Keep thinking of ways to improve the model!

## Other helpful links
[The Machine Learning Algorithm Cheat Sheet](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice#the-machine-learning-algorithm-cheat-sheet?WT.mc_id=github-blog-casiljan)

[How to choose algorithms for Azure Machine Learning Studio](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice?WT.mc_id=github-blog-casiljan)

