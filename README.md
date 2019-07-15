# Intro to NLP - Predict the Quality and Price of a Wine from a Wine Experts Description. Then deploy the model via Azure Functions.
A project to introduce you to a simple Bag of Words NLP using SciKit Learns and Python. You can use this same logic for document classification or any text classification problem you may be trying to solve.

## Prerequisites
There are a few different ways to follow along on this tutorial:
1. Create an [Azure account](https://azure.microsoft.com/en-us/free/) and [Create Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-run-cloud-notebook) to use the Notebook VMs. This gives you a LOT of functionality and I would highly recommend this for models you plan to put in production.
2. [Azure Notebooks](https://notebooks.azure.com/) - an online Jupyter notebook that makes it easy to share and access your notebook from anywhere.
3. [Download Jupyter](https://jupyter.org/) notebooks and run it locally. The notebook is included in the source for this tutorial.

Once you are set with one of the above notebook environment configurations its time to start building!

## Import packages and data
### Import the Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
```

### We need Data!
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
## Visualize the data
Once we have the data then its time to analyize it and do some [Feature Selection and Engineering](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/create-features). We will visualize our data using [Seaborn](https://seaborn.pydata.org/). This will allow us to see if there is a strong correlation between different data points and help us answer questions about our data. Since our initial question was around predicting `price` or `points` from the `description` we already know that our `feature` will be the `description` and our `label` will be `price` or `points`. 

For fun, lets ask some questions about the data and answer them by graphing it with Seaborn.

### 1. Is there a correlation between price and points?
```python
sns.barplot(x = 'points', y = 'price', data = df)
```
![graph](\assests\priceandpoints.PNG)

```python
sns.boxplot(x = 'points', y = 'price', data = df)
```
![graph](\assests\priceandpoints2.PNG)

### 2. Does one wine critic give higher ratings than the others?

```python
sns.catplot(x = 'points', y = 'taster_name', data = df)
```
TODO ADD GRAPH HERE

What other questions could you ask and answer by graphing this data?

## Azure Tools and Data
### Create Resource in Azure
1. Go to [Azure Portal](https://portal.azure.com/) and login or [Create an Account](https://azure.microsoft.com/en-us/free/)
2. Click "Create resource"
3. Select "AI + Machine Learning" then "Machine Learning service workspace"
4. Fill in required fields and select "Review + Create" then select "Create"
 </br> ![createamlresource][createamlresource]
5. It will take a few minutes to create the resources needed for your workspace. Below is a list of all the resources that are created:
</br> ![workspaceresourcelist][workspaceresourcelist]
