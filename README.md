# Intro to NLP - Predict the Quality and Price of a Wine from a Wine Experts Description. Then deploy the model via Azure Functions.
A project to introduce you to a simple Bag of Words NLP using SciKit Learns and Python. You can use this same logic for document classification or any text classification problem you may be trying to solve.

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
```

### The first step for any model building: We need Data!
![data](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSwvLv12Qt9SOXvdGwlqQP0ORHhvO1OI7hAxqAvXbf3tpRl4t2Isw)
1. I used a dataset I found on Kaggle. Kaggle is an online community of data scientists. 
2. Download the dataset from this repo or kaggle.
* [Wine Dataset from Repo](https://github.com/cassieview/intro-nlp-wine-reviews/blob/master/dataset/winemag-review.csv)
* [Kaggle Dataset](https://www.kaggle.com/zynicide/wine-reviews)

3. Import the data as [Pandas](https://pandas.pydata.org/pandas-docs/stable/) DataFrame
```python
#File path to the csv file
csv_file = r"C:\path-to-your-file\winemag-review.csv"

# Read csv file into dataframe
df = pd.read_csv(csv_file)

# Print first 5 rows in the dataframe
df.head()
```
### Visualize the data
1. Once we have the data its time to take a look at it and do some [Feature Selection and Engineering](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/create-features)


## Azure Tools and Data
### Create Resource in Azure
1. Go to [Azure Portal](https://portal.azure.com/) and login or [Create an Account](https://azure.microsoft.com/en-us/free/)
2. Click "Create resource"
3. Select "AI + Machine Learning" then "Machine Learning service workspace"
4. Fill in required fields and select "Review + Create" then select "Create"
 </br> ![createamlresource][createamlresource]
5. It will take a few minutes to create the resources needed for your workspace. Below is a list of all the resources that are created:
</br> ![workspaceresourcelist][workspaceresourcelist]
