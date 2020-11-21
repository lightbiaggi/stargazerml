from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pdfrom sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required

nasa-stars = TabularDatasetFactory.from_delimited_files("https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey?select=Skyserver_SQL2_27_2018+6_51_39+PM.csv", validate=True, include_path=False, infer_column_types=True, set_column_types=None, separator=',', header=True, partition_format=None, support_multi_line=False, empty_as_string=False, encoding='utf8')

df = nasa-stars.to_pandas_dataframe()
df.head()
df.describe()
df.info()

x = df.drop(['class'], axis=1)
y = df['class']

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)



run = Run.get_context()



def main():
       # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--c', type=string, default="criterion")
    parser.add_argument('--max_depth', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--min_samples_split', type=int, default=1)
    parser.add_argument('--n_estimators', type=int, default=1)
    args = parser.parse_args()

    
    clf = RandomForestClassifier()
    clf.set_params(criterion = args.c, max_features = None, max_depth = args.max_depth, min_samples_leaf = args.min_samples_leaf, min_samples_split = args.min_samples_split, n_estimators = args.n_estimators)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    f1score = f1_score(y_test, y_pred, average = None)
    run.log("Accuracy:",accuracy_score(y_test, preds))
    run.log("f1_score:",f1_score)
if __name__ == '__main__':
    main()
