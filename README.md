# Udacity Project 2: Disaster Response Pipeline

## ETL
For the first part of the project I have condicted the Extract, Transform, and Load process. Here, I read the dataset, clean the data, and then store it in a SQLite database. The cleaning was done with pandas.
To load the data into an SQLite database, I have used the pandas dataframe .to_sql() method, which I used with an SQLAlchemy engine.

## Machine Learning Pipeline
Next, I have splitted the data into a training set and a test set and built a machine learning pipeline.
The pipeline uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification).
Finally, I exported the model to a pickle file.

## Flask App
In the last step, I have displayed my results in a Flask web app using the workspace provided by Udacity.
