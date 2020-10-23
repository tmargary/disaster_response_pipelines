# Udacity Project 2: Disaster Response Pipeline

## ETL
For the first part of the project, I have conducted the Extract, Transform, and Load process. Here, I read the dataset, clean the data, and then store it in a `SQLite` database. The cleaning was done with `pandas`.
To load the data into an `SQLite` database, I have used the `pandas` dataframe `.to_sql()` method, which I used with an `SQLAlchemy engine`.

## Machine Learning Pipeline
Next, I have split the data into a training set and a test set and built a machine learning pipeline.
The pipeline uses `NLTK`, as well as scikit-learn's `Pipeline` and `GridSearchCV` to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, I exported the model to a `pickle` file.

## Flask App
In the last step, I have displayed my results in a `Flask` web app using the workspace provided by Udacity.

## Deployment
The final result of the project is a website that shows visualizations, as well as classifies a message inputted by the user. The following screenshots shows the results of multi-output classification:</br></br>
![](https://github.com/tmargary/disaster_response_pipelines/blob/master/deployment_screenshots/2.png)</br></br>

## Deployment Visualisations:</br></br>
![](https://github.com/tmargary/disaster_response_pipelines/blob/master/deployment_screenshots/3.png)

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to the website.

## Resources
- **Mentorship:** Udacity team. Special thanks to Rajat P.<br/>
- **Python Version:** 3.8<br/>
- **Packages:** pandas, numpy, sklearn, sqlalchemy, plotly, NLTK, joblib, flask<br/>
- **Data Source:** Figure Eight<br/>
