import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    global messages
    global categories
    
    mes = pd.read_csv(messages_filepath)
    cat = pd.read_csv(categories_filepath)
    
    messages = pd.DataFrame(mes)
    categories = pd.DataFrame(cat)


def clean_data(df):
    global messages
    global categories
    # merge datasets
    df = messages.merge(categories, on=('id'))
    
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda x: x[:-2]))
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in category_colnames:
        categories[column] = categories.apply(lambda x: x[column][-1:], axis = 1)
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # solving the issue with pd.concat
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(keep=False,inplace=True)
    
    # Dealing with 2's in 'related'
    df.loc[(df.related == '2'),'related'] = list(df['related'].mode())[0]   
    
    return df


def save_data(df, database_filename):
    # save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)  

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')


if __name__ == '__main__':
    main()