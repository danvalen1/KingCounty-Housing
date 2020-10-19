import pandas as pd

def LoadHousingData():
    # Read in targetcsv as Pandas df
    df = pd.read_csv('dsc-phase-2-project/data/kc_house_data.csv')
    
    # Drop unnecessary columns
    
    drop_cols = ['date',
             'view',
             'sqft_above',
             'sqft_basement',
             'yr_renovated', 
             'sqft_living15',
             'sqft_lot15']

    df.drop(columns=drop_cols,inplace=True)
    
    return df