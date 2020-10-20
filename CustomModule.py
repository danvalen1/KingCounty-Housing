import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.formula.api import ols

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

# Setting global variables

figsize = (10,8)
fontscale = 1.3
sns.set(font_scale = fontscale, style = 'whitegrid')

labels_dict = {'id': 'House ID',
             'price': 'Sale Price ($)',
             'bedrooms': 'Number of Bedrooms',
             'bathrooms': 'Number of Bathrooms',
             'sqft_living': 'Square Footage of Living Area',
             'sqft_lot': 'Square Footage of Land Lot',
             'floors': 'Number of Floors',
             'waterfront': 'Is on Waterfront?',
             'condition': 'Condition of House',
             'grade': 'Grade of House Given by King County',
             'yr_built': 'Year House Built',
             'zipcode': 'Zipcode',
             'lat': 'Latitude',
             'long': 'Longitude',
              'date': 'Date of Sale',
             'view': 'Has Been Viewed?',
             'sqft_above': 'Square Footage of House Apart from Basement',
             'sqft_basement': 'Square Footage of Basement',
             'yr_renovated': 'Year When Renovated', 
             'sqft_living15': 'Square Footage of Living Area for Neighbors',
             'sqft_lot15': 'Square Footage of Land Lot for Neighbors',
             'cost': 'Cost of Building House ($)'
              }

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
    
    # If waterfront is in variables replace NaN with 0s since low number of waterfront properties
    if df['waterfront'].any():
        df['waterfront'].fillna(0, inplace=True)
        
    # Generate cost columns assuming 
    df['cost'] = [x*153 for x in df['sqft_living']]
    
    return df


def PlotScatter(df, xvar, yvar):
    title = f'{labels_dict[yvar]} v. {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize)
                           
    sns.scatterplot(x=xvar,
                y=yvar,
                data=df)
    
    ax.set(title=f'{title}',
          xlabel=labels_dict[xvar],
          ylabel=labels_dict[yvar]
          )
    
    fig.savefig(f'{title}.png', bbox_inches='tight')
                           
    return plt.show()
    
def PlotHist(df, xvar):
    title = f'Frequency of {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data=df,
                x=xvar)
    
    ax.set(title=title,
          xlabel=labels_dict[xvar],
          ylabel='Frequency'
          )
    fig.savefig(f'{title}.png', bbox_inches='tight')
    return plt.show()

def BaselineModel(df, y, xlist):
    X_train = df[xlist]
    y_train = df[y]
    dummy = DummyRegressor()  # by default this will use the mean

    dummy.fit(X_train, y_train)
    
    y_pred = dummy.predict(X_train)

    score = dummy.score(X_train, y_train) # the score of a regression model is the r-squared value
    
    dummy_rmse = mean_squared_error(y_train, y_pred, squared=False)
    
    return print(f'R-squared = {score}',
                 '\n',
                 f'RMSE = {dummy_rmse}')