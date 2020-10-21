import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from statsmodels.formula.api import ols

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setting global variables

figsize = (20,16)
fontscale = 2
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
             'cost': 'Cost of Building House ($)',
             'space_x_grade': 'Grade per sq ft',
             'sqft_per_occupant': 'Square Footage per Bedroom'
              }



def LoadHousingData(varlist):

    # Read in targetcsv as Pandas df
    df = pd.read_csv('dsc-phase-2-project/data/kc_house_data.csv')
    
    # Drop unnecessary columns
    df = df[varlist]
    
    # Generate QOL variable
    df['sqft_per_occupant'] = df['sqft_living'] / df['bedrooms']
    df['space_x_grade'] = df['sqft_living'] * df['grade']
    
    # Dummy vars for grade excluding lowest grade
    df_dummies = pd.get_dummies(df.grade).iloc[:,1:]
    
    # Combine dummy vars with df
    df = pd.concat([df,df_dummies], axis = 1)
    
    # Train-test split
    train_set, test_set = train_test_split(df, test_size = .2, random_state = 5)
    
    split_dfs = {'df': df, 'train_set': train_set, 'test_set': test_set}
    
    return split_dfs

def PlotScatter(df, xvar, yvar, hue=None):
    title = f'{labels_dict[yvar]} v. {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize)
                      
    sns.scatterplot(x=xvar,
                y=yvar,
                data=df,
                   hue=hue,
           palette="Spectral")
    
    ax.set(title=f'{title}',
          xlabel=labels_dict[xvar],
          ylabel=labels_dict[yvar]
          )
    

    
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    fig.savefig(f'images/{title}.png', bbox_inches='tight')
                        
    return plt.show()
    
def PlotHist(df, xvar, bins):
    title = f'Frequency of {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data=df,
                x=xvar,
                bins=bins)
    
    ax.set(title=title,
          xlabel=labels_dict[xvar],
          ylabel='Frequency'
          )
    
    
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.xticks(rotation=-45)
    
    fig.savefig(f'images/{title}.png', bbox_inches='tight')
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


def Linreg(df, y, xlist):
    formula = f'{y}~' + "+".join(xlist)
    king_model = ols(formula=formula, data = df).fit()
    return king_model.summary()

def CorrHeatmap(df):
    fig = sns.heatmap(df.corr(), cmap='bwr', center=0, annot=True)
    fig.savefig(f'images/CorrHeatmap.png', bbox_inches='tight')
    return fig.show()
    