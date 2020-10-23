import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import statsmodels.api as sm

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from yellowbrick.regressor import ResidualsPlot

# Setting global variables

dpi = 300
figsize = (5, 4)
fontscale = .8
sns.set(font_scale = fontscale, style = 'whitegrid')
markersize = 75


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
             'sqft_per_occupant': 'Square Footage of Living Area per Bedroom'
              }



def LoadHousingData(varlist, clean=False):
    """Load the King County housing data into a pandas data frame.
        Specify the variables of interest passing a list of strings as `varlist`. 
        Specify whether raw data is used by passing a boolean value through `clean`. 
        A True value means that the data preparation used for our analysis is used. Note that when clean=True a dictionary of dataframe is returned, not a dataframe itself.
    """

    # Read in targetcsv as Pandas df
    df = pd.read_csv('dsc-phase-2-project/data/kc_house_data.csv')
    
    # Drop unnecessary columns
    df = df[varlist]
    
    if clean == True:
        
        # Generate QOL variable
        df['sqft_per_occupant'] = df['sqft_living'] / df['bedrooms']
        df['space_x_grade'] = df['sqft_living'] * df['grade']
    
        # Dummy vars for grade excluding lowest grade
        df_dummies = pd.get_dummies(df.grade).iloc[:,1:]

        # Combine grade dummy vars with df
        df = pd.concat([df,df_dummies], axis = 1)
        
        # Dummy vars for sqft_lot
        bins = [0,8000, 40000, 500000]

        bin_names = ['urban', 'suburban', 'rural']

        df['sqft_lot_transform'] = pd.cut(df['sqft_lot'], bins, labels = bin_names)

        lot_dummies = pd.get_dummies(df.sqft_lot_transform).iloc[:,:2]

        df = pd.concat([df, lot_dummies], axis = 1)
        

        # Train-test split
        train_set, test_set = train_test_split(df, test_size = .2, random_state = 5)
                
        train_set = train_set.drop([
                      'sqft_living', 
                      'grade', 
                      'sqft_lot', 
                      'space_x_grade',
                        'sqft_lot_transform'], axis=1)
        
        test_set = test_set.drop([
                      'sqft_living', 
                      'grade', 
                      'sqft_lot', 
                      'space_x_grade',
        'sqft_lot_transform'], axis=1)


        split_dfs = {'df': df, 'train_set': train_set, 'test_set': test_set} 

        return split_dfs
    
    else:
        # Train-test split
        train_set, test_set = train_test_split(df, test_size = .2, random_state = 5)
        split_dfs = {'df': df, 'train_set': train_set, 'test_set': test_set} 
        return split_dfs

def PlotScatter(df, xvar, yvar, hue=None):
    """Plots a scatter of `xvar` and `yvar` in df. A hue can be added through `hue`. 
    """
    title = f'{labels_dict[yvar]} v. {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                      
    scatter = sns.scatterplot(x=xvar,
                    y=yvar,
                    data=df,
                    hue=hue,
                    palette="Spectral",
                    s=markersize,
                    alpha = .3
                   )
    if hue:
        ax.legend([hue])
        ax.legend(markerscale=1.5)
        title = f'{labels_dict[yvar]} v. {labels_dict[xvar]}\n Colored By {hue.title()} Category'
    
    ax.set(title=f'{title}',
          xlabel=labels_dict[xvar],
          ylabel=labels_dict[yvar]
          )
    
    
        
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    if xvar=='sqft_lot':
        plt.xticks(rotation=-45)
    
    fig.savefig(f'images/{title}.png', bbox_inches='tight')
                        
    return plt.show()
    
def PlotHist(df, xvar, bins):
    """Plot a histogram with automatic labels provided in a global dict in CustomModule. 
        Pass a dataframe through `df`, a string through `xvar`, and the number of bins through `bins`.
    """
    title = f'Frequency of {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
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

def BaselineModel(df, y):
    """Uses a dataframe with dependent and independent variables (y) to generate
    R-squared and RMSE values for a dummy regression using scikit-learn
    """
    X_train = df.drop(labels=y, axis=1)
    y_train = df[y]
    dummy = DummyRegressor()  # by default this will use the mean

    dummy.fit(X_train, y_train)
    
    y_pred = dummy.predict(X_train)

    score = dummy.score(X_train, y_train) # the score of a regression model is the r-squared value
    
    dummy_rmse = mean_squared_error(y_train, y_pred, squared=False)
    
    return print(f'R-squared = {score}',
                 '\n',
                 f'RMSE = {dummy_rmse}')


def Models(df, y):
    """Uses a dataframe with dependent and independent variables (y) to generate
    R-squared and RMSE values for a linear regression using scikit-learn
    """
    X_train = df.drop(labels=y, axis=1)
    y_train = df[y]
    model = LinearRegression()  # by default this will use the mean

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)

    score = model.score(X_train, y_train) # the score of a regression model is the r-squared value
    
    model_rmse = mean_squared_error(y_train, y_pred, squared=False)
    
    return print(f'R-squared = {score}',
                 '\n',
                 f'RMSE = {model_rmse}')


def Linreg(df):
    """Uses a dataframe with dependent and independent variable `price` to generate
    R-squared and RMSE values for a linear regression using statsmodels.
    """
    X = df.drop(labels='price', axis=1)
    y = df.price
    
    X = sm.tools.tools.add_constant(X)
    king_model = sm.OLS(y, X).fit()
    
    return king_model.summary()

def CorrHeatmap(df):
    """Generates a correlation heatmap from a dataframe with variables of interest.
    """
    fig, ax = plt.subplots(figsize=(10,8), dpi=150)
    sns.heatmap(df.corr(), cmap='bwr', center=0, annot=True)
    fig.savefig(f'images/CorrHeatmap.png', bbox_inches='tight')
    return fig.show()
    
def Resid(df):
    """Generates a residual plot using a dictionary containing a test set and training set.
    """
    fig, ax = plt.subplots(figsize=(10,8), dpi=150)

    X_train = df['train_set'].drop(labels='price', axis=1)
    y_train = df['train_set'].price
    
    X_test = df['test_set'].drop(labels='price', axis=1)
    y_test = df['test_set'].price
    
    
    
    
    # Instantiate the linear model and visualizer
    model = LinearRegression(fit_intercept=True)
    

    visualizer = ResidualsPlot(model)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    
    
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.xticks(rotation=-45)
    
    fig.savefig(f'images/Residuals.png', dpi=dpi, bbox_inches='tight')
    return visualizer.show()
