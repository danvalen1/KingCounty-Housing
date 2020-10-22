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
figsize = (20,16)
fontscale = 1.75
sns.set(font_scale = fontscale, style = 'whitegrid')
markersize = 150


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

        df['sqft_lot'] = pd.cut(df['sqft_lot'], bins, labels = bin_names)

        lot_dummies = pd.get_dummies(df.sqft_lot).iloc[:,:2]

        df = pd.concat([df, lot_dummies], axis = 1)
        

        # Train-test split
        train_set, test_set = train_test_split(df, test_size = .2, random_state = 5)
                
        train_set = train_set.drop(['date', 
                      'bathrooms', 
                      'sqft_living', 
                      'grade', 
                      'sqft_lot', 
                      'space_x_grade', 
                      'yr_built'], axis=1)
        
        test_set = test_set.drop(['date', 
                      'bathrooms', 
                      'sqft_living', 
                      'grade', 
                      'sqft_lot', 
                      'space_x_grade', 
                      'yr_built'], axis=1)


        split_dfs = {'df': df, 'train_set': train_set, 'test_set': test_set} 

        return split_dfs
    
    else:
        return df

def PlotScatter(df, xvar, yvar, hue=None):
    title = f'{labels_dict[yvar]} v. {labels_dict[xvar]}'
    
    fig, ax = plt.subplots(figsize=figsize)
                      
    scatter = sns.scatterplot(x=xvar,
                    y=yvar,
                    data=df,
                    hue=hue,
                    palette="Spectral",
                    s=markersize
                   )
    
    ax.set(title=f'{title}',
          xlabel=labels_dict[xvar],
          ylabel=labels_dict[yvar]
          )
    if hue:
        ax.legend(markerscale=3)
        
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

def BaselineModel(df, y):
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
    X = df.drop(labels='price', axis=1)
    y = df.price
    
    X = sm.tools.tools.add_constant(X)
    king_model = sm.OLS(y, X).fit()
    
    plt.rc('figure', figsize=(12, 7))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 0.05, str(king_model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/OLSLinReg.png', bbox_inches='tight')
    return king_model.summary()

def CorrHeatmap(df):
    fig, ax = plt.subplots(figsize=(40,32))
    sns.heatmap(df.corr(), cmap='bwr', center=0, annot=True)
    fig.savefig(f'images/CorrHeatmap.png', bbox_inches='tight')
    return fig.show()
    
def Resid(df):
    fig, ax = plt.subplots(figsize=(10,8), dpi=dpi)

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
