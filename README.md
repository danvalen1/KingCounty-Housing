# Predicting Housing Prices for King County Housing Authority
**Authors**: Eon Slemp, Dan Valenzuela

## Repository Structure

```
├── README.md                           <- The top-level README for reviewers of this project
├── Project_Walkthrough.ipynb           <- Narrative documentation of analysis in Jupyter notebook
├── KingCounty-Housing_Presentation.pdf <- PDF version of project presentation
├── CustomLibrary.py                    <- Module of function used in analysis
├── images                              <- Both sourced externally and generated from code
├── notebooks                           <- Noteboooks used to build Walkthrough
└── dsc-phase-2-project                 <- Templates referenced in building project, including kc_house_data.csv
```


## Overview

This project analyzes housing prices in King County, Washington for houses sold between May 2014 and May 2015. A model was created to predict housing prices so that King County Housing Authority can better understand how much their subsidies for housing may cost given features of a house. Specifically, this project used the data to predict price on homes based on their build density (e.g., whether they were urban or suburban), the number of people they could house, and the quality of life they can offer.


## Business Problem

Affordability of housing in King County and surrounding areas has been progressively [declining](https://www.huduser.gov/portal/publications/pdf/SeattleWA-CHMA-19.pdf) since 2012. The King County Housing Authority has been tasked with understanding the housing market in order to effectively intervene. The purpose of this project is to help the Authority predict housing prices based on characteristics of homes they would prefer to subsidize. In effect, the model would allow the Authority to predict further how much a subsidy program may cost. 

With respect to the Authority's priority in understanding homes they would prefer to subsidize, the questions this project attempts to answer are:

  1. What is the relationship between a home's price and the number of people it can house?

  2. What is the relationship between a home's price and the quality of life it can offer?
  
  3. What is the relationship between a home's price and the size of the lot it is on?

## Data Understanding

The dataset used in this project is from [kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction). It contains data about 27 thousand home sales in King County between May 2014 and May 2015.

For the purposes of this analysis, the variables of interest are `price`, `bedrooms`, `sqft_lot`, `sqft_living`, and `grade` with `price` as the target variable and the rest being features. Given the business problem and questions set out, `bedrooms` acts as a measure how many people a house can hold; `sqft_lot` measures how densely the house can be built given that it measures how large the lot the house is built on; and `grade` and `sqft_living` help measure the quality of living of the house.

## Data Preparation

No transformations or changes were applied to `price` or `bedrooms`. However, `sqft_lot`, `grade`, and `sqft_living` had a number of operations performed to them. `sqft_lot` was binned into `urban` and `suburban` with `rural` as the reference category. Dummy variables were constructed for `grade`. And `sqft_living` was divided by `bedrooms` to approximate how much space is available per occupant.

After having transformed the variables, the dataset was split into a test set and a train set with the test set 20% of the whole dataset as seen below.

## Results
Our final model shows that it is possible to account for 57.2\% of the variance in `price` without too much of an impact on adj. R-squared because of the additional variables created. By dividing `sqft_living` by `bedrooms`, creating dummy variables for `grade`, and binning `sqft_lot` into density bins, one can use the model to predict `price` with an RMSE of 241,262.

However, this model also shows that the `grade` categories may be less statistically-significant than previously thought. This will require more investigation as a scatter plot of this model of `price` v. `sqft_per_occupant` including `grade` as a hue shows that there appears to be a relationship with `grade` that this model does not account for.

Generally the model shows that for each additional bedroom the price of a house increases by approximately \$75,000; that for each additional square foot of living space per occupant increases the price by approximately $500; and that urban houses are almost twice as expensive as suburban houses.

## Evaluation
The model produced on the training set does almost just as well with the test set with R-squared values being approximately equal and residuals being similar. However, looking at the residuals against the predicted values, one sees that the variance in residuals increases as `price` increases, meaning this model is heteroskedastic. 

Further, one sees that the errors are approximately normally distributed, meaning at least one of the assumptions of linear regression is met.

    
## Conclusion
The pricing model developed here is able to predict price of homes in King County, Washington using bins of `sqft_lot`, `bedrooms` alone, and the constructed `sqft_per_occupant` and account for approximately 57\% of the variability in `price`. However, using dummy variables of `grade` did not provide statistically significant predictions of `price`. This means that our model can do a better job of modeling the effect of quality of life of a house on its price. 

## Next Steps
Thihis model can be improved by providing additional transformations to the `grade` variable. As you can see in the graph below, grade seems to have a correlation with price at every  grade level and that relationship will need to be teased out.

Further, the features of this model can be scaled to better compare the comparative effect of the variables on `price`. 

![graph](/images/Sale Price ($) v. Square Footage of Living Area per Bedroom
 Colored By Grade Category.png)
