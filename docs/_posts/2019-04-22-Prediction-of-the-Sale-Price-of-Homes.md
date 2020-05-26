---
layout: post
title: Prediction of the Sale Price of Homes
date: 2019-04-22
mathjax: true
---


This notebook will attempt to predict the sale price of homes based on a number of different features. This comes from the "Knowledge" section of Kaggle, so it is not a direct competition, but instead is a nice entry level project.


```python
# First import everything
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
pd.options.display.max_columns = 999
```

## Data Collection
Data comes pre-compiled directly from Kaggle.


```python
fullData = pd.read_csv('train.csv')
fullData.head()
```
<style type="text/css">
	.table-wrapper {
		overflow-x: scroll;
	}
</style>

<div class="table-wrapper" markdown="block">

|    |   Id |   MSSubClass | MSZoning   |   LotFrontage |   LotArea | Street   |   Alley | LotShape   | LandContour   | Utilities   | LotConfig   | LandSlope   | Neighborhood   | Condition1   | Condition2   | BldgType   | HouseStyle   |   OverallQual |   OverallCond |   YearBuilt |   YearRemodAdd | RoofStyle   | RoofMatl   | Exterior1st   | Exterior2nd   | MasVnrType   |   MasVnrArea | ExterQual   | ExterCond   | Foundation   | BsmtQual   | BsmtCond   | BsmtExposure   | BsmtFinType1   |   BsmtFinSF1 | BsmtFinType2   |   BsmtFinSF2 |   BsmtUnfSF |   TotalBsmtSF | Heating   | HeatingQC   | CentralAir   | Electrical   |   1stFlrSF |   2ndFlrSF |   LowQualFinSF |   GrLivArea |   BsmtFullBath |   BsmtHalfBath |   FullBath |   HalfBath |   BedroomAbvGr |   KitchenAbvGr | KitchenQual   |   TotRmsAbvGrd | Functional   |   Fireplaces | FireplaceQu   | GarageType   |   GarageYrBlt | GarageFinish   |   GarageCars |   GarageArea | GarageQual   | GarageCond   | PavedDrive   |   WoodDeckSF |   OpenPorchSF |   EnclosedPorch |   3SsnPorch |   ScreenPorch |   PoolArea |   PoolQC |   Fence |   MiscFeature |   MiscVal |   MoSold |   YrSold | SaleType   | SaleCondition   |   SalePrice |
|----|------|--------------|------------|---------------|-----------|----------|---------|------------|---------------|-------------|-------------|-------------|----------------|--------------|--------------|------------|--------------|---------------|---------------|-------------|----------------|-------------|------------|---------------|---------------|--------------|--------------|-------------|-------------|--------------|------------|------------|----------------|----------------|--------------|----------------|--------------|-------------|---------------|-----------|-------------|--------------|--------------|------------|------------|----------------|-------------|----------------|----------------|------------|------------|----------------|----------------|---------------|----------------|--------------|--------------|---------------|--------------|---------------|----------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-----------------|-------------|---------------|------------|----------|---------|---------------|-----------|----------|----------|------------|-----------------|-------------|
|  0 |    1 |           60 | RL         |            65 |      8450 | Pave     |     nan | Reg        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        2003 |           2003 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          196 | Gd          | TA          | PConc        | Gd         | TA         | No             | GLQ            |          706 | Unf            |            0 |         150 |           856 | GasA      | Ex          | Y            | SBrkr        |        856 |        854 |              0 |        1710 |              1 |              0 |          2 |          1 |              3 |              1 | Gd            |              8 | Typ          |            0 | nan           | Attchd       |          2003 | RFn            |            2 |          548 | TA           | TA           | Y            |            0 |            61 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        2 |     2008 | WD         | Normal          |      208500 |
|  1 |    2 |           20 | RL         |            80 |      9600 | Pave     |     nan | Reg        | Lvl           | AllPub      | FR2         | Gtl         | Veenker        | Feedr        | Norm         | 1Fam       | 1Story       |             6 |             8 |        1976 |           1976 | Gable       | CompShg    | MetalSd       | MetalSd       | None         |            0 | TA          | TA          | CBlock       | Gd         | TA         | Gd             | ALQ            |          978 | Unf            |            0 |         284 |          1262 | GasA      | Ex          | Y            | SBrkr        |       1262 |          0 |              0 |        1262 |              0 |              1 |          2 |          0 |              3 |              1 | TA            |              6 | Typ          |            1 | TA            | Attchd       |          1976 | RFn            |            2 |          460 | TA           | TA           | Y            |          298 |             0 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        5 |     2007 | WD         | Normal          |      181500 |
|  2 |    3 |           60 | RL         |            68 |     11250 | Pave     |     nan | IR1        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        2001 |           2002 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          162 | Gd          | TA          | PConc        | Gd         | TA         | Mn             | GLQ            |          486 | Unf            |            0 |         434 |           920 | GasA      | Ex          | Y            | SBrkr        |        920 |        866 |              0 |        1786 |              1 |              0 |          2 |          1 |              3 |              1 | Gd            |              6 | Typ          |            1 | TA            | Attchd       |          2001 | RFn            |            2 |          608 | TA           | TA           | Y            |            0 |            42 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        9 |     2008 | WD         | Normal          |      223500 |
|  3 |    4 |           70 | RL         |            60 |      9550 | Pave     |     nan | IR1        | Lvl           | AllPub      | Corner      | Gtl         | Crawfor        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        1915 |           1970 | Gable       | CompShg    | Wd Sdng       | Wd Shng       | None         |            0 | TA          | TA          | BrkTil       | TA         | Gd         | No             | ALQ            |          216 | Unf            |            0 |         540 |           756 | GasA      | Gd          | Y            | SBrkr        |        961 |        756 |              0 |        1717 |              1 |              0 |          1 |          0 |              3 |              1 | Gd            |              7 | Typ          |            1 | Gd            | Detchd       |          1998 | Unf            |            3 |          642 | TA           | TA           | Y            |            0 |            35 |             272 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        2 |     2006 | WD         | Abnorml         |      140000 |
|  4 |    5 |           60 | RL         |            84 |     14260 | Pave     |     nan | IR1        | Lvl           | AllPub      | FR2         | Gtl         | NoRidge        | Norm         | Norm         | 1Fam       | 2Story       |             8 |             5 |        2000 |           2000 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          350 | Gd          | TA          | PConc        | Gd         | TA         | Av             | GLQ            |          655 | Unf            |            0 |         490 |          1145 | GasA      | Ex          | Y            | SBrkr        |       1145 |       1053 |              0 |        2198 |              1 |              0 |          2 |          1 |              4 |              1 | Gd            |              9 | Typ          |            1 | TA            | Attchd       |          2000 | RFn            |            3 |          836 | TA           | TA           | Y            |          192 |            84 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |       12 |     2008 | WD         | Normal          |      250000 |

</div>

```python
# Test Data
testData = pd.read_csv('test.csv')
testData.head()
```

<div class="table-wrapper" markdown="block">


|    |   Id |   MSSubClass | MSZoning   |   LotFrontage |   LotArea | Street   |   Alley | LotShape   | LandContour   | Utilities   | LotConfig   | LandSlope   | Neighborhood   | Condition1   | Condition2   | BldgType   | HouseStyle   |   OverallQual |   OverallCond |   YearBuilt |   YearRemodAdd | RoofStyle   | RoofMatl   | Exterior1st   | Exterior2nd   | MasVnrType   |   MasVnrArea | ExterQual   | ExterCond   | Foundation   | BsmtQual   | BsmtCond   | BsmtExposure   | BsmtFinType1   |   BsmtFinSF1 | BsmtFinType2   |   BsmtFinSF2 |   BsmtUnfSF |   TotalBsmtSF | Heating   | HeatingQC   | CentralAir   | Electrical   |   1stFlrSF |   2ndFlrSF |   LowQualFinSF |   GrLivArea |   BsmtFullBath |   BsmtHalfBath |   FullBath |   HalfBath |   BedroomAbvGr |   KitchenAbvGr | KitchenQual   |   TotRmsAbvGrd | Functional   |   Fireplaces | FireplaceQu   | GarageType   |   GarageYrBlt | GarageFinish   |   GarageCars |   GarageArea | GarageQual   | GarageCond   | PavedDrive   |   WoodDeckSF |   OpenPorchSF |   EnclosedPorch |   3SsnPorch |   ScreenPorch |   PoolArea |   PoolQC | Fence   | MiscFeature   |   MiscVal |   MoSold |   YrSold | SaleType   | SaleCondition   |
|----|------|--------------|------------|---------------|-----------|----------|---------|------------|---------------|-------------|-------------|-------------|----------------|--------------|--------------|------------|--------------|---------------|---------------|-------------|----------------|-------------|------------|---------------|---------------|--------------|--------------|-------------|-------------|--------------|------------|------------|----------------|----------------|--------------|----------------|--------------|-------------|---------------|-----------|-------------|--------------|--------------|------------|------------|----------------|-------------|----------------|----------------|------------|------------|----------------|----------------|---------------|----------------|--------------|--------------|---------------|--------------|---------------|----------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|-----------------|-------------|---------------|------------|----------|---------|---------------|-----------|----------|----------|------------|-----------------|
|  0 | 1461 |           20 | RH         |            80 |     11622 | Pave     |     nan | Reg        | Lvl           | AllPub      | Inside      | Gtl         | NAmes          | Feedr        | Norm         | 1Fam       | 1Story       |             5 |             6 |        1961 |           1961 | Gable       | CompShg    | VinylSd       | VinylSd       | None         |            0 | TA          | TA          | CBlock       | TA         | TA         | No             | Rec            |          468 | LwQ            |          144 |         270 |           882 | GasA      | TA          | Y            | SBrkr        |        896 |          0 |              0 |         896 |              0 |              0 |          1 |          0 |              2 |              1 | TA            |              5 | Typ          |            0 | nan           | Attchd       |          1961 | Unf            |            1 |          730 | TA           | TA           | Y            |          140 |             0 |               0 |           0 |           120 |          0 |      nan | MnPrv   | nan           |         0 |        6 |     2010 | WD         | Normal          |
|  1 | 1462 |           20 | RL         |            81 |     14267 | Pave     |     nan | IR1        | Lvl           | AllPub      | Corner      | Gtl         | NAmes          | Norm         | Norm         | 1Fam       | 1Story       |             6 |             6 |        1958 |           1958 | Hip         | CompShg    | Wd Sdng       | Wd Sdng       | BrkFace      |          108 | TA          | TA          | CBlock       | TA         | TA         | No             | ALQ            |          923 | Unf            |            0 |         406 |          1329 | GasA      | TA          | Y            | SBrkr        |       1329 |          0 |              0 |        1329 |              0 |              0 |          1 |          1 |              3 |              1 | Gd            |              6 | Typ          |            0 | nan           | Attchd       |          1958 | Unf            |            1 |          312 | TA           | TA           | Y            |          393 |            36 |               0 |           0 |             0 |          0 |      nan | nan     | Gar2          |     12500 |        6 |     2010 | WD         | Normal          |
|  2 | 1463 |           60 | RL         |            74 |     13830 | Pave     |     nan | IR1        | Lvl           | AllPub      | Inside      | Gtl         | Gilbert        | Norm         | Norm         | 1Fam       | 2Story       |             5 |             5 |        1997 |           1998 | Gable       | CompShg    | VinylSd       | VinylSd       | None         |            0 | TA          | TA          | PConc        | Gd         | TA         | No             | GLQ            |          791 | Unf            |            0 |         137 |           928 | GasA      | Gd          | Y            | SBrkr        |        928 |        701 |              0 |        1629 |              0 |              0 |          2 |          1 |              3 |              1 | TA            |              6 | Typ          |            1 | TA            | Attchd       |          1997 | Fin            |            2 |          482 | TA           | TA           | Y            |          212 |            34 |               0 |           0 |             0 |          0 |      nan | MnPrv   | nan           |         0 |        3 |     2010 | WD         | Normal          |
|  3 | 1464 |           60 | RL         |            78 |      9978 | Pave     |     nan | IR1        | Lvl           | AllPub      | Inside      | Gtl         | Gilbert        | Norm         | Norm         | 1Fam       | 2Story       |             6 |             6 |        1998 |           1998 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |           20 | TA          | TA          | PConc        | TA         | TA         | No             | GLQ            |          602 | Unf            |            0 |         324 |           926 | GasA      | Ex          | Y            | SBrkr        |        926 |        678 |              0 |        1604 |              0 |              0 |          2 |          1 |              3 |              1 | Gd            |              7 | Typ          |            1 | Gd            | Attchd       |          1998 | Fin            |            2 |          470 | TA           | TA           | Y            |          360 |            36 |               0 |           0 |             0 |          0 |      nan | nan     | nan           |         0 |        6 |     2010 | WD         | Normal          |
|  4 | 1465 |          120 | RL         |            43 |      5005 | Pave     |     nan | IR1        | HLS           | AllPub      | Inside      | Gtl         | StoneBr        | Norm         | Norm         | TwnhsE     | 1Story       |             8 |             5 |        1992 |           1992 | Gable       | CompShg    | HdBoard       | HdBoard       | None         |            0 | Gd          | TA          | PConc        | Gd         | TA         | No             | ALQ            |          263 | Unf            |            0 |        1017 |          1280 | GasA      | Ex          | Y            | SBrkr        |       1280 |          0 |              0 |        1280 |              0 |              0 |          2 |          0 |              2 |              1 | Gd            |              5 | Typ          |            0 | nan           | Attchd       |          1992 | RFn            |            2 |          506 | TA           | TA           | Y            |            0 |            82 |               0 |           0 |           144 |          0 |      nan | nan     | nan           |         0 |        1 |     2010 | WD         | Normal          |

</div>



The description of each of these columns are given a [text file](http://localhost:8888/edit/Data-Science-Projects/House%20Prices/data_description.txt) that came with the data. Some are self-explanatory such as "YearBuilt", but others that have codes for specific detalis (Ex. "Unf" = Unfinished (when referring to basements among other things)).

## Recreating Kaggle's Benchmark
For a "proof of concept", I will be recreating Kaggle's benchmark submission that predicts sale price based on a multiple linear regression of the year and month sold, the square footage and the number of bedrooms. This corresponds to columns **YrSold, MoSold, LotArea,** and **BedroomAbvGr**. The last variable is the number of bedrooms not including the basement. However, there is no variable representing the number of bedrooms in the basement. Thus, this is the closest variable we have.

### Side Note: Root Mean Square Logarithmic Error
As a side note, Kaggle usually uses the root mean square logarithmic error (RMSLE) as the accuracy metric. As opposed to root mean square error (RMSE), RMSLE does not penalize large differences when both the predicted and actual values are the same. Thus, it takes into account more so the percent difference than the actual difference. This works perfectly with house prices as they tend to be large. The formula is below.

$$
RMSLE = \sqrt{\frac{1}{n}\sum_{i=1}^n \left(\ln\left(\hat{y}_i+1\right) - \ln\left(y_i+1\right)\right)^2} =
\sqrt{\frac{1}{n}\sum_{i=1}^n \left(\ln\frac{\hat{y}_i+1}{y_i+1}\right)^2}
$$

where $$\hat{y}_i$$ is the predicted value and $$y_i$$ is the actual value.

## Multiple Linear Regression


```python
from sklearn.linear_model import LinearRegression
```


```python
subsettedData = fullData[["SalePrice", "YrSold", "MoSold", "LotArea", "BedroomAbvGr"]]
subsettedData.head()
```

<div class="table-wrapper" markdown="block">

|    |   SalePrice |   YrSold |   MoSold |   LotArea |   BedroomAbvGr |
|----|-------------|----------|----------|-----------|----------------|
|  0 |      208500 |     2008 |        2 |      8450 |              3 |
|  1 |      181500 |     2007 |        5 |      9600 |              3 |
|  2 |      223500 |     2008 |        9 |     11250 |              3 |
|  3 |      140000 |     2006 |        2 |      9550 |              3 |
|  4 |      250000 |     2008 |       12 |     14260 |              4 |

</div>



```python
X = subsettedData.drop('SalePrice', axis=1)
y = subsettedData['SalePrice']
model = LinearRegression()
model.fit(subsettedData.drop("SalePrice", axis=1), subsettedData['SalePrice'])
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

    Intercept: 1917554.08451
    Coefficients: [ -8.97676960e+02   1.10489255e+03   1.96803835e+00   1.32758801e+04]


The values above show the intercept for the model and the coefficients for each variable, in the same order as the table. So, the year sold variable has a coefficient of approximately -897.677. Now let's predict the actual values using the same features.


```python
# Method for RMSLE
def rmsle(actual, pred):
    return np.sqrt(np.mean(np.log((pred + 1) / (actual + 1)) ** 2))
preds = model.predict(subsettedData.drop('SalePrice', axis=1))
actual = subsettedData['SalePrice'].values
```


```python
rmsle(actual, preds)
```


    0.38192247230870091



Okay so our RMSLE is 0.38. However, this was checked through the model itself. What is mostly important is doing a validation data set. I'll use a 80-20 split on training data and validation data.


```python
from sklearn.model_selection import train_test_split
```


```python
Xtrain, Xval, Ytrain, Yval = train_test_split(X, y, test_size=0.2, random_state=1)
```


```python
# Build the model on training data and output error on validation data.
model = LinearRegression()
model.fit(Xtrain, Ytrain)
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
preds = model.predict(Xval)
print('RMSLE:', rmsle(Yval, preds))
```

    Intercept: -618251.757662
    Coefficients: [  3.66667325e+02   1.28320359e+03   2.01770663e+00   1.21543846e+04]
    RMSLE: 0.4103278685


As expected the error is a bit higher because we didn't test against the training data. Now let's test on the actual test data and submit.


```python
testResults = model.predict(testData[['YrSold', 'MoSold', 'LotArea', 'BedroomAbvGr']])
testVals = np.array([testData['Id'].values, testResults]).T
```




    array([[   1461.        ,  174207.34307145],
           [   1462.        ,  191698.56166012],
           [   1463.        ,  186967.2131038 ],
           ..., 
           [   2917.        ,  217803.39977709],
           [   2918.        ,  183795.35038424],
           [   2919.        ,  187285.75153451]])




```python
# Method to make the submission file. I'll most likely use this
# later on as well
def makeSubmissionFile(filename, testVals, testCols):
    # Make df of values and ids
    resdf = DataFrame(data=testVals, columns=testCols)
    resdf[testCols[0]] = resdf[testCols[0]].astype(int)
    resdf.to_csv(filename, index=False)
```


```python
makeSubmissionFile('benchmarkSub.csv', testVals, ['Id', 'SalePrice'])
```

After submitting, the error was 0.41057. But of course, this is just a benchmark and we can most likely do better.

# More Variables?
Of course, the easiest thing to do is add more variables. But we have to be careful here: we can't simply add all of them. There will be some that are correlated with each other, which will skew the final predictions. For example, in this dataset, there are variables that list the square footage of this 1st floor and the square footage of the 2nd floor. This is highly correlated with the total square footage of the house.


```python
fullData.dtypes
```




    Id                 int64
    MSSubClass         int64
    MSZoning          object
    LotFrontage      float64
    LotArea            int64
    Street            object
    Alley             object
    LotShape          object
    LandContour       object
    Utilities         object
    LotConfig         object
    LandSlope         object
    Neighborhood      object
    Condition1        object
    Condition2        object
    BldgType          object
    HouseStyle        object
    OverallQual        int64
    OverallCond        int64
    YearBuilt          int64
    YearRemodAdd       int64
    RoofStyle         object
    RoofMatl          object
    Exterior1st       object
    Exterior2nd       object
    MasVnrType        object
    MasVnrArea       float64
    ExterQual         object
    ExterCond         object
    Foundation        object
                      ...   
    BedroomAbvGr       int64
    KitchenAbvGr       int64
    KitchenQual       object
    TotRmsAbvGrd       int64
    Functional        object
    Fireplaces         int64
    FireplaceQu       object
    GarageType        object
    GarageYrBlt      float64
    GarageFinish      object
    GarageCars         int64
    GarageArea         int64
    GarageQual        object
    GarageCond        object
    PavedDrive        object
    WoodDeckSF         int64
    OpenPorchSF        int64
    EnclosedPorch      int64
    3SsnPorch          int64
    ScreenPorch        int64
    PoolArea           int64
    PoolQC            object
    Fence             object
    MiscFeature       object
    MiscVal            int64
    MoSold             int64
    YrSold             int64
    SaleType          object
    SaleCondition     object
    SalePrice          int64
    dtype: object



First, to see what variables are most important, I'll convert all of the 'objects' to categorical variables.


```python
for col in fullData.columns:
    if fullData[col].dtype == 'object':
        fullData[col] = fullData[col].astype('category')
fullData.Street.head()
```




    0    Pave
    1    Pave
    2    Pave
    3    Pave
    4    Pave
    Name: Street, dtype: category
    Categories (2, object): [Grvl, Pave]



It appears to have worked to perfection. In order to see which of the categorical variables (I will get to numerical variables later in the document) matter, I can use Levene's test for homeoscedasticity. This test is to see if the samples of the different groups came from the same population. Hence, the groups would staistically have the same variance. The null hypothesis is that the population variances are equal. Hence, the variables that show the lowest p-values are the ones where the different group distinctions matter.


```python
from scipy.stats import levene
```


```python
leveneVals = []
varNames = []
# Loop over every categorical variable
for col in fullData.columns:
    if fullData[col].dtype.name == 'category':
        varNames.append(col)
        # Gather all categories and group them
        cats = fullData[col].cat.categories
        groups = []
        for cat in cats:
            groups.append(fullData['SalePrice'][fullData[col] == cat])
        # Run Levene's test on the groups
        leveneW, leveneP = levene(*groups)
        leveneVals.append(leveneP)
        print(col + ": " + str(leveneP))
leveneVals = Series(data=leveneVals, index=varNames)
```

    MSZoning: 7.63531180631e-10
    Street: 0.753910659281
    Alley: 0.773760104376
    LotShape: 0.00407833351284
    LandContour: 0.000742488253912
    Utilities: 0.351327102246
    LotConfig: 0.397947589138
    LandSlope: 0.0748775725309
    Neighborhood: 3.36224919733e-43
    Condition1: 0.000308509922861
    Condition2: 0.435873369641
    BldgType: 1.30056271794e-07
    HouseStyle: 3.79913681871e-09
    RoofStyle: 2.51682063508e-16
    RoofMatl: 0.00538652757886
    Exterior1st: 4.01962459515e-16
    Exterior2nd: 1.71693114905e-17
    MasVnrType: 7.23963355999e-12
    ExterQual: 1.01259552913e-31
    ExterCond: 0.0718496830362
    Foundation: 2.47181514878e-20
    BsmtQual: 4.78197314686e-35
    BsmtCond: 0.0483961835357
    BsmtExposure: 1.87433542175e-20
    BsmtFinType1: 3.35109748103e-21
    BsmtFinType2: 0.00166782885971
    Heating: 0.173309968198
    HeatingQC: 1.29476574201e-20
    CentralAir: 4.22322798127e-05
    Electrical: 1.31841216052e-05
    KitchenQual: 1.27928785916e-45
    Functional: 0.0039415564787
    FireplaceQu: 1.28217773623e-09
    GarageType: 2.666183136e-19
    GarageFinish: 2.49013621701e-28
    GarageQual: 0.00867164423537
    GarageCond: 0.0220641592879
    PavedDrive: 9.90414942125e-05
    PoolQC: 0.00765753259065
    Fence: 0.661295791136
    MiscFeature: 0.578768550116
    SaleType: 3.29013488712e-07
    SaleCondition: 5.5376860639e-08


From the above, it appears that the lowest p-value, and hence the most significant, is the Kitchen Quality. This makes sense, as poor kitchen quality will most likely result in a lower sale price. The second lowest is the neighborhood. Low prices around the target house will pull that house's price down as well. Poor schools and high crime in the area can also influence the price.


```python
catVars = leveneVals.sort_values(ascending=True).head(10).index
```

### Numerical Variables
Now I'll deal with the numerical variables. Variable importance works a little differently, as it isn't straightforward to determine "the most important variable". Instead, I can use Principal Component Analysis to extract n components and transform my original data and finally run regression on the transformations.


```python
# Filter out all non-numerical variables
numVars = fullData.dtypes[fullData.dtypes != 'category'].index
fullData[numVars].isnull().sum()
```




    Id                 0
    MSSubClass         0
    LotFrontage      259
    LotArea            0
    OverallQual        0
    OverallCond        0
    YearBuilt          0
    YearRemodAdd       0
    MasVnrArea         8
    BsmtFinSF1         0
    BsmtFinSF2         0
    BsmtUnfSF          0
    TotalBsmtSF        0
    1stFlrSF           0
    2ndFlrSF           0
    LowQualFinSF       0
    GrLivArea          0
    BsmtFullBath       0
    BsmtHalfBath       0
    FullBath           0
    HalfBath           0
    BedroomAbvGr       0
    KitchenAbvGr       0
    TotRmsAbvGrd       0
    Fireplaces         0
    GarageYrBlt       81
    GarageCars         0
    GarageArea         0
    WoodDeckSF         0
    OpenPorchSF        0
    EnclosedPorch      0
    3SsnPorch          0
    ScreenPorch        0
    PoolArea           0
    MiscVal            0
    MoSold             0
    YrSold             0
    SalePrice          0
    dtype: int64



It looks like LotFrontage, MasVnrArea, and GarageYrBlt have some NaNs present. Thus, I'll remove these variables. Id should also be removed because it holds no information.


```python
numVars = numVars.drop(['Id','LotFrontage', 'MasVnrArea', 'GarageYrBlt'])
```


```python
from sklearn.decomposition import PCA
```


```python
# Normalize each column
numDF = fullData[numVars]
# Save the mean and standard deviation so we can
# properly normalize the test data when we get to it
trainMean = numDF.mean()
trainSTD = numDF.std()
normedDF = (numDF - trainMean) / trainSTD
# REMOVE THE SALE PRICE
normedDF = normedDF.drop('SalePrice', axis=1)
```


```python
normedDF.head()
```

<div class="table-wrapper" markdown="block">

|    |   MSSubClass |    LotArea |   OverallQual |   OverallCond |   YearBuilt |   YearRemodAdd |   BsmtFinSF1 |   BsmtFinSF2 |   BsmtUnfSF |   TotalBsmtSF |   1stFlrSF |   2ndFlrSF |   LowQualFinSF |   GrLivArea |   BsmtFullBath |   BsmtHalfBath |   FullBath |   HalfBath |   BedroomAbvGr |   KitchenAbvGr |   TotRmsAbvGrd |   Fireplaces |   GarageCars |   GarageArea |   WoodDeckSF |   OpenPorchSF |   EnclosedPorch |   3SsnPorch |   ScreenPorch |   PoolArea |    MiscVal |    MoSold |    YrSold |
|----|--------------|------------|---------------|---------------|-------------|----------------|--------------|--------------|-------------|---------------|------------|------------|----------------|-------------|----------------|----------------|------------|------------|----------------|----------------|----------------|--------------|--------------|--------------|--------------|---------------|-----------------|-------------|---------------|------------|------------|-----------|-----------|
|  0 |    0.0733498 | -0.207071  |     0.651256  |     -0.517023 |    1.05063  |       0.878367 |    0.575228  |    -0.288554 |  -0.944267  |     -0.459145 | -0.793162  |   1.16145  |      -0.120201 |    0.370207 |       1.10743  |      -0.240978 |    0.78947 |    1.22716 |       0.163723 |      -0.211381 |       0.911897 |    -0.950901 |     0.311618 |    0.35088   |    -0.751918 |     0.216429  |       -0.359202 |   -0.116299 |     -0.270116 | -0.0686682 | -0.0876578 | -1.59856  |  0.13873  |
|  1 |   -0.872264  | -0.0918549 |    -0.0718115 |      2.17888  |    0.15668  |      -0.42943  |    1.17159   |    -0.288554 |  -0.641008  |      0.466305 |  0.257052  |  -0.794891 |      -0.120201 |   -0.482347 |      -0.819684 |       3.94746  |    0.78947 |   -0.76136 |       0.163723 |      -0.211381 |      -0.318574 |     0.600289 |     0.311618 |   -0.0607102 |     1.62564  |    -0.704242  |       -0.359202 |   -0.116299 |     -0.270116 | -0.0686682 | -0.0876578 | -0.488943 | -0.614228 |
|  2 |    0.0733498 |  0.0734548 |     0.651256  |     -0.517023 |    0.984415 |       0.82993  |    0.0928754 |    -0.288554 |  -0.30154   |     -0.313261 | -0.627611  |   1.18894  |      -0.120201 |    0.514836 |       1.10743  |      -0.240978 |    0.78947 |    1.22716 |       0.163723 |      -0.211381 |      -0.318574 |     0.600289 |     0.311618 |    0.63151   |    -0.751918 |    -0.0703374 |       -0.359202 |   -0.116299 |     -0.270116 | -0.0686682 | -0.0876578 |  0.990552 |  0.13873  |
|  3 |    0.309753  | -0.0968643 |     0.651256  |     -0.517023 |   -1.86299  |      -0.720051 |   -0.499103  |    -0.288554 |  -0.0616484 |     -0.687089 | -0.521555  |   0.936955 |      -0.120201 |    0.383528 |       1.10743  |      -0.240978 |   -1.02569 |   -0.76136 |       0.163723 |      -0.211381 |       0.296662 |     0.600289 |     1.64974  |    0.790533  |    -0.751918 |    -0.175988  |        4.09112  |   -0.116299 |     -0.270116 | -0.0686682 | -0.0876578 | -1.59856  | -1.36719  |
|  4 |    0.0733498 |  0.37502   |     1.37432   |     -0.517023 |    0.951306 |       0.733056 |    0.46341   |    -0.288554 |  -0.174805  |      0.199611 | -0.0455956 |   1.61732  |      -0.120201 |    1.29888  |       1.10743  |      -0.240978 |    0.78947 |    1.22716 |       1.38955  |      -0.211381 |       1.52713  |     0.600289 |     1.64974  |    1.6979    |     0.77993  |     0.563567  |       -0.359202 |   -0.116299 |     -0.270116 | -0.0686682 | -0.0876578 |  2.10017  |  0.13873  |

</div>




```python
pca = PCA(n_components=10)
pca.fit_transform(normedDF)
pca.explained_variance_ratio_
```




    array([ 0.19386639,  0.09564243,  0.06510207,  0.05986886,  0.04439883,
            0.03579139,  0.03477911,  0.03354216,  0.03237783,  0.03141367])



The above array shows that the most important component counts for about 19% of the variance in the sale price.
## Putting it Together
Now we have our PCA components and our top 10 categorical variables. We can now prepare the training data for linear regression.


```python
# The categorical variables
catVars
```




    Index(['KitchenQual', 'Neighborhood', 'BsmtQual', 'ExterQual', 'GarageFinish',
           'BsmtFinType1', 'HeatingQC', 'BsmtExposure', 'Foundation',
           'GarageType'],
          dtype='object')




```python
# The PCA component vectors, ordered according to explained variance ratio
pca.components_
```




    array([[-0.01144775,  0.11716568,  0.3175264 , -0.07927477,  0.23371292,
             0.21328179,  0.14354421, -0.00966022,  0.12478685,  0.27136905,
             0.27285029,  0.1544571 , -0.00810637,  0.32829263,  0.07587995,
            -0.01276899,  0.28029767,  0.13682643,  0.13225492, -0.01193569,
             0.26747248,  0.2030522 ,  0.29644861,  0.28981523,  0.1422787 ,
             0.16185741, -0.07135596,  0.01587492,  0.0376356 ,  0.05570352,
            -0.01013584,  0.02646643, -0.01640278],
           [ 0.18581718, -0.06688223, -0.02897891,  0.04745998, -0.17377722,
            -0.07120355, -0.32492349, -0.08799764,  0.11939259, -0.24991047,
            -0.20069333,  0.42775414,  0.11487164,  0.21832724, -0.31306429,
            -0.0031042 ,  0.14639495,  0.22771056,  0.34170233,  0.17068491,
             0.29896409, -0.02636508, -0.07862111, -0.11604252, -0.08111536,
             0.02946647,  0.10346949, -0.03116904, -0.00115575, -0.00414277,
             0.02396261,  0.02563381, -0.03467519],
           [-0.07445028,  0.28589084, -0.14320263,  0.16086227, -0.3542583 ,
            -0.28041604,  0.29089313,  0.21732575, -0.28986124,  0.0903899 ,
             0.16972117,  0.05398886,  0.14416365,  0.18305056,  0.24943369,
             0.08962967, -0.12160159, -0.03927105,  0.19615128,  0.10269423,
             0.17388388,  0.20826183, -0.15624625, -0.09392718,  0.06417138,
            -0.01580694,  0.20066159, -0.03163643,  0.14773639,  0.19252758,
             0.06770929, -0.02225253,  0.01031902],
           [-0.32986227,  0.07243755, -0.04028223, -0.04086595, -0.17056284,
            -0.14196268, -0.24168747, -0.01278588,  0.50407234,  0.25173414,
             0.30584549, -0.26884652,  0.0873004 ,  0.00974685, -0.28294509,
            -0.00745574,  0.03291061, -0.36481884,  0.10936531,  0.09814584,
             0.07461655, -0.01937812, -0.02834016, -0.00158589, -0.11702547,
            -0.04121343,  0.14243471,  0.03507289, -0.00483109, -0.01406849,
             0.00114179,  0.04072845, -0.04909265],
           [ 0.31800058, -0.06370195, -0.11731013, -0.40944798,  0.06738235,
            -0.12318253,  0.14109536, -0.11298065, -0.06305904,  0.04163082,
             0.10052938, -0.07694847, -0.00442442,  0.00962567,  0.21953624,
            -0.25778799,  0.15857598, -0.16925298,  0.04834307,  0.56995376,
             0.09244362, -0.19901865,  0.00716383,  0.00407266, -0.04964465,
            -0.14847193,  0.01143631, -0.06080923, -0.23767228, -0.01288102,
            -0.00867304, -0.07417711,  0.06894734],
           [-0.08484596,  0.03755305,  0.03999344,  0.25507455, -0.03583602,
             0.19971723, -0.0689783 ,  0.04120836,  0.06650488,  0.01042418,
             0.00362992,  0.027912  ,  0.03224704,  0.02884144,  0.03313599,
            -0.16632149,  0.02904962, -0.00489729,  0.00788877, -0.04534276,
             0.04780867, -0.07416056, -0.01884119, -0.00611001,  0.13194496,
            -0.15500125,  0.12784971,  0.0380569 , -0.14485224, -0.08227734,
             0.11459572, -0.56333581,  0.642278  ],
           [ 0.044988  ,  0.10850217, -0.05270799,  0.30464922,  0.01122632,
             0.15889156,  0.05357019, -0.00702034, -0.08738381, -0.03490103,
             0.04472069, -0.04374779, -0.20863212, -0.02274685, -0.08862082,
             0.52606802,  0.11429168, -0.14888153,  0.11595198,  0.23653077,
             0.05822951, -0.10692132, -0.00321333, -0.02152168,  0.24810939,
            -0.16799962, -0.15783879,  0.29150655, -0.32247704, -0.16047945,
             0.12822873,  0.23469502,  0.00718269],
           [ 0.08566004, -0.14961357,  0.13185269,  0.21798014, -0.09192399,
             0.19137956,  0.13944815, -0.33589372,  0.01298562,  0.03454214,
            -0.00317855,  0.03229137,  0.31224037,  0.05337859,  0.08006244,
            -0.07213393,  0.01061944, -0.12041892, -0.12146108, -0.14813016,
            -0.04134165, -0.09607634, -0.06799711, -0.02535246,  0.03039166,
             0.09270777,  0.40324978,  0.12682766, -0.43076219,  0.32178295,
            -0.07165678,  0.03742506, -0.24959011],
           [ 0.03194989, -0.18988745,  0.03795263,  0.26346125, -0.08318009,
             0.07900221,  0.14586241, -0.32896994, -0.02006101,  0.0104719 ,
             0.0414586 , -0.02961162,  0.02779207,  0.00847303,  0.06659408,
            -0.17797593,  0.01857304, -0.08017563, -0.04461517,  0.16393606,
             0.02448332,  0.02520971, -0.02034911,  0.0013772 , -0.3745047 ,
             0.17067141, -0.16562884,  0.3476325 ,  0.38999819, -0.05549094,
             0.40133727,  0.13731186,  0.13889883],
           [ 0.22955189, -0.18000455,  0.01990597,  0.0953794 ,  0.09520976,
             0.19090499, -0.08587357,  0.29651415,  0.04623734,  0.06632568,
             0.04849588, -0.11006802,  0.42179011, -0.01672978, -0.05015867,
             0.17760947,  0.118691  , -0.17095455, -0.02025871,  0.09170184,
            -0.02069945, -0.13913518, -0.15407312, -0.14948792,  0.14272648,
             0.15869672, -0.35642912, -0.27290127,  0.17361364,  0.33829688,
             0.11825237, -0.08754927,  0.00652005]])



We need to create dummy variables for our categories, and drop one to prevent multicoolinearity.


```python
categoricalDums = pd.get_dummies(fullData[catVars], drop_first=True)
```


```python
# Make the dataframe of the PCAs
# We have to dot our larger space by the component space
pcaDF = DataFrame(data=np.dot(normedDF.values, pca.components_.T), columns=['PCA_' + str(i+1) for i in range(pca.n_components_)])
pcaDF.head()
```

<div class="table-wrapper" markdown="block">

|    |     PCA_1 |     PCA_2 |      PCA_3 |     PCA_4 |      PCA_5 |     PCA_6 |     PCA_7 |      PCA_8 |      PCA_9 |    PCA_10 |
|----|-----------|-----------|------------|-----------|------------|-----------|-----------|------------|------------|-----------|
|  0 |  1.22352  |  0.589885 | -0.688187  | -2.33775  |  0.703052  | -0.990091 | -0.696729 |  0.174321  | -0.0359122 | -0.324323 |
|  1 |  0.021512 | -1.20798  |  0.974832  |  0.279447 | -1.90917   |  0.319969 |  3.35422  |  0.0989099 | -0.697438  |  1.12994  |
|  2 |  1.4673   |  0.410598 | -0.795529  | -1.78799  |  0.0518015 |  0.474733 | -0.294814 | -0.0461904 |  0.121852  | -0.754562 |
|  3 | -0.31302  |  1.01254  |  1.40114   |  0.374859 |  0.321726  | -0.501043 | -1.39692  |  2.13436   | -0.736633  | -2.25244  |
|  4 |  4.10583  |  1.05073  | -0.0764433 | -1.53499  |  0.0410682 |  0.842158 |  0.479617 | -0.0927415 | -0.339045  | -1.27851  |

</div>




```python
finalX = pd.concat([pcaDF, categoricalDums], axis=1)
finalX.head()
```

<div class="table-wrapper" markdown="block">

|    |     PCA_1 |     PCA_2 |      PCA_3 |     PCA_4 |      PCA_5 |     PCA_6 |     PCA_7 |      PCA_8 |      PCA_9 |    PCA_10 |   KitchenQual_Fa |   KitchenQual_Gd |   KitchenQual_TA |   Neighborhood_Blueste |   Neighborhood_BrDale |   Neighborhood_BrkSide |   Neighborhood_ClearCr |   Neighborhood_CollgCr |   Neighborhood_Crawfor |   Neighborhood_Edwards |   Neighborhood_Gilbert |   Neighborhood_IDOTRR |   Neighborhood_MeadowV |   Neighborhood_Mitchel |   Neighborhood_NAmes |   Neighborhood_NPkVill |   Neighborhood_NWAmes |   Neighborhood_NoRidge |   Neighborhood_NridgHt |   Neighborhood_OldTown |   Neighborhood_SWISU |   Neighborhood_Sawyer |   Neighborhood_SawyerW |   Neighborhood_Somerst |   Neighborhood_StoneBr |   Neighborhood_Timber |   Neighborhood_Veenker |   BsmtQual_Fa |   BsmtQual_Gd |   BsmtQual_TA |   ExterQual_Fa |   ExterQual_Gd |   ExterQual_TA |   GarageFinish_RFn |   GarageFinish_Unf |   BsmtFinType1_BLQ |   BsmtFinType1_GLQ |   BsmtFinType1_LwQ |   BsmtFinType1_Rec |   BsmtFinType1_Unf |   HeatingQC_Fa |   HeatingQC_Gd |   HeatingQC_Po |   HeatingQC_TA |   BsmtExposure_Gd |   BsmtExposure_Mn |   BsmtExposure_No |   Foundation_CBlock |   Foundation_PConc |   Foundation_Slab |   Foundation_Stone |   Foundation_Wood |   GarageType_Attchd |   GarageType_Basment |   GarageType_BuiltIn |   GarageType_CarPort |   GarageType_Detchd |
|----|-----------|-----------|------------|-----------|------------|-----------|-----------|------------|------------|-----------|------------------|------------------|------------------|------------------------|-----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|-----------------------|------------------------|------------------------|----------------------|------------------------|-----------------------|------------------------|------------------------|------------------------|----------------------|-----------------------|------------------------|------------------------|------------------------|-----------------------|------------------------|---------------|---------------|---------------|----------------|----------------|----------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|----------------|----------------|----------------|----------------|-------------------|-------------------|-------------------|---------------------|--------------------|-------------------|--------------------|-------------------|---------------------|----------------------|----------------------|----------------------|---------------------|
|  0 |  1.22352  |  0.589885 | -0.688187  | -2.33775  |  0.703052  | -0.990091 | -0.696729 |  0.174321  | -0.0359122 | -0.324323 |                0 |                1 |                0 |                      0 |                     0 |                      0 |                      0 |                      1 |                      0 |                      0 |                      0 |                     0 |                      0 |                      0 |                    0 |                      0 |                     0 |                      0 |                      0 |                      0 |                    0 |                     0 |                      0 |                      0 |                      0 |                     0 |                      0 |             0 |             1 |             0 |              0 |              1 |              0 |                  1 |                  0 |                  0 |                  1 |                  0 |                  0 |                  0 |              0 |              0 |              0 |              0 |                 0 |                 0 |                 1 |                   0 |                  1 |                 0 |                  0 |                 0 |                   1 |                    0 |                    0 |                    0 |                   0 |
|  1 |  0.021512 | -1.20798  |  0.974832  |  0.279447 | -1.90917   |  0.319969 |  3.35422  |  0.0989099 | -0.697438  |  1.12994  |                0 |                0 |                1 |                      0 |                     0 |                      0 |                      0 |                      0 |                      0 |                      0 |                      0 |                     0 |                      0 |                      0 |                    0 |                      0 |                     0 |                      0 |                      0 |                      0 |                    0 |                     0 |                      0 |                      0 |                      0 |                     0 |                      1 |             0 |             1 |             0 |              0 |              0 |              1 |                  1 |                  0 |                  0 |                  0 |                  0 |                  0 |                  0 |              0 |              0 |              0 |              0 |                 1 |                 0 |                 0 |                   1 |                  0 |                 0 |                  0 |                 0 |                   1 |                    0 |                    0 |                    0 |                   0 |
|  2 |  1.4673   |  0.410598 | -0.795529  | -1.78799  |  0.0518015 |  0.474733 | -0.294814 | -0.0461904 |  0.121852  | -0.754562 |                0 |                1 |                0 |                      0 |                     0 |                      0 |                      0 |                      1 |                      0 |                      0 |                      0 |                     0 |                      0 |                      0 |                    0 |                      0 |                     0 |                      0 |                      0 |                      0 |                    0 |                     0 |                      0 |                      0 |                      0 |                     0 |                      0 |             0 |             1 |             0 |              0 |              1 |              0 |                  1 |                  0 |                  0 |                  1 |                  0 |                  0 |                  0 |              0 |              0 |              0 |              0 |                 0 |                 1 |                 0 |                   0 |                  1 |                 0 |                  0 |                 0 |                   1 |                    0 |                    0 |                    0 |                   0 |
|  3 | -0.31302  |  1.01254  |  1.40114   |  0.374859 |  0.321726  | -0.501043 | -1.39692  |  2.13436   | -0.736633  | -2.25244  |                0 |                1 |                0 |                      0 |                     0 |                      0 |                      0 |                      0 |                      1 |                      0 |                      0 |                     0 |                      0 |                      0 |                    0 |                      0 |                     0 |                      0 |                      0 |                      0 |                    0 |                     0 |                      0 |                      0 |                      0 |                     0 |                      0 |             0 |             0 |             1 |              0 |              0 |              1 |                  0 |                  1 |                  0 |                  0 |                  0 |                  0 |                  0 |              0 |              1 |              0 |              0 |                 0 |                 0 |                 1 |                   0 |                  0 |                 0 |                  0 |                 0 |                   0 |                    0 |                    0 |                    0 |                   1 |
|  4 |  4.10583  |  1.05073  | -0.0764433 | -1.53499  |  0.0410682 |  0.842158 |  0.479617 | -0.0927415 | -0.339045  | -1.27851  |                0 |                1 |                0 |                      0 |                     0 |                      0 |                      0 |                      0 |                      0 |                      0 |                      0 |                     0 |                      0 |                      0 |                    0 |                      0 |                     0 |                      1 |                      0 |                      0 |                    0 |                     0 |                      0 |                      0 |                      0 |                     0 |                      0 |             0 |             1 |             0 |              0 |              1 |              0 |                  1 |                  0 |                  0 |                  1 |                  0 |                  0 |                  0 |              0 |              0 |              0 |              0 |                 0 |                 0 |                 0 |                   0 |                  1 |                 0 |                  0 |                 0 |                   1 |                    0 |                    0 |                    0 |                   0 |

</div>



## The Model


```python
model = LinearRegression()
# Don't forget to normalize y as well
yMean = np.mean(y)
ySTD = np.std(y)
normedY = (y - yMean) / ySTD
model.fit(finalX, normedY)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
rmsle(y, model.predict(finalX) * ySTD + yMean)
```




    0.1412499871685208



Wow what a result. Miles better than the first blunt test against the training data. Let's see how we do with a train test split!


```python
Xtrain, Xval, Ytrain, Yval = train_test_split(finalX, normedY, test_size=0.2, random_state=1)
```


```python
model = LinearRegression()
model.fit(Xtrain, Ytrain)
print('RMSLE:', rmsle(Yval * ySTD + yMean, model.predict(Xval) * ySTD + yMean))
```

    RMSLE: 0.155953580834


Things are looking up now! Next step is to prepare the submission to Kaggle. Unlike before there's a little bit more work put into this, as I trained on normalized values. Thus, I'll need to use the trained mean and standard deviation to "standardize" the test data. To get the actual sale price values, I'll need to restandardize using the trained sale price mean and standard deviation.


```python
# First convert all object categories to categoricals
for col in testData.columns:
    if testData[col].dtype == 'object':
        testData[col] = testData[col].astype('category')
preparedTest = DataFrame.copy(testData)
# Pull out numerical variables
numTestVars = testData.dtypes[testData.dtypes != 'category'].index
numTestVars = numTestVars.drop(['Id', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt'])
numTestDF = testData[numTestVars]
# Normalize the data
numTestDF = (numTestDF - trainMean) / trainSTD
# For some reason, numTestDF's columns went out of order.
# In order to properly apply the PCA transformation, we need to put it back.
numTestDF = numTestDF[normedDF.columns]
```


```python
# Check for null values in numTestDF
numTestDF.isnull().sum()
```




    MSSubClass       0
    LotArea          0
    OverallQual      0
    OverallCond      0
    YearBuilt        0
    YearRemodAdd     0
    BsmtFinSF1       1
    BsmtFinSF2       1
    BsmtUnfSF        1
    TotalBsmtSF      1
    1stFlrSF         0
    2ndFlrSF         0
    LowQualFinSF     0
    GrLivArea        0
    BsmtFullBath     2
    BsmtHalfBath     2
    FullBath         0
    HalfBath         0
    BedroomAbvGr     0
    KitchenAbvGr     0
    TotRmsAbvGrd     0
    Fireplaces       0
    GarageCars       1
    GarageArea       1
    WoodDeckSF       0
    OpenPorchSF      0
    EnclosedPorch    0
    3SsnPorch        0
    ScreenPorch      0
    PoolArea         0
    MiscVal          0
    MoSold           0
    YrSold           0
    dtype: int64



Okay so it appears that there are some null values with this test data. Looking at the variables involved, it will most likely be safe to substitute these with 0 square foot/bathrooms. I'll explain: Suppose a family is looking to buy a house. The null values indicate that there is no information about the basement/garage. They will most likely not go through as they want to be extremely sure what they're getting. Thus, having no information will deter the price of a house, which is what having no basements/garage will do as well.


```python
numTestDF = numTestDF.fillna(0)
```


```python
# Apply PCA Transformation and concatenate with dummies
testCatVars = pd.get_dummies(testData[catVars], drop_first=True)
pcaTestDF = DataFrame(data=np.dot(numTestDF.values, pca.components_.T), columns=['PCA_' + str(i+1) for i in range(pca.n_components_)])
finalTestX = pd.concat([pcaTestDF, testCatVars], axis=1)
# Train on ALL of our training data
model = LinearRegression()
model.fit(finalX, normedY)
# Find the predicted y values, renoramlized
predictedVals = model.predict(finalTestX) * ySTD + yMean
predictedVals
```




    array([ 120103.57577568,  165180.76927381,  186513.2865078 , ...,
            173556.99974137,  114622.01047829,  209822.1047847 ])




```python
testVals = np.array([testData['Id'].values, predictedVals]).T
testVals
```




    array([[   1461.        ,  120103.57577568],
           [   1462.        ,  165180.76927381],
           [   1463.        ,  186513.2865078 ],
           ..., 
           [   2917.        ,  173556.99974137],
           [   2918.        ,  114622.01047829],
           [   2919.        ,  209822.1047847 ]])




```python
makeSubmissionFile('PCACat10.csv', testVals, ['Id', 'SalePrice'])
```

# Result
This linear regression model scored a 0.15706 on the Kaggle. But even so, I am still placed 2102 as of January 26, 2018. Because of the scoring method, it will get extremely difficult to lower this score. This is evidenced by the fact that the no. 50 spot has a score of 0.11349.
