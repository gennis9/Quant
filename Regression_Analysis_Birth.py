# Copyright (C) 2023 Gennis
'''
Initiated Date    : 2023/10/25
Last Updated Date : 2023/12/31
Aim: Regression Analysis on birth rate as response.
Input Data: National Statistics: County and city statitistics.
    https://winsta.dgbas.gov.tw/DgbasWeb/ZWeb/StateFile_ZWeb.aspx
'''


#%% Enviornment

from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import os
import pandas as pd
import pingouin as pg
import re
import seaborn as sns
import scipy
# import sklearn as sk
from sklearn import linear_model as skl
from sklearn import model_selection as skm
from sklearn import preprocessing as skp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
%matplotlib inline


## Set the path of data.
_Path = r'D:\03Programs_Clouds\Google Drive\NSYSU\01Reg Analysis\Mid Term Report\Input Data'
os.chdir(_Path)

## Correctly show the Mandarin in the graphs.
matplotlib.rc('font', family='Microsoft JhengHei')
## Correctly show the negative sign in the graphs.
plt.rcParams['axes.unicode_minus'] = False


## Set the model specification/ parameters.
# Set the response and its corresponding dataset location.
response = '粗出生率(‰)'
y_loc    = 0

# How many missing values are allowed in the predictors.
# If >=1, the thresholds is number of missing values.
# If <1, the thresholds is the proportion of the missing values.
missNum = 0.01

# Set the start year and end year for the investigation period.
startYear = 2003
endYear   = 2022

# whether the fixed effect dummies enters the round of variable selection.
dummySelect = False

# Define the selection criteria in the process of variable selection.
# S1 stands for the step of finding candidate models by adding 1 variable;
# S1: 'p-value' / 'R^2'.
# S2 stands for the step of checking whether a stopping criterion is reached;
# S2: 'AIC' / 'BIC' / 'R^2' / 'adj R^2' / 'partial f test' / False.
# S3: stands for the step of choosing the best among the candidate models;
# S3: 'global' / 'final'.
# stop parameter: control whether stopping criterion is applied.
# diff_either: control whether either level or growth value of a variable is put into model.
# std: an optional parameter that could be used for removing the variables with low marginal effects using standardization threshold.
model = 'linear'
S1 = 'R^2'
S2 = 'AIC'
S3 = 'global'
significance = 0.05
stop = False
diff_either = True
std = 0.2

# Set the file type of the graphs exported.
# If no graph is exported, set False.
graph = False



# %% Data Access

## Read the data.
_FileNames = ['Demographic.csv', 'Statistics.csv', 'HealthCare.csv',
              'Enviornment.csv', 'PublicFinance.csv', 'HouseholdFinance.csv']
RawTables = [ pd.read_csv(file) for file in _FileNames ]




# %% Extract-Transform-Load

def Func_FillCounty(table):
    ## Prevent data contagion.
    dta = deepcopy(table)
    
    ## Correctly display the name of indexing variables.
    _col = dta.columns.tolist()
    _col[0] = 'County'
    _col[1] = 'Year'
    
    dta.columns = _col
    
    ## Fill back the missing data of the county index.
    dta['County'] = dta['County'].fillna(method="ffill") 

    ## Remove the redundant rows due to the county index.
    dta.dropna(subset=['Year'], inplace=True)
    
    ## Correctly reflect the missing values in format.
    return dta

Preclean = [ Func_FillCounty(dta) for dta in RawTables ]


## Remove the observations out of the investigation period.
for _i in range(len(Preclean)):
    Preclean[_i] = Preclean[_i][ Preclean[_i]['Year'] >= startYear ]
    Preclean[_i] = Preclean[_i][ Preclean[_i]['Year'] <= endYear ]




# %% Data Cleaning

## Convert the data to correct data type.
def Func_Format(dta):
    ## Remove the variables with more than 1 missing value.
    dta = dta.replace( '-', np.nan )
    dta = dta.replace( '...', np.nan )
    
    ## Remove the thousands separators in the data.
    dta = dta.astype(str)

    for col in dta.columns[2:]:
        dta[col] = dta[col].str.replace(',','',regex=True).astype(float)

    dta['Year'] = dta['Year'].astype(float).astype(int)

    return dta


## Clean the data.
def Func_Cleaner(dta, missNum):
    ## Remove the columns with too many missing values.
    # Calculate threshold.
    if missNum < 1:
        missNum = int( len(dta) * missNum )
    
    dta = dta.dropna(axis='columns', thresh=len(dta)-missNum)
    
    ## Fill the remaining missing values by interpolation.
    dta = dta.interpolate()

    return dta


## Generate differencing variable.
def Func_Diff(dta):
    ## Calculate the growth rate of predictors.
    diff = dta.iloc[:, 2:].pct_change()
    ## Concatenate the indexing variables back.
    diff = pd.concat([dta.iloc[:, :2], diff], axis = 1)
    ## Remove the rows of first period.
    diff = diff[ diff['Year'] != startYear ]
    
    ## Change the column names by adding '△' for identification.
    col = dta.columns[:2].tolist()
    for name in dta.columns[2:]:
        name = '△' + name
        col.append(name)

    diff.columns = col
    
    return diff

Preclean = [ Func_Format(dta) for dta in Preclean ]

Data = [ Func_Cleaner(dta, missNum=missNum) for dta in Preclean ]

## The response would be analysed with leading 1 period.
# Response: Birth rate.
y = Data[y_loc][['County', 'Year', response]]

## Remove the response from the predictor dataset.
_col = Data[y_loc].columns.tolist()
_col.remove(response)
Data[y_loc] = Data[y_loc][_col]


Growth = [ Func_Diff(dta) for dta in Data ]


## Remove some rows of data for balanced panel.
# Growth: 1999-2021; Data: 1999-2021; y: 2000-2022.
for _i in range( len(Data) ):
    Data[_i]   = Data[_i][ Data[_i]['Year'] != startYear ]
    Data[_i]   = Data[_i][ Data[_i]['Year'] != endYear ]
    Growth[_i] = Growth[_i][ Growth[_i]['Year'] != endYear ]

y = y[ y['Year'] != startYear ]
y = y[ y['Year'] != startYear+1 ]

## Reset index for data compilation.
for _i in range( len(Data) ):
    Data[_i]   = Data[_i].reset_index(drop=True)
    Growth[_i] = Growth[_i].reset_index(drop=True)
y = y.reset_index(drop=True)


## Seperate the indexing variables.
FixedEff = Data[0].iloc[:, :2]
y = pd.DataFrame( y.iloc[:, 2] )

## To scale up to unify the unit of all variables as percentage.
y = y / 10

## Remove the differencing variables if there are infinity.
for _i in range( len(Growth) ):
    Growth[_i] = Growth[_i].replace([np.inf, -np.inf], np.nan).dropna(axis=1)


## Generate dummy variables.
Dummy_Full = FixedEff
for _county in Dummy_Full['County'].unique():
    Dummy_Full[_county] = 0
    Dummy_Full[_county][ Dummy_Full['County'] == _county ] = 1
## Drop 1 dummy to avoid perfect collinearity.
Dummy = Dummy_Full.iloc[:, :-1]

for _yr in Dummy_Full['Year'].unique():
    Dummy_Full[_yr] = 0
    Dummy_Full[_yr][ Dummy_Full['Year'] == _yr ] = 1

    Dummy[_yr] = 0
    Dummy[_yr][ Dummy['Year'] == _yr ] = 1

Dummy_Full = Dummy_Full.iloc[:, 2:]
Dummy = Dummy.iloc[:, 2:-1]




# %% Exploratory Data Analysis
''' '''
if graph:
    ## Show nullity patterns.
    for _i in range(len(Preclean)):
        msno.matrix(Preclean[_i])
    
    ## Simple visualization of nullity by column.
    for _i in range(len(Preclean)):
        msno.bar(Preclean[_i])


## EDA over chosen variables.
Data[0].columns.tolist()
Data[1].columns.tolist()
Data[2].columns.tolist()
Data[3].columns.tolist()
Data[4].columns.tolist()
Data[5].columns.tolist()

# var_Demo = ['15-64歲人口數(人)', '青壯年人口比率(15-64歲)(％)', '性比例(女=100)',
#             '遷入人口數(人)', '遷出人口數(人)', '人口密度(人/平方公里)',
#             '粗結婚率(‰)', '粗離婚率(‰)']
var_Demo = ['粗結婚率(‰)', '15-49婦女人口數(人)', '15歲以上有偶人口數(人)',
            '15歲以上喪偶人口數(人)', '戶籍登記戶數(戶)', '離婚登記對數(對)',
            '遷入人口數(人)', '遷出人口數(人)', '青壯年人口比率(15-64歲)(％)']
var_Heal = ['平均每一醫療院所服務人數(人/所)', '平均每一醫療院所服務面積(平方公里/所)',
            '執業醫事人員數(人)', '平均每一病床服務之人口數(人/床)',
            '政府部門醫療保健支出(千元)'] 
var_Envi = ['自來水水質檢驗件數(件)'] 
var_Publ = ['經常支出(百萬元)', '歲出政事別結構比-一般政務支出(％)',
            '歲出政事別結構比-經濟發展支出(％)', '歲出政事別結構比-教育科學文化支出(％)',
            '歲出政事別結構比-社會福利支出(％)', '歲出政事別結構比-社區發展及環境保護支出(％)',
            '自籌財源比率(％)', '歲入來源別結構比-補助及協助收入(％)'] 
var_Home = ['家庭收支-平均每戶全年經常性收入(元)', '家庭收支-平均每戶全年經常性支出(元)',
            '家庭現代化設備(每百戶擁有數)-彩色電視機(臺)', '家庭收支-平均消費傾向(％)',
            '家庭收支-平均儲蓄傾向(％)', '家庭收支-自有住宅比率(％)'] 

## Construct the dataset with chosen variables.
EDA_Demo = pd.concat([y, Data[0][var_Demo]], axis=1)
EDA_LabI = pd.concat([y, Data[1].iloc[:, 2:]], axis=1)
EDA_Heal = pd.concat([y, Data[2][var_Heal]], axis=1)
EDA_Envi = pd.concat([y, Data[3][var_Envi]], axis=1)
EDA_Publ = pd.concat([y, Data[4][var_Publ]], axis=1)
EDA_Home = pd.concat([y, Data[5][var_Home]], axis=1)


## Sketch the pairplots.
if graph:
    sns.pairplot(EDA_Demo)
    sns.pairplot(EDA_LabI)
    sns.pairplot(EDA_Heal)
    sns.pairplot(EDA_Envi)
    sns.pairplot(EDA_Publ)
    sns.pairplot(EDA_Home)


## Sketch the heatmap.
if graph:
    sns.heatmap(EDA_Demo.corr(), annot=True)
    sns.heatmap(EDA_LabI.corr(), annot=True)
    sns.heatmap(EDA_Heal.corr(), annot=True)
    sns.heatmap(EDA_Envi.corr(), annot=True)
    sns.heatmap(EDA_Publ.corr(), annot=True)
    sns.heatmap(EDA_Home.corr(), annot=True)
    
## Show summary statistics.
SS_Demo = EDA_Demo.describe()
SS_LabI = Data[1].iloc[:, 2:].describe()
SS_Heal = Data[2][var_Heal].describe()
SS_Envi = Data[3][var_Envi].describe()
SS_Publ = Data[4][var_Publ].describe()
SS_Home = Data[5][var_Home].describe()




# %% Variable Selection

# def Func_AddConst(dta):
#     ## Add constant term.
#     dta['intercept'] = 1
#     ## Put the intercept as b_0.
#     cols = dta.columns.tolist()
#     cols = cols[-1:] + cols[:-1]
#     dta = dta[cols]

#     return dta

## Define the regression model used in the selection process.
def Func_Reg(y, X, model='linear', const=True):
    if model == "linear":
        if const:
            regressor = sm.OLS(y, sm.add_constant(X)).fit()
        else:
            regressor = sm.OLS(y, X).fit()
    elif model == "logistic":
        if const:
            regressor = sm.Logit(y, sm.add_constant(X)).fit()
        else:
            regressor = sm.Logit(y, X).fit()
    return regressor

## Standardize the variables.
def Func_Stdn(dta):
    col = dta.columns
    idx = dta.index
    stdn = skp.StandardScaler()

    dta = stdn.fit_transform(dta)
    dta = pd.DataFrame(dta, index = idx, columns = col)
    
    return dta


## Execute forward selection.
# model: 'linear' / 'logistic';
# var_criteria: Choose the selection criterion for adding additional variable (S1);
# model_criteria: Choose the elimination criterion for choosing the best candidate model (S2);
# signLv: significance level.
def Func_ForwardSelect(y, X_lv, X_diff, model='linear', twodata=False, var_criteria='p-value', stop_criteria='AIC', model_criteria='global', signLv=0.05, std=False):
    if twodata:
        ## Remove low impact variables by standardization.
        if std:
            X_lv   = Func_Stdn(X_lv)
            X_diff = Func_Stdn(X_diff)
            
        ## Merge the dataset of the potential predictors.
        X = pd.concat([ X_lv, X_diff ], axis=1)
    
    else:
        if std:
            X = Func_Stdn(X_lv)
        else:
            X = X_lv

    ## Use list to save the candidate models.
    Candidates = []
    selected_vars = []
    remaining_vars = deepcopy(X_lv.columns.tolist())
    
    # Step 1: Null model.
    ## Build the null model first.
    ## Get the regression result of the null model.
    result = Func_Reg( y, pd.DataFrame([1]*len(y), columns=['intercept']), model=model, const=False )
    Candidates.append(result)
    
    # Step 2: Determine which one is worthy to add among the remaining variables.
    # i.e., finding the candidate model with additional 1 variable.
    ## For each predictor,
    for i in range(X_lv.shape[1]):
        
        ## Initialize to collect the p-values and  of each remaining variable in current round (loop).
        Criteria = pd.DataFrame(columns = ['Var','Pval', 'R^2'])
        
        # Step 2A: Consider all models that augment one additional predictor.        
        ## For each remaining predictor,
        for j in remaining_vars:
            ## Run regression.
            result_lv = Func_Reg(y, X_lv[j], model=model)
            
            result = result_lv
            var = j
            
            if twodata:
                ## Exclude the case that the differencing variable is removed due to inf value previously.
                j_diff = '△' + j
                if j_diff in X_diff.columns.tolist():
                    result_diff = Func_Reg(y, X_diff[j_diff], model=model)
                    
                    ## Compare the p-values between the level and the growth values of the jth variables.
                    if result_diff.pvalues[ j_diff ] < result_lv.pvalues[ j ]:
                        result = result_diff
                        var = j_diff

            
            ## Collect the selection criteria for the regression if the marginal effect is significant.
            if result.pvalues[var] <= signLv:
                Criteria = Criteria.append(
                    pd.DataFrame([[ var, result.pvalues[var], result.rsquared ]],
                                 columns = ['Var','Pval', 'R^2']), 
                                 ignore_index=True)
        
        # Step 2B: Choose the best model as the candidate model among the new models generated in current round.
        ## Break if no significant variable can be added, o/w continue the selection.
        if Criteria.shape[0] > 0:
            ## Choose the best additional variable to augment.
            if var_criteria == 'p-value':
                var_best = Criteria['Var'][ Criteria['Pval'] == Criteria['Pval'].min() ].tolist()[0]
            elif var_criteria == 'R^2':
                var_best = Criteria['Var'][ Criteria['R^2'] == Criteria['R^2'].max() ].tolist()[0]
            
            ## Construct the candidate model.
            result = Func_Reg(y, X[selected_vars+[var_best]], model=model)
            
            # Step 2C: Check whether the process of variable selection should be stopped.
            if stop:
                # Break when the model cannot be improved by the additional variable.
                if stop_criteria == 'AIC':
                    if result.aic >= Candidates[-1].aic:
                        stopInfo = 'Stoppiing criterion reached: AIC: ' + str(result.aic)
                        print(stopInfo)
                        break
                elif stop_criteria == 'BIC':
                    if result.bic >= Candidates[-1].bic:
                        stopInfo = 'Stoppiing criterion reached: BIC: ' + str(result.bic)
                        print(stopInfo)
                        break
                elif stop_criteria == 'R^2' and model == 'linear':
                    if result.rsquared <= Candidates[-1].rsquared:
                        stopInfo = 'Stoppiing criterion reached: r2: '+str(result.rsquared)
                        print(stopInfo)
                        break
                elif stop_criteria == 'adj R^2' and model == 'linear':
                    if result.rsquared <= Candidates[-1].rsquared_adj:
                        stopInfo = 'Stoppiing criterion reached: adj r2: '+str(result.rsquared_adj)
                        print(stopInfo)
                        break
                elif stop_criteria == 'partial f test':
                    f = scipy.stats.f.ppf(q=1-signLv, dfn=1, dfd=len(y)-len(selected_vars)-1)
                    f_0 = (result.ess - Candidates[-1].ess) / (result.ssr / (result.nobs-len(selected_vars)) )
                    if np.abs(f_0) < np.abs(f):
                        stopInfo = 'Stoppiing criterion reached: partial F test: '+str(f_0)
                        print(stopInfo)
                        break
                            
            ## When the stop criterion is not applied, collect all candidate models over the effective model space.
            Candidates.append(result)
            
            ## Remove the variable after either its level or growth value is added to the candidate model.
            if '△' in var_best and twodata:
                remaining_vars.remove(var_best[1:])
            else:
                remaining_vars.remove(var_best)
            
            ## Step 2D (optional): the variable with low marginal effect is not considered in the final model.
            if std:
                # Note that this process would not guarantee all beta greater than the threshold since the marginal effects would be changed in different model specification.
                if np.abs( result.params[var_best] ) >= std:
                    selected_vars.append(var_best)
            else:
                selected_vars.append(var_best)


        else:
            stopInfo = 'Stoppiing criterion reached: no significant variable can be added.'
            print(stopInfo)
            break


    ## Step 3: Select a single best model among the candidate models.
    if model_criteria == 'global':
        Criteria = pd.DataFrame(columns = ['NumOfVars', S2])
        for i in range(len(Candidates)):
            if stop_criteria == 'AIC':
                value = Candidates[i].aic
            elif stop_criteria == 'BIC':
                value = Candidates[i].bic
            elif stop_criteria == 'R^2' and model == 'linear':
                value = Candidates[i].rsquared
            elif stop_criteria == 'adj R^2' and model == 'linear':
                value = Candidates[i].rsquared_adj
            
            ## Group the values of criterion of the candidate models as a table.
            Criteria = Criteria.append(
                           pd.DataFrame([[ i, value ]],
                                        columns = ['NumOfVars', S2]), 
                                        ignore_index=True)

        ## Find the best candidate model.
        if stop_criteria == 'AIC':
            model_best = Criteria['NumOfVars'][ Criteria[S2] == Criteria[S2].min() ].tolist()[0]
        elif stop_criteria == 'BIC':
            model_best = Criteria['NumOfVars'][ Criteria[S2] == Criteria[S2].min() ].tolist()[0]
        elif stop_criteria == 'R^2' and model == 'linear':
            model_best = Criteria['NumOfVars'][ Criteria[S2] == Criteria[S2].max() ].tolist()[0]
        elif stop_criteria == 'adj R^2' and model == 'linear':
            model_best = Criteria['NumOfVars'][ Criteria[S2] == Criteria[S2].max() ].tolist()[0]
        
        model_best = Candidates[model_best]
    
    # 'final' means extract the candidate model with the most number of variables as the best model.
    elif model_criteria == 'final':
        model_best = Candidates[-1]
    
    return model_best


## Extract the predictors and merge them with same category.
Data   = [ dta.iloc[:, 2:] for dta in Data ]
Growth = [ dta.iloc[:, 2:] for dta in Growth ]

## Merge the predictor dataset.
Predictors = pd.DataFrame()
for _i in range(len(Data)):
    Predictors = pd.concat([ Predictors, Data[_i] ], axis=1)
    Predictors = pd.concat([ Predictors, Growth[_i] ], axis=1)
    

## Run variable selection for each category.
# Remove coefficients with low marginal effect.
Cat_forward = []
for _i in range(len(Data)):
    Cat_forward.append(
        Func_ForwardSelect(y, Data[_i], Growth[_i], twodata=diff_either, model=model, var_criteria=S1, stop_criteria=S2, model_criteria=S3, signLv=significance, std=0.025)
        )

Cat_forward[2].summary()


# No coefficient is removed.
Cat_forward_noStd = []
for _i in range(len(Data)):
    Cat_forward_noStd.append(
        Func_ForwardSelect(y, Data[_i], Growth[_i], twodata=diff_either, model=model, var_criteria=S1, stop_criteria=S2, model_criteria=S3, signLv=significance, std=0.01)
        )

Cat_forward_noStd[3].summary()





# %% Model Running


## Choose the variables pass the variable selection under each category.
var_LabI = ['勞動力參與率(％)',
            '△老年基本保證年金核付人數(人)'
            ]

var_Heal = ['平均每一護理人員服務之人口數(人/人)', 
            '△平均每一醫療院所服務人數(人/所)',
            '△護理人員數(人)'
            ]

var_Envi = ['一般廢棄物回收率(%)',
            '△一般廢棄物妥善處理率(％)',
            '自來水水質檢驗件數(件)',
            '清運人員數(人)',
            '事業廢水列管家數(家)'
            ]

var_Publ = ['歲出政事別結構比-一般政務支出(％)',
            '歲入來源別-補助及協助收入(百萬元)',
            '歲入來源別結構比-其他收入(％)',
            '自籌財源比率(％)',
            '歲出政事別結構比-經濟發展支出(％)',
            '歲出政事別結構比-退休撫卹支出(％)',
            '補助及協助收入依存度(％)',
            '歲出政事別結構比-社會福利支出(％)',
            '歲入來源別結構比-補助及協助收入(％)',
            '歲出政事別-退休撫卹支出(百萬元)',
            '△經常支出(百萬元)'
            ]

var_Home = ['家庭現代化設備(每百戶擁有數)-彩色電視機(臺)',
            '家庭現代化設備(每百戶擁有數)-報紙(份)',
            '家庭收支-平均每戶就業人數(人)',
            '家庭現代化設備(每百戶擁有數)-家用電腦(臺)',
            '家庭收支-平均每戶全年經常性收入(元)',
            '家庭收支-平均消費傾向(％)',
            '家庭收支-平均儲蓄傾向(％)',
            '家庭收支-平均每戶可支配所得(元)',
            '家庭收支-平均每戶全年經常性支出(元)',
            '飲食費(不含家外食物)占消費支出比率(%)',
            ]


var_LabI = [#'勞動力參與率(％)',
            '△老年基本保證年金核付人數(人)'
            ]

var_Heal = [# '平均每一護理人員服務之人口數(人/人)', 
            # '△平均每一醫療院所服務人數(人/所)',
            # '△護理人員數(人)'
            ]

var_Envi = [#'一般廢棄物回收率(%)',
            # '△一般廢棄物妥善處理率(％)',
            # '自來水水質檢驗件數(件)',
            # '清運人員數(人)',
            # '事業廢水列管家數(家)'
            ]

var_Publ = ['歲出政事別結構比-一般政務支出(％)',
            # '歲入來源別-補助及協助收入(百萬元)',
            '歲入來源別結構比-其他收入(％)',
            '自籌財源比率(％)',
            # '歲出政事別結構比-經濟發展支出(％)',
            # '歲出政事別結構比-退休撫卹支出(％)',
            # '補助及協助收入依存度(％)',
            '歲出政事別結構比-社會福利支出(％)',
            '歲入來源別結構比-補助及協助收入(％)',
            # '歲出政事別-退休撫卹支出(百萬元)',
            # '△經常支出(百萬元)'
            ]

var_Home = ['家庭現代化設備(每百戶擁有數)-彩色電視機(臺)',
            # '家庭現代化設備(每百戶擁有數)-報紙(份)',
            '家庭收支-平均每戶就業人數(人)',
            # '家庭現代化設備(每百戶擁有數)-家用電腦(臺)',
            # '家庭收支-平均每戶全年經常性收入(元)',
            '家庭收支-平均消費傾向(％)',
            # '家庭收支-平均儲蓄傾向(％)',
            # '家庭收支-平均每戶可支配所得(元)',
            # '家庭收支-平均每戶全年經常性支出(元)',
            # '飲食費(不含家外食物)占消費支出比率(%)',
            ]

var_chosen = var_LabI + var_Heal + var_Envi + var_Publ + var_Home

## Presentation
# '△食品衛生查驗件數(件)' for _std = 0.05.
var_chosen = [
              '△老年基本保證年金核付人數(人)',
              '△護理人員數(人)',
              '清運人員數(人)',
              '事業廢水列管家數(家)',
              '自籌財源比率(％)',
              '歲出政事別結構比-社會福利支出(％)',
              '歲入來源別結構比-補助及協助收入(％)',
              '家庭收支-平均消費傾向(％)',
              '平均每一醫療院所服務人數(人/所)'
              ]


result_chosen = Func_ForwardSelect(y, Predictors[var_chosen], None, twodata=False, model=model, var_criteria=S1, stop_criteria=S2, model_criteria=S3, signLv=significance, std=0)


result_chosen.summary()




X = pd.concat([ Predictors[var_chosen], Dummy ], axis=1)
Func_Reg(y, X).summary()

var_remove = ['家庭收支-平均每戶就業人數(人)',
              '家庭現代化設備(每百戶擁有數)-彩色電視機(臺)',
              '歲出政事別結構比-一般政務支出(％)',
              '歲入來源別結構比-其他收入(％)',
              ]
var_chosen_2 = [ var for var in var_chosen if var not in var_remove ]

X = pd.concat([ Predictors[var_chosen_2], Dummy ], axis=1)
Func_Reg(y, X).summary()



## VIF
# For each category.
## Extract the variables of the best candidate model.
_col = Cat_forward[2].params.index.tolist()
_col.remove('const')

if graph:
    VIF_j = pd.Series([VIF(Predictors[_col].values, i) 
                        for i in range(Predictors[_col].shape[1])], 
                        index=Predictors[_col].columns)
    
    
    VIF = pd.Series([VIF(Predictors[var_chosen_2].values, i) 
                     for i in range(Predictors[var_chosen_2].shape[1])], 
                    index=Predictors[var_chosen_2].columns)
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
    
    var_chosen_6 = ['家庭收支-平均消費傾向(％)',
                    '自籌財源比率(％)',
                    '歲出政事別結構比-社會福利支出(％)',
                    '歲入來源別結構比-補助及協助收入(％)',
                    ]
    
    VIF_6 = pd.Series([VIF(Predictors[var_chosen_6].values, i) 
                     for i in range(Predictors[var_chosen_6].shape[1])], 
                    index=Predictors[var_chosen_6].columns)
    
    _col = Cat_forward_noStd[2].params.index.tolist()[1:]
    VIF_health = pd.Series([VIF(Predictors[_col].values, i) 
                     for i in range(Predictors[_col].shape[1])], 
                    index=Predictors[_col].columns)



## Final model.
var_chosen_3 = [#'△老年基本保證年金核付人數(人)',
                '自籌財源比率(％)',
                '歲出政事別結構比-社會福利支出(％)',
                '歲入來源別結構比-補助及協助收入(％)',
                ]
# var_chosen_3.remove('家庭收支-平均消費傾向(％)')


## To scale up to unify the unit of all variables as percentage.
X3 = Predictors[var_chosen_3]
# X3['△老年基本保證年金核付人數(人)'] = X3['△老年基本保證年金核付人數(人)'] * 100

y.describe()
X3 = pd.concat([ X3, Dummy ], axis=1)
final = Func_Reg(y, X3)
final.summary()

if graph:
    VIF = pd.Series([VIF(Predictors[var_chosen_3].values, i) 
                     for i in range(Predictors[var_chosen_3].shape[1])], 
                    index=Predictors[var_chosen_3].columns)


## Robustness Checking
var_chosen_5 = ['自籌財源比率(％)',
                '歲入來源別結構比-補助及協助收入(％)',
                # '△老年基本保證年金核付人數(人)',
                '歲出政事別結構比-社會福利支出(％)',
                '歲出政事別結構比-一般政務支出(％)',
                '歲出政事別結構比-經濟發展支出(％)'
                ]

X5 = Predictors[var_chosen_5]
# X5['△老年基本保證年金核付人數(人)'] = X5['△老年基本保證年金核付人數(人)'] * 100

X5 = pd.concat([ X5, Dummy ], axis=1)
final = Func_Reg(y, X5)
final.summary()

final.bse


# %% Partial F-test

## Full model SSR.
ess_full = Func_Reg(y, X3).ess

## Find Reduced model SSR.
Func_Reg(y, Dummy).summary()

F_0 = (ess_full - Func_Reg(y, Dummy).ess) / 3 / (Func_Reg(y, X3).ess / (Func_Reg(y, X3).df_model-1))

## The f statistic threshold.
F = scipy.stats.f.ppf(q=1-.05, dfn=3, dfd=354)

print( np.abs(F_0) > F )




# %% Model Diagnosis



plt.rcParams['axes.unicode_minus'] = False


resid_std = (final.resid - final.resid.mean()) / final.resid.std()


## Standardized residual plot.
plt.figure(figsize=(8,8))
plt.scatter(final.fittedvalues, resid_std)
plt.xlabel('Fitted value', fontsize=20)
plt.ylabel('Standardized Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title('Standardized Residual Plot', fontsize=20)
plt.show()



## Residual qq plot.


pg.qqplot(final.resid, dist='norm', confidence=.95)

# sm.qqplot(final.resid, line='45')

final.resid.hist()

print(scipy.stats.kurtosis(final.resid, axis=0, bias=True))

print(scipy.stats.skew(final.resid, axis=0, bias=True))

resid_std[resid_std>2.5]
resid_std[resid_std<-2.5]



## Leverage Plot
resid_student = final.resid / final.resid.std() / (1 - final.get_influence().hat_matrix_diag)**0.5



plt.figure(figsize=(8,8))
plt.scatter(final.get_influence().hat_matrix_diag, resid_student)
plt.xlabel('Leverage', fontsize=20)
plt.ylabel('Studentized Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.title('Leverage Plot', fontsize=20)
plt.show()


resid_student[resid_student>3]
resid_student[resid_student<-2.5]



## Partial Regression (Leverage)

def Func_PartialRegPlot(var, width=0.01):
    _col = X3.columns.tolist()
    _col.remove(var)

    p_reg_y_1 = Func_Reg(y, X3[_col])
    p_reg_X_1 = Func_Reg(X3[var], X3[_col])

    _x_max = p_reg_X_1.resid.max() + width
    _x_min = p_reg_X_1.resid.min() - width
    _y_max = p_reg_y_1.resid.max() + width
    _y_min = p_reg_y_1.resid.min() - width
    _slope = Func_Reg(p_reg_y_1.resid, p_reg_X_1.resid).params.iloc[0]


    plt.figure(figsize=(8,8))
    plt.scatter(p_reg_X_1.resid, p_reg_y_1.resid)
    plt.xlabel(var + '|others', fontsize=20)
    plt.ylabel('粗出生率(‰)|others', fontsize=20)
    plt.title('Leverage Plot', fontsize=20)
    plt.plot([_x_min, _x_max], [_x_min*_slope, _x_max*_slope], color='r')
    plt.axis([_x_min, _x_max, _y_min, _y_max])
    plt.show()

Func_PartialRegPlot('△老年基本保證年金核付人數(人)')
Func_PartialRegPlot('自籌財源比率(％)')
Func_PartialRegPlot('歲出政事別結構比-社會福利支出(％)')
Func_PartialRegPlot('歲入來源別結構比-補助及協助收入(％)')




def leverage_plot(self, ax=None, high_leverage_threshold=False, cooks_threshold='baseR'):
    """
    Residual vs Leverage plot

    Points falling outside Cook's distance curves are considered observation that can sway the fit
    aka are influential.
    Good to have none outside the curves.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(
        self.leverage,
        self.residual_norm,
        alpha=0.5);

    sns.regplot(
        x=self.leverage,
        y=self.residual_norm,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
        ax=ax)

    # annotations
    leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
    for i in leverage_top_3:
        ax.annotate(
            i,
            xy=(self.leverage[i], self.residual_norm[i]),
            color = 'C3')

    factors = []
    if cooks_threshold == 'baseR' or cooks_threshold is None:
        factors = [1, 0.5]
    elif cooks_threshold == 'convention':
        factors = [4/self.nresids]
    elif cooks_threshold == 'dof':
        factors = [4/ (self.nresids - self.nparams)]
    else:
        raise ValueError("threshold_method must be one of the following: 'convention', 'dof', or 'baseR' (default)")
    for i, factor in enumerate(factors):
        label = "Cook's distance" if i == 0 else None
        xtemp, ytemp = self.__cooks_dist_line(factor)
        ax.plot(xtemp, ytemp, label=label, lw=1.25, ls='--', color='red')
        ax.plot(xtemp, np.negative(ytemp), lw=1.25, ls='--', color='red')

    if high_leverage_threshold:
        high_leverage = 2 * self.nparams / self.nresids
        if max(self.leverage) > high_leverage:
            ax.axvline(high_leverage, label='High leverage', ls='-.', color='purple', lw=1)

    ax.axhline(0, ls='dotted', color='black', lw=1.25)
    ax.set_xlim(0, max(self.leverage)+0.01)
    ax.set_ylim(min(self.residual_norm)-0.1, max(self.residual_norm)+0.1)
    ax.set_title('Residuals vs Leverage', fontweight="bold")
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')
    plt.legend(loc='best')
    return ax




#%% LASSO - Level Variables

## Merge the predictor datasets seperately.
Predictors_lv = pd.DataFrame()
Predictors_gr = pd.DataFrame()
for _i in range(len(Data)):
    Predictors_lv = pd.concat([ Predictors_lv, Data[_i] ], axis=1)
    Predictors_gr = pd.concat([ Predictors_gr, Growth[_i] ], axis=1)


## Standardize the data for lasso.
scaler = skp.StandardScaler(with_mean=True,  with_std=True)
Y_stdn = scaler.fit_transform(y)

scaler = skp.StandardScaler(with_mean=True,  with_std=True)
X_lv_stdn = scaler.fit_transform(Predictors_lv)


## Choose the best regularized hyperparameter by 5-fold cross-validation.
K = 5
kfold = skm.KFold(K,
                  random_state=2023,
                  shuffle=True)

# Scan λ from 10e4 to 10e-4 with log-spaced 100 samples.
alphas = np.logspace(-4, 4, 100)

lassoCV = skl.ElasticNetCV(l1_ratio=1,
                           alphas = alphas,
                           cv=kfold)

Lasso_lv = lassoCV.fit(X_lv_stdn, Y_stdn)

# optimal value of λ.
# The amount of penalization chosen by cross validation along alphas_.
Lasso_lv.alpha_



## plot of the cross-validation error.
lassoCV_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(Lasso_lv.alphas_),
            # mse_path_: Mean square error for the test set on each fold, varying alpha.
            Lasso_lv.mse_path_.mean(axis=1),
            yerr=Lasso_lv.mse_path_.std(axis=1) / np.sqrt(K))
ax.axvline(-np.log(Lasso_lv.alpha_), c='k', ls='--')
# ax.set_ylim([50000,250000])
ax.set_xlabel('-log(λ)', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)


## Collect the coefficients of Lasso reg under each λ.
lasso = skl.Lasso()
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_lv_stdn, Y_stdn) 
    coefs.append(lasso.coef_)

# Associated with each value of λ is a vector of ridge regression coefficients, 
# that can be accessed by a column of soln_array. 
# In this case, soln_array is a matrix, with 10 columns (one for each predictor) 
# and 100 rows (one for each value of λ).
soln_array = np.array(coefs)

# Take transpose to reverse the position of λ and predictors.
soln_path = pd.DataFrame(soln_array,
                         columns=Predictors_lv.columns,
                         index=-np.log(alphas))
soln_path.index.name = 'negative log(lambda)'

# Randomly pick serveral variables as demonstration.
np.random.seed(2023)
idx_show = np.random.choice(np.arange(0, len(Predictors_lv.columns)), size=15, replace=False)
soln_path_reduced = soln_path.iloc[:, idx_show]


# Plot the paths to get a sense of how the coefficients vary with λ.
path_fig, ax = plt.subplots(figsize=(16,8))
# soln_path.plot(ax=ax, legend=False)
soln_path_reduced.plot(ax=ax)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
path_fig.tight_layout()
plt.show()



#%% LASSO - Growth Variables

scaler = skp.StandardScaler(with_mean=True,  with_std=True)
X_gr_stdn = scaler.fit_transform(Predictors_gr)

## Choose the best regularized hyperparameter by 5-fold cross-validation.
K = 5
kfold = skm.KFold(K,
                  random_state=2023,
                  shuffle=True)

# Scan λ from 10e4 to 10e-4 with log-spaced 100 samples.
alphas = np.logspace(-4, 4, 100)

lassoCV = skl.ElasticNetCV(l1_ratio=1,
                           alphas = alphas,
                           cv=kfold)

Lasso_gr = lassoCV.fit(X_gr_stdn, Y_stdn)

# optimal value of λ.
# The amount of penalization chosen by cross validation along alphas_.
Lasso_gr.alpha_


## plot of the cross-validation error.
lassoCV_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(Lasso_gr.alphas_),
            # mse_path_: Mean square error for the test set on each fold, varying alpha.
            Lasso_gr.mse_path_.mean(axis=1),
            yerr=Lasso_gr.mse_path_.std(axis=1) / np.sqrt(K))
ax.axvline(-np.log(Lasso_gr.alpha_), c='k', ls='--')
# ax.set_ylim([50000,250000])
ax.set_xlabel('-log(λ)', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)



## Collect the coefficients of Lasso reg under each λ.
lasso = skl.Lasso()
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_gr_stdn, Y_stdn) 
    coefs.append(lasso.coef_)

# Associated with each value of λ is a vector of ridge regression coefficients, 
# that can be accessed by a column of soln_array. 
# In this case, soln_array is a matrix, with 10 columns (one for each predictor) 
# and 100 rows (one for each value of λ).
soln_array = np.array(coefs)

# Take transpose to reverse the position of λ and predictors.
soln_path = pd.DataFrame(soln_array,
                         columns=Predictors_gr.columns,
                         index=-np.log(alphas))
soln_path.index.name = 'negative log(lambda)'

# Randomly pick serveral variables as demonstration.
np.random.seed(2023)
idx_show = np.random.choice(np.arange(0, len(Predictors_gr.columns)), size=16, replace=False)
soln_path_reduced = soln_path.iloc[:, idx_show]


# Plot the paths to get a sense of how the coefficients vary with λ.
path_fig, ax = plt.subplots(figsize=(16,8))
# soln_path.plot(ax=ax, legend=False)
soln_path_reduced.plot(ax=ax)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
path_fig.tight_layout()
plt.show()




#%% Model Construction

## Choose the variables selected by Lasso.
coeff_lasso_lv = pd.Series(Lasso_lv.coef_)
idx_lasso_lv   = coeff_lasso_lv[ np.abs(coeff_lasso_lv) > 10**-5 ].index
X_new_lv = Predictors_lv.iloc[ :, idx_lasso_lv ]

coeff_lasso_gr = pd.Series(Lasso_gr.coef_)
idx_lasso_gr   = coeff_lasso_gr[ np.abs(coeff_lasso_gr) > 10**-5 ].index
X_new_gr = Predictors_gr.iloc[ :, idx_lasso_gr ]


Predictors_lv.columns[idx_lasso_lv]
var_selected_lv = [#'65歲以上人口數(人)', 
                   #'幼年人口比率(0-14歲)(％)', 
                   #'人口總增加率(‰)', 
                   #'粗結婚率(‰)', 
                   #'粗離婚率(‰)',
                   #'15歲以上人口未婚比率(％)', 
                   #'15歲以上人口離婚比率(％)', 
                   #'勞動力參與率(％)', 
                   #'失業率(％)',
                   #'年齡別失業率-25-44歲(％)', 
                   #'都市計畫公共設施用地面積(公頃)', 
                   '車輛成長率-小客車(％)', 
                   '車輛成長率-機車(％)',
                   '公司登記現有資本額(百萬元)', 
                   '平均每人稅賦(元)', 
                   '平均每一醫療院所服務面積(平方公里/所)', 
                   #'每萬人口病床數(床/萬人)',
                   #'平均每一病床服務之人口數(人/床)', 
                   '醫療保健支出占政府支出比率(年度)(％)', 
                   #'法定傳染病患者數(人)',
                   '每十萬人法定傳染病患者(人/十萬人)', 
                   '食品衛生查驗件數(件)', 
                   #'廚餘回收量(公噸)', 
                   #'垃圾回收清除車輛數(輛)',
                   #'一般廢棄物妥善處理率(％)', 
                   #'自來水水質檢驗件數(件)', 
                   #'事業廢水列管家數(家)', 
                   #'經常收支賸餘(百萬元)',
                   #'歲入來源別-規費收入(百萬元)', 
                   #'歲入來源別-補助及協助收入(百萬元)', 
                   #'歲入來源別-財產收入(百萬元)',
                   '歲入來源別結構比-財產收入(％)', 
                   #'歲出政事別-社會福利支出(百萬元)', 
                   #'歲出政事別-退休撫卹支出(百萬元)',
                   #'歲出政事別結構比-一般政務支出(％)', 
                   #'歲出政事別結構比-經濟發展支出(％)', 
                   #'歲出政事別結構比-教育科學文化支出(％)',
                   #'歲出政事別結構比-社會福利支出(％)', 
                   #'歲出政事別結構比-社區發展及環境保護支出(％)', 
                   '歲出政事別結構比-退休撫卹支出(％)',
                   #'稅課收入-統籌分配稅收入(百萬元)', 
                   #'自籌財源(百萬元)', 
                   #'自籌財源比率(％)', 
                   '融資需求-歲入歲出差短(百萬元)',
                   #'家庭收支-平均戶內人口數(人)', 
                   #'家庭收支-平均消費傾向(％)', 
                   #'家庭收支-平均儲蓄傾向(％)',
                   #'家庭現代化設備(每百戶擁有數)-有線電視頻道設備(戶)', 
                   #'家庭現代化設備(每百戶擁有數)-家用電腦(臺)',
                   #'家庭現代化設備(每百戶擁有數)-彩色電視機(臺)', 
                   #'家庭現代化設備(每百戶擁有數)-報紙(份)',
                   #'家庭現代化設備(每百戶擁有數)-電話機(臺)', 
                   #'家庭現代化設備(每百戶擁有數)-機車(輛)', 
                   #'平均每人居住面積(坪)',
                   #'平均每戶書報雜誌文具支出占消費支出比率(％)', 
                   #'飲食費(含家外食物)占消費支出比率(%)',
                   #'飲食費(不含家外食物)占消費支出比率(%)'
                  ]


## Compute the VIF between the variables selected.
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

VIF_lv = pd.Series([VIF(Predictors[var_selected_lv].values, i) 
                   for i in range(Predictors[var_selected_lv].shape[1])], 
                   index=Predictors[var_selected_lv].columns)



X_lv = pd.concat([ Predictors_lv[var_selected_lv], Dummy ], axis=1)
final_lv = Func_Reg(y, X_lv)
final_lv.summary()


final_lv.params
final_lv.bse
final_lv.tvalues
final_lv.pvalues

Table_lv = pd.DataFrame({'Coefficient': final_lv.params,
                         'Standard Error': final_lv.bse,
                         't-value': final_lv.tvalues,
                         'p-value': final_lv.pvalues
                         }
    )
Table_lv_show = Table_lv.iloc[: len(var_selected_lv)+1]
Table_lv_show['t-value'] = Table_lv_show['t-value'].abs()




Predictors_gr.columns[idx_lasso_gr]
# var_selected_gr = Predictors_gr.columns[idx_lasso_gr].tolist()
var_selected_gr = [#'△戶籍登記戶數(戶)', 
                     #'△0-14歲人口數(人)', 
                     #'△青壯年人口比率(15-64歲)(％)', 
                     '△老化指數()', 
                     #'△遷入人口數(人)',
                     #'△結婚登記對數(對)', 
                     '△離婚登記對數(對)', 
                     '△中低收入老人生活津貼核定人數(人)', 
                     '△老年基本保證年金核付人數(人)',
                     '△老年基本保證年金核付金額(元)', 
                     '△公司登記現有資本額(百萬元)', 
                     '△執業醫事人員數(人)', 
                     '△執行機關資源回收量(公噸)',
                     '△垃圾回收清除車輛數(輛)', 
                     '△清運人員數(人)', 
                     '△歲入來源別-規費收入(百萬元)', 
                     '△歲入來源別-其他收入(百萬元)'
                     ]


## Compute the VIF between the variables selected.
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

VIF_gr = pd.Series([VIF(Predictors[var_selected_gr].values, i) 
                   for i in range(Predictors[var_selected_gr].shape[1])], 
                   index=Predictors[var_selected_gr].columns)



X_gr = pd.concat([ Predictors_gr[var_selected_gr], Dummy ], axis=1)
final_gr = Func_Reg(y, X_gr)
final_gr.summary()




var_combined = ['車輛成長率-小客車(％)',
                '車輛成長率-機車(％)',
                '公司登記現有資本額(百萬元)',
                '平均每人稅賦(元)',
                #'平均每一醫療院所服務面積(平方公里/所)',
                #'醫療保健支出占政府支出比率(年度)(％)',
                #'每十萬人法定傳染病患者(人/十萬人)',
                #'食品衛生查驗件數(件)',
                #'歲入來源別結構比-財產收入(％)',
                #'歲出政事別結構比-退休撫卹支出(％)',
                '融資需求-歲入歲出差短(百萬元)',
                #'△老化指數()', 
                #'△離婚登記對數(對)', 
                #'△中低收入老人生活津貼核定人數(人)', 
                #'△老年基本保證年金核付人數(人)',
                '△老年基本保證年金核付金額(元)', 
                #'△公司登記現有資本額(百萬元)', 
                '△執業醫事人員數(人)', 
                #'△執行機關資源回收量(公噸)',
                #'△垃圾回收清除車輛數(輛)', 
                #'△清運人員數(人)', 
                #'△歲入來源別-規費收入(百萬元)', 
                #'△歲入來源別-其他收入(百萬元)'
                ]


## Compute the VIF between the variables selected.
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

VIF_combined = pd.Series([VIF(Predictors[var_combined].values, i) 
                   for i in range(Predictors[var_combined].shape[1])], 
                   index=Predictors[var_combined].columns)



X_combined = pd.concat([ Predictors[var_combined], Dummy ], axis=1)
final_combined = Func_Reg(y, X_combined)
final_combined.summary()

final_combined.rsquared
final_combined.rsquared_adj


Table_final = pd.DataFrame({'Coefficient': final_combined.params,
                         'Standard Error': final_combined.bse,
                         't-value': final_combined.tvalues,
                         'p-value': final_combined.pvalues
                         }
    )
Table_final_show = Table_final.iloc[: len(var_combined)+1]
Table_final_show['t-value'] = Table_final_show['t-value'].abs()




#%% Partial F-Test (Fixed Effect)

## Full model SSR.
ess_full = final_combined.ess

## Only time effect model SSR.
X_time = pd.concat([ Predictors[var_combined], Dummy.iloc[:, 21:] ], axis=1)
final_time = Func_Reg(y, X_time)
final_time.summary()
ess_time = final_time.ess

## Only county effect model SSR.
X_county = pd.concat([ Predictors[var_combined], Dummy.iloc[:, :21] ], axis=1)
final_county = Func_Reg(y, X_county)
final_county.summary()
ess_county = final_county.ess

## Find Reduced model SSR.
final_reduced = Func_Reg(y, Predictors[var_combined])
final_reduced.summary()
ess_reduced = final_reduced.ess

## Only fixed effect model SSR.
ess_dummy = Func_Reg(y, Dummy).ess


## When compared with the reduced model, the F-values and the F statistics thresholds:
# Full.
(ess_full - ess_reduced) / 38 / (ess_full / (final_combined.df_model-1))
scipy.stats.f.ppf(q=1-.05, dfn=38, dfd=final_combined.df_resid)
scipy.stats.f.ppf(q=1-.1, dfn=38, dfd=final_combined.df_resid)

# Only time effect.
(ess_time - ess_reduced) / 21 / (ess_time / (final_time.df_model-1))
scipy.stats.f.ppf(q=1-.05, dfn=21, dfd=final_time.df_resid)
scipy.stats.f.ppf(q=1-.1, dfn=21, dfd=final_time.df_resid)

# Only county effect.
(ess_county - ess_reduced) / 17 / (ess_county / (final_county.df_model-1))
scipy.stats.f.ppf(q=1-.05, dfn=17, dfd=final_county.df_resid)
scipy.stats.f.ppf(q=1-.1, dfn=21, dfd=final_time.df_resid)






# %% Model Diagnosis (LASSO)



plt.rcParams['axes.unicode_minus'] = False


resid_std = (final_combined.resid - final_combined.resid.mean()) / final_combined.resid.std()


## Standardized residual plot.
plt.figure(figsize=(8,8))
plt.scatter(final_combined.fittedvalues, resid_std)
plt.xlabel('Fitted value', fontsize=20)
plt.ylabel('Standardized Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title('Standardized Residual Plot', fontsize=20)
plt.show()



## Residual qq plot.
pg.qqplot(final_combined.resid, dist='norm', confidence=.95)


# sm.qqplot(final_combined.resid, line='45')

final_combined.resid.hist()

print(scipy.stats.kurtosis(final_combined.resid, axis=0, bias=True))

print(scipy.stats.skew(final_combined.resid, axis=0, bias=True))

resid_std[resid_std>2.5]
resid_std[resid_std<-2.5]



## Leverage Plot
resid_student = final_combined.resid / final_combined.resid.std() / (1 - final_combined.get_influence().hat_matrix_diag)**0.5

plt.figure(figsize=(8,8))
plt.scatter(final_combined.get_influence().hat_matrix_diag, resid_student)
plt.xlabel('Leverage', fontsize=20)
plt.ylabel('Studentized Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.title('Leverage Plot', fontsize=20)
plt.show()


resid_student[resid_student>2.5]
resid_student[resid_student<-2.5]

final_combined.get_influence().hat_matrix_diag.argmax()




#%% Supplementary Partial Regression

## Recall the mid term model
midTerm = Func_Reg(y, X3)
midTerm.summary()


resid_std_MT = (midTerm.resid - midTerm.resid.mean()) / midTerm.resid.std()

## Check for the exogenity of the endogenous variables.
check_MT_Var1 = Func_Reg(resid_std_MT, X3.iloc[:, 0])
check_MT_Var1.summary()

check_MT_Var2 = Func_Reg(resid_std_MT, X3.iloc[:, 1])
check_MT_Var2.summary()

check_MT_Var3 = Func_Reg(resid_std_MT, X3.iloc[:, 2])
check_MT_Var3.summary()


resid_std = (final_combined.resid - final_combined.resid.mean()) / final_combined.resid.std()


## Standardized residual plot.
matplotlib.rcParams['text.usetex'] = True

plt.figure(figsize=(8,8))
plt.scatter(X_combined.iloc[:, 0], resid_std)
plt.xlabel(r'$\triangle$ Num of Motorbikes', fontsize=20)
plt.ylabel('Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title(r'Residual Plot against $\triangle$ Num of Motorbikes', fontsize=20)
plt.show()


plt.figure(figsize=(8,8))
plt.scatter(X_combined.iloc[:, 1], resid_std)
plt.xlabel(r'$\triangle$ Num of Sedans', fontsize=20)
plt.ylabel('Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title(r'Residual Plot against $\triangle$ Num of Sedans', fontsize=20)
plt.show()


plt.figure(figsize=(8,8))
plt.scatter(X_combined.iloc[:, 2], resid_std)
plt.xlabel(r'Capital of Firms', fontsize=20)
plt.ylabel('Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title(r'Residual Plot against Capital of Firms', fontsize=20)
plt.show()


plt.figure(figsize=(8,8))
plt.scatter(X_combined.iloc[:, 3], resid_std)
plt.xlabel(r'Tax per Capita', fontsize=20)
plt.ylabel('Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title(r'Residual Plot against Tax per Capita', fontsize=20)
plt.show()


plt.figure(figsize=(8,8))
plt.scatter(X_combined.iloc[:, 4], resid_std)
plt.xlabel(r'Financial Deficit', fontsize=20)
plt.ylabel('Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title(r'Residual Plot against Financial Deficit', fontsize=20)
plt.show()


plt.figure(figsize=(8,8))
plt.scatter(X_combined.iloc[:, 5], resid_std)
plt.xlabel(r'$\triangle$ Pension Payments', fontsize=20)
plt.ylabel('Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title(r'Residual Plot against $\triangle$ Pension Payments', fontsize=20)
plt.show()


plt.figure(figsize=(8,8))
plt.scatter(X_combined.iloc[:, 6], resid_std)
plt.xlabel(r'$\triangle$ Med Person', fontsize=20)
plt.ylabel('Residual', fontsize=20)
plt.axhline(0, c='r', ls='--')
plt.axhline(2.5, c='k', ls='-')
plt.axhline(-2.5, c='k', ls='-')
plt.title(r'Residual Plot against $\triangle$ Med Person', fontsize=20)
plt.show()




# %% Appendix - Figures

## Draw line plot.
startYear = 2001
endYear   = 2022

Preclean_2 = [ Func_FillCounty(dta) for dta in RawTables ]
for _i in range(len(Preclean_2)):
    Preclean_2[_i] = Preclean_2[_i][ Preclean_2[_i]['Year'] >= startYear ]
    Preclean_2[_i] = Preclean_2[_i][ Preclean_2[_i]['Year'] <= endYear ]
Data_2 = [ Func_Cleaner(dta, missNum=missNum) for dta in Preclean_2 ]

y_2 = Data_2[y_loc][['County', 'Year', response]]
Draw = Data_2[4][['自籌財源比率(％)', '歲出政事別結構比-社會福利支出(％)', 
          '歲出政事別結構比-經濟發展支出(％)', '歲出政事別結構比-教育科學文化支出(％)']]

Draw = pd.concat( [ y_2, Draw ], axis=1 )

Draw_1 = Data_2[4][[#'自籌財源比率(％)', 
                    '歲出政事別結構比-社會福利支出(％)', 
                    # '歲出政事別結構比-經濟發展支出(％)', 
                    # '歲出政事別結構比-教育科學文化支出(％)'
                    ]]

Draw_1 = pd.concat( [ y_2, Draw_1 ], axis=1 )

# Draw['歲入來源別結構比-補助及協助收入(％)'] = pd.to_numeric( Draw['歲入來源別結構比-補助及協助收入(％)'] )

Draw.columns = ['County', 'Year', '粗出生率(％)', '自籌財源比率(％)', '歲出政事別結構比-社會福利支出(％)', '歲出政事別結構比-經濟發展支出(％)', '歲出政事別結構比-教育科學文化支出(％)']

Draw_1.columns = ['County', 'Year', '粗出生率(％)', '歲出政事別結構比-社會福利支出(％)']

Draw_year = Draw.groupby('Year').agg({
    '粗出生率(％)': 'mean',
    '歲出政事別結構比-社會福利支出(％)': 'mean',
    '自籌財源比率(％)': 'mean',
    # '歲入來源別結構比-補助及協助收入(％)': 'mean',
    '歲出政事別結構比-經濟發展支出(％)': 'mean',
    '歲出政事別結構比-教育科學文化支出(％)': 'mean',
    
})


Draw_year_1 = Draw_1.groupby('Year').agg({
    '粗出生率(％)': 'mean',
    '歲出政事別結構比-社會福利支出(％)': 'mean',
    # '自籌財源比率(％)': 'mean',
    # '歲入來源別結構比-補助及協助收入(％)': 'mean',
    # '歲出政事別結構比-經濟發展支出(％)': 'mean',
    # '歲出政事別結構比-教育科學文化支出(％)': 'mean',
    
})


Draw_county = Draw.groupby('County').agg({
    '粗出生率(％)': 'mean',
    '歲出政事別結構比-社會福利支出(％)': 'mean',
    '自籌財源比率(％)': 'mean',
    # '歲入來源別結構比-補助及協助收入(％)': 'mean',
    '歲出政事別結構比-經濟發展支出(％)': 'mean',
    '歲出政事別結構比-教育科學文化支出(％)': 'mean',
})






# Create a plot
plt.figure(figsize=(12, 6))
for column in Draw_year.columns:
    plt.plot(Draw_year.index, Draw_year[column], label=column)

plt.xlabel('Year', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
# plt.title('The Change in Birth Rate and Public Finance from 2001 to 2022',
#           fontsize=20)
plt.grid(True)
plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=13)
plt.show()




# Create a plot
plt.figure(figsize=(12, 6))
for column in Draw_year_1.columns:
    plt.plot(Draw_year_1.index, Draw_year_1[column], label=column)

plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.title('The Change in Birth Rate and Public Finance from 2001 to 2022')
plt.grid(True)
plt.legend(loc=(1, 0), fontsize=20)
plt.show()



# Visualize the state/province wise death cases.
plt.figure(figsize=(12, 6))
plt.bar(Draw_county.index, Draw_county['粗出生率(％)'])
plt.xlabel('County or City')
plt.ylabel('Percentage (%)')
plt.title('The Average Annual Birth Rate of Different Region from 2001 to 2022 in Taiwan')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.show()




