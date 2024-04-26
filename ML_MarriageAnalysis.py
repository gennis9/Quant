# Copyright (C) 2023 Gennis
'''
Initiated Date    : 2023/11/09
Last Updated Date : 2023/12/16
Aim: Machine Learning task on marriage rate as response.
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
import re
import seaborn as sns
import scipy
from sklearn import preprocessing as skp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import warnings
%matplotlib inline


## Set the path of data.
_Path = r'C:\Users\Gennis\JupyterNB\Python with ML\MidTermReport\Input Data'
os.chdir(_Path)


## Set the model specification/ parameters.
# Set the response and its corresponding dataset location.
# remove parameter: whether remove the category of the responses located 
# in order to explore the effects of the variables in other categories.
responses = ['粗結婚率(‰)', '粗離婚率(‰)']
y_loc     = [0, 0]
remove    = True

# How many missing values are allowed in the predictors.
# If >=1, the thresholds is number of missing values.
# If <1, the thresholds is the proportion of the missing values.
missNum = 0.01

# Set the start year and end year for the investigation period.
startYear = 2003
endYear   = 2022

# Whether lagging period is adopted for predictors.
lag = 0

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
_FileNames = ['Demographic.csv', 'Labor.csv', 'Crime.csv',
              'Old-Age Welfare.csv', 'Argriculture.csv', 'Civil Planning.csv',
              'Vehicles.csv', 'Industry.csv', 'Public Finance.csv',
              'Healthcare.csv', 'Environment.csv', 'Household Finance.csv']
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
    dta = dta.replace( '…', np.nan )
    dta = dta.replace( '_', np.nan )
    
    ## Remove the thousands separators in the data.
    dta = dta.astype(str)

    for col in dta.columns[2:]:
        dta[col] = dta[col].str.replace(',','',regex=True).astype(float)

    dta['Year'] = dta['Year'].astype(float).astype(int)

    return dta


## Clean the data for missing values.
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

## Extract the responses from the dataset.
# Response: Marriage rate, divorce rate.
Ys = Data[0][['County', 'Year']]
for _var in range(len( responses )):
    Ys = pd.concat([ Ys, Data[ y_loc[_var] ][[responses[_var]]] ], axis=1)
    
    ## Remove the response from the predictor dataset.
    _col = Data[ y_loc[_var] ].columns.tolist()
    _col.remove( responses[_var] )
    Data[ y_loc[_var] ] = Data[ y_loc[_var] ][_col]


## We also remove the variables within same category of the response
# in order to explore the influences from other categories.
if remove:
    for _i in set(y_loc):
        Data = [ Data[col] for col in range(len(Data)) if col != _i ]


## Generate additional variables.
# Economic stability index
# 經濟穩定指數=勞動力參與率×(1−失業率)×勞動力人口數
Data[0]['經濟穩定指數'] = Data[0]['勞動力參與率(％)']/100 * (1- Data[0]['失業率(％)']/100) * Data[0]['勞動力人口數(千人)']*1000

# 儲蓄傾向 Saving Tendency
Data[10]['儲蓄傾向'] = Data[10]['家庭收支-平均每戶儲蓄額(元)'] / Data[10]['家庭收支-平均每戶全年經常性支出(元)']


## Generate new variables: growth rate.
Growth = [ Func_Diff(dta) for dta in Data ]


## Remove some rows of data for balanced panel.
# Match with growth rate varaibles.
for _i in range( len(Data) ):
    Data[_i]   = Data[_i][ Data[_i]['Year'] != startYear ]
Ys = Ys[ Ys['Year'] != startYear ]

# Match response to lagged predictors.
for _yr in range(lag):  
    for _i in range( len(Data) ):
        Data[_i]   = Data[_i][ Data[_i]['Year'] != endYear - _yr ]
        Growth[_i] = Growth[_i][ Growth[_i]['Year'] != endYear - _yr ]
        
    Ys = Ys[ Ys['Year'] != (startYear+1) + _yr ]

## Reset index for data compilation.
for _i in range( len(Data) ):
    Data[_i]   = Data[_i].reset_index(drop=True)
    Growth[_i] = Growth[_i].reset_index(drop=True)
Ys = Ys.reset_index(drop=True)


## Seperate the indexing variables.
FixedEff = Data[0].iloc[:, :2]
Ys = pd.DataFrame( Ys.iloc[:, 2:] )

## To scale up to unify the unit of the response(s) as percentage.
Ys = Ys / 10

## Remove the differencing variables if there are infinity.
for _i in range( len(Growth) ):
    Growth[_i] = Growth[_i].replace([np.inf, -np.inf], np.nan).dropna(axis=1)




#%% Variable Generation

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


## Generate region dummies.
Region = FixedEff.iloc[:, :2]
North = ['臺北市', '新北市', '基隆市', '新竹市', '桃園市', '新竹縣', '宜蘭縣']
Middle = ['臺中市', '苗栗縣', '彰化縣', '南投縣', '雲林縣']
South = ['高雄市', '臺南市', '嘉義市', '嘉義縣', '屏東縣', '澎湖縣']
East = ['花蓮縣', '臺東縣', '金門縣', '連江縣']

Region['North'] = 0
for _var in North:
    Region['North'][ Region['County'] == _var ] = 1
    
Region['Middle'] = 0
for _var in Middle:
    Region['Middle'][ Region['County'] == _var ] = 1

Region['South'] = 0
for _var in South:
    Region['South'][ Region['County'] == _var ] = 1
    
Region['East'] = 0
for _var in East:
    Region['East'][ Region['County'] == _var ] = 1




# %% Variable Selection

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
            y      = Func_Stdn(y)
            
        ## Merge the dataset of the potential predictors.
        X = pd.concat([ X_lv, X_diff ], axis=1)
    
    else:
        if std:
            X = Func_Stdn(X_lv)
            y = Func_Stdn(y)
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
    ## Suppress all warnings to show the newest candidate model collected
    warnings.filterwarnings("ignore")
    
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
                
            ## Print the newest candidate model collected.
            print(str(i) + '. ', selected_vars, 'response:', y.columns.tolist()[0])


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
# Remove the category which has no variable left.
Level  = [ dta.iloc[:, 2:] for dta in Data if len(dta.columns) > 2 ]
Growth = [ dta.iloc[:, 2:] for dta in Growth if len(dta.columns) > 2 ]

## Merge the predictor dataset.
Predictors = pd.DataFrame()
for _i in range(len(Level)):
    Predictors = pd.concat([ Predictors, Level[_i] ], axis=1)
    Predictors = pd.concat([ Predictors, Growth[_i] ], axis=1)


## Run variable selection for each category w.r.t. each response.
# Remove coefficients with low marginal effect.
Results_forward = []
Vars_pass = []
for _col in range(len(Ys.columns)):
    _y = pd.DataFrame(Ys.iloc[:, _col])
    
    results_i = []
    var_i = []
    
    for _i in range(len(Level)):
        ## Save the selection results.
        results_i.append(
            Func_ForwardSelect(_y, Level[_i], Growth[_i], twodata=diff_either, model=model, var_criteria=S1, stop_criteria=S2, model_criteria=S3, signLv=significance, std=std)
            )
        
        ## Collect all variables that pass the first-round variable selection.
        var_i += results_i[_i].params.index.tolist()
        var_i.remove('const')
        
    Results_forward.append( results_i )
    Vars_pass.append( var_i )
    
# [the ith response][the jth category of predictors]
Results_forward[0][2].summary()


## Run variable selection for each category w.r.t. each response.
# Remove coefficients with low marginal effect.
Results_forward_nostd = []
Vars_pass_nostd = []
for _col in range(len(Ys.columns)):
    _y = pd.DataFrame(Ys.iloc[:, _col])
    
    results_i = []
    var_i = []
    
    for _i in range(len(Level)):
        ## Save the selection results.
        results_i.append(
            Func_ForwardSelect(_y, Level[_i], Growth[_i], twodata=diff_either, model=model, var_criteria=S1, stop_criteria=S2, model_criteria=S3, signLv=significance, std=0.01)
            )
        
        ## Collect all variables that pass the first-round variable selection.
        var_i += results_i[_i].params.index.tolist()
        var_i.remove('const')
        
    Results_forward_nostd.append( results_i )
    Vars_pass_nostd.append( var_i )
    
# [the ith response][the jth category of predictors]
Results_forward_nostd[0][2].summary()




#%% 2nd Round Variable Selection: 1st Response - Marriage Rate

Vars_pass[0]

Var_chosen_1 = [#'const',
'就業者之年齡別結構-25-44歲(％)',
# '就業者之年齡別結構-65歲及以上(％)',
# '家庭收支-平均戶內人口數(人)',
# '電力用戶平均每戶售電量(萬度/戶)',
# '歲入來源別結構比-規費及罰款收入(％)',
'老年基本保證年金核付人數(人)',
# '市內電話用戶數(戶)',
# '電燈售電量(百萬度)',
# '都市計畫區現況人口(人)',
# '公司登記新設家數(家)',
# '勞動力人口數(千人)',
# '小客車登記數(輛)',
# '汽車登記數(輛)',
# '中低收入老人生活津貼與老年農民福利津貼核付金額占總決算歲出比率(％)',
# '機動車輛登記數(輛)',
# '機車登記數(輛)',
'現有藥商家數(家)',
'執業醫事人員數(人)',
# '護理人員數(人)',
'老人長期照顧、安養機構數(所)',
# '執業醫師數(人)',
# '老人長期照顧、安養機構可供進住人數(人)',
# '中低收入老人生活津貼、老年基本保證年金與老年農民福利津貼核付金額占總決算歲出比率(％)',
# '小貨車登記數(輛)',
# '執行機關資源回收量(公噸)',
# '老人長期照顧、安養機構實際進住人數(人)',
# '老人長期照顧、安養機構工作人員數(人)',
# '一般廢棄物回收率(%)',
# '△每萬人口執業醫事人員(人/萬人)'
]



Var_Vehicles = ['小客車登記數(輛)', '汽車登記數(輛)', '機動車輛登記數(輛)', 
                '機車登記數(輛)', '小貨車登記數(輛)']


Var_ElderWelfare = ['老年基本保證年金核付人數(人)', 
                    '中低收入老人生活津貼與老年農民福利津貼核付金額占總決算歲出比率(％)',
                    '中低收入老人生活津貼、老年基本保證年金與老年農民福利津貼核付金額占總決算歲出比率(％)',
                    ]

Var_Healthcare = ['現有藥商家數(家)', '執業醫事人員數(人)', '護理人員數(人)',
                  '執業醫師數(人)', '△每萬人口執業醫事人員(人/萬人)']

Var_Eldercare = ['老人長期照顧、安養機構數(所)', '老人長期照顧、安養機構可供進住人數(人)',
                 '老人長期照顧、安養機構實際進住人數(人)',
                 '老人長期照顧、安養機構工作人員數(人)',
                 ]

Var_Recycle = ['執行機關資源回收量(公噸)', '一般廢棄物回收率(%)',]


result_2nd_y1 = Func_ForwardSelect(Ys.iloc[:, [0]], Predictors[Var_Recycle], None, twodata=False, model=model, var_criteria=S1, stop_criteria=S2, model_criteria=S3, signLv=significance, std=0)

result_2nd_y1.summary()




#%% 2nd Round Variable Selection: 2nd Response - Divorce Rate

Vars_pass[1]

Var_chosen_2 = [
# '就業者之年齡別結構-45-64歲(％)',
# '失業率(％)',
'年齡別失業率-45-64歲(％)',
# '就業者之年齡別結構-65歲及以上(％)',
# '就業者之教育程度結構-高中(職)(％)',
'刑案發生率(件/十萬人)',
# '竊盜犯罪人口率(人/十萬人)',
# '△青年嫌疑犯人數(人)',
'老人長期照顧、安養機構每位工作人員服務老人數(人/人)',
# '老人長期照顧、安養機構實際進住人數占老年人口比率(人/萬人)',
# '中低收入老人生活津貼與老年農民福利津貼核付金額占總決算歲出比率(％)',
# '△每千人電話用戶數(戶)',
'每萬人公園、綠地、兒童遊樂場、體育場所及廣場面積數(公頃)',
# '都市計畫區現況人口密度與該縣市人口密度比(倍)',
# '電力用戶平均每戶售電量(萬度/戶)',
# '都市計畫區計畫人口(人)',
# '電燈售電量(百萬度)',
# '都市計畫區面積(平方公里)',
# '都市計畫公共設施用地面積(公頃)',
# '電燈用戶數(戶)',
# '都市計畫公共設施用地計畫面積-道路占都市計畫土地公共設施用地面積之比率(％)',
# '△汽車登記數(輛)',
# '機車登記數(輛)',
'就業者從業身分結構-雇主(％)',
# '△公司登記現有家數(家)',
# '△公司登記現有資本額(百萬元)',
# '歲出政事別結構比-經濟發展支出(％)',
# '歲入來源別結構比-其他收入(％)',
# '歲入來源別結構比-規費及罰款收入(％)',
# '歲入來源別結構比-稅課收入(％)',
# '融資需求-歲入歲出差短(百萬元)',
'平均每一病床服務之人口數(人/床)',
'醫療保健支出占政府支出比率(年度)(％)',
# '平均每一醫療院所服務面積(平方公里/所)',
# '每萬人口執業醫事人員(人/萬人)',
# '醫療院所數(所)',
# '自來水水質檢驗件數(件)',
'一般廢棄物回收率(%)',
# '△執行機關資源回收量(公噸)',
# '家庭收支-平均儲蓄傾向(％)',
# '家庭收支-平均消費傾向(％)',
'家庭收支-平均每戶儲蓄額(元)',
# '家庭現代化設備(每百戶擁有數)-彩色電視機(臺)',
# '平均每人居住面積(坪)',
# '飲食費(不含家外食物)占消費支出比率(%)',
# '平均每人每年可支配所得(元)'
]


Var_Vehicles = ['△汽車登記數(輛)', '機車登記數(輛)',
                ]

Var_Healthcare = ['平均每一病床服務之人口數(人/床)',
                  '醫療保健支出占政府支出比率(年度)(％)',
                  '平均每一醫療院所服務面積(平方公里/所)',
                  '每萬人口執業醫事人員(人/萬人)',
                  '醫療院所數(所)'
                  ]

Var_Eldercare = ['老人長期照顧、安養機構每位工作人員服務老人數(人/人)',
                 '老人長期照顧、安養機構實際進住人數占老年人口比率(人/萬人)',
                 ]

Var_Recycle = ['自來水水質檢驗件數(件)', '一般廢棄物回收率(%)', '△執行機關資源回收量(公噸)',]

Var_Crime = ['刑案發生率(件/十萬人)', '竊盜犯罪人口率(人/十萬人)', '△青年嫌疑犯人數(人)',
             ]

Var_CivilPlan = ['每萬人公園、綠地、兒童遊樂場、體育場所及廣場面積數(公頃)',
                 '都市計畫區面積(平方公里)', '都市計畫公共設施用地面積(公頃)',
                 '都市計畫公共設施用地計畫面積-道路占都市計畫土地公共設施用地面積之比率(％)',]

Var_HouseFin = ['家庭收支-平均消費傾向(％)',
                '家庭收支-平均每戶儲蓄額(元)',
                '家庭現代化設備(每百戶擁有數)-彩色電視機(臺)',
                '平均每人居住面積(坪)',
                '飲食費(不含家外食物)占消費支出比率(%)',
                '平均每人每年可支配所得(元)'
                ]

result_2nd_y2 = Func_ForwardSelect(Ys.iloc[:, [1]], Predictors[Var_Recycle], None, twodata=False, model=model, var_criteria=S1, stop_criteria=S2, model_criteria=S3, signLv=significance, std=0)

result_2nd_y2.summary()


#%% Model Running

## Panel regression on chosen variables.
X1 = pd.concat([ Predictors[Var_chosen_1], Dummy ], axis=1)
Func_Reg(Ys.iloc[:, 0], X1).summary()



X2 = pd.concat([ Predictors[Var_chosen_2], Dummy ], axis=1)
Func_Reg(Ys.iloc[:, 1], X2).summary()




#%% LASSO

import sklearn as sk
import sklearn.linear_model as skl
import sklearn.model_selection as skm
import sklearn.preprocessing as skp
import sklearn.pipeline
import sklearn.svm as sksvm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, auc


## Merge the level predictor dataset only (based on the result of forward selection).
Predictors_lv = pd.DataFrame()
for _i in range(len(Level)):
    Predictors_lv = pd.concat([ Predictors_lv, Level[_i] ], axis=1)


## Correctly show the negative sign in the graphs.
plt.rcParams['axes.unicode_minus'] = False

# Assuming 'X' is your feature variables and 'y' is your target variable
X_train, X_test, y_train, y_test = skm.train_test_split(Predictors_lv, Ys, test_size=0.2, random_state=2023)
# X_train, X_test, y_train, y_test = skm.train_test_split(Predictors[Var_chosen_1], Ys, test_size=0.2, random_state=2023)


### Marry Rate

# 10-fold cross-validation.
K = 10
kfold = skm.KFold(K,
                  random_state=2023,
                  shuffle=True)


# Method 2: Scan λ from 5e4 to 5e-4 with evenly spaced 100 samples.
lambdas = np.logspace(-4, 4, 100)
lassoCV = skl.ElasticNetCV(l1_ratio=1,
                           alphas = lambdas,
                           cv=kfold)

scaler = skp.StandardScaler(with_mean=True,  with_std=True)
pipeCV = sk.pipeline.Pipeline(steps=[('scaler', scaler),
                                          ('lasso', lassoCV)])
pipeCV.fit(X_train, y_train[ responses[0] ])
tuned_lasso_1 = pipeCV.named_steps['lasso']

# optimal value of λ.
# The amount of penalization chosen by cross validation along alphas_.
tuned_lasso_1.alpha_


## plot of the cross-validation error.
lassoCV_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(tuned_lasso_1.alphas_),
            # mse_path_: Mean square error for the test set on each fold, varying alpha.
            tuned_lasso_1.mse_path_.mean(axis=1),
            yerr=tuned_lasso_1.mse_path_.std(axis=1) / np.sqrt(K))
ax.axvline(-np.log(tuned_lasso_1.alpha_), c='k', ls='--')
# ax.set_ylim([50000,250000])
ax.set_xlabel('-log(λ)', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)



## Plot the paths to get a sense of how the coefficients vary with λ.
# We standardize features before lasso regression
X_stdn = X_train - X_train.mean()
X_scale = X_train.std()
X_stdn = X_stdn / X_scale


## Collect the coefficients of Lasso reg under each λ.
lasso = skl.Lasso()
coefs = []

for a in lambdas:
    lasso.set_params(alpha=a)
    lasso.fit(X_stdn, y_train[ responses[0] ]) 
    coefs.append(lasso.coef_)

# Associated with each value of λ is a vector of ridge regression coefficients, 
# that can be accessed by a column of soln_array. 
# In this case, soln_array is a matrix, with 10 columns (one for each predictor) 
# and 100 rows (one for each value of λ).
soln_array = np.array(coefs)

# Take transpose to reverse the position of λ and predictors.
soln_path = pd.DataFrame(soln_array,
                         columns=X_train.columns,
                         index=-np.log(lambdas))
soln_path.index.name = 'negative log(lambda)'

# Randomly pick serveral variables as demonstration.
np.random.seed(2023)
idx_show = np.random.choice(np.arange(0, len(X_train.columns)), size=20, replace=False)
soln_path_reduced = soln_path.iloc[:, idx_show]


# Plot the paths to get a sense of how the coefficients vary with λ.
path_fig, ax = plt.subplots(figsize=(12,8))
soln_path_reduced.plot(ax=ax, legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
path_fig.tight_layout()

# Make predictions on the test data
y1_pred = tuned_lasso.predict(X_test)

# Evaluate the model
lasso_cv_score1 = tuned_lasso_1.score(X_test, y_test[responses[0]])
print(f'R-squared score on test data: {lasso_cv_score1}')


## Choose the variables that used in SVM
coeff_lasso_1 = pd.Series(tuned_lasso_1.coef_)
idx_lasso_1   = coeff_lasso_1[ np.abs(coeff_lasso_1) > 10**-5 ].index
X_new_1 = Predictors_lv.iloc[ :, idx_lasso_1 ]


### Divorce Rate

# 10-fold cross-validation.
K = 10
kfold = skm.KFold(K,
                  random_state=2023,
                  shuffle=True)


# Method 2: Scan λ from 5e4 to 5e-4 with evenly spaced 100 samples.
lambdas = np.logspace(-4, 4, 100)
lassoCV = skl.ElasticNetCV(l1_ratio=1,
                           alphas = lambdas,
                           cv=kfold)

scaler = skp.StandardScaler(with_mean=True,  with_std=True)
pipeCV = sk.pipeline.Pipeline(steps=[('scaler', scaler),
                                          ('lasso', lassoCV)])
pipeCV.fit(X_train, y_train[ responses[0] ])
tuned_lasso_2 = pipeCV.named_steps['lasso']

# optimal value of λ.
# The amount of penalization chosen by cross validation along alphas_.
tuned_lasso_2.alpha_


## plot of the cross-validation error.
lassoCV_fig, ax = plt.subplots(figsize=(8,8))
ax.errorbar(-np.log(tuned_lasso_2.alphas_),
            # mse_path_: Mean square error for the test set on each fold, varying alpha.
            tuned_lasso_2.mse_path_.mean(axis=1),
            yerr=tuned_lasso_2.mse_path_.std(axis=1) / np.sqrt(K))
ax.axvline(-np.log(tuned_lasso_2.alpha_), c='k', ls='--')
# ax.set_ylim([50000,250000])
ax.set_xlabel('-log(λ)', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20)


## Choose the variables that used in SVM
coeff_lasso_2 = pd.Series(tuned_lasso_2.coef_)
idx_lasso_2   = coeff_lasso_2[ np.abs(coeff_lasso_2) > 10**-5 ].index
X_new_2 = Predictors_lv.iloc[ :, idx_lasso_2 ]




#%% SVM


## Demean with time-specific factors.
DemeanX = pd.concat([FixedEff.iloc[:, 0], X_new_1], axis=1)
DemeanY = pd.concat([FixedEff.iloc[:, 0], Ys[responses[0]]], axis=1)

## Calculate time-wise means.
Time_meansY = DemeanY.groupby('County')[responses[0]].transform('median')
Time_meansX = DemeanX.groupby('County')[DemeanX.columns[1:]].transform('median')

## Subtract time-wise means from the variables.
DemeanY = DemeanY.iloc[:, 1] - Time_meansY
DemeanX = DemeanX.iloc[:, 1:] - Time_meansX


## Assign -1 for the lower-than-mean values and 1 for the higher-than-mean values.
# DemeanY = DemeanY.applymap(lambda x: -1 if x < 0 else (1 if x > 0 else 0))

## Assign 1 for the higher-than-mean values and 0 otherwise.
DemeanY = (DemeanY > 0).astype(int)

# DemeanY[responses[0]].value_counts()
# DemeanY[responses[1]].value_counts()


## Standardize data.
scaler = skp.StandardScaler()
X_std = scaler.fit_transform(DemeanX)

## Train SVM.
X_train, X_test, y_train, y_test = skm.train_test_split(X_std, DemeanY, test_size=0.2, random_state=2023)

# svm_model1 = sksvm.SVC(kernel='linear', C=1, probability=True)
# svm_model1.fit(X_train, y_train)

## Grid search for best alpha for SVM training.
# Specify the hyperparameter grid for 'C'
param_grid = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}

# Initialize the SVM model
svm_model = sksvm.SVC(kernel='linear', probability=True)  # You can adjust the kernel based on your problem

# Initialize GridSearchCV for hyperparameter tuning
grid_search = skm.GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')

# Fit the model using cross-validation
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_svm_model1 = grid_search.best_estimator_


## Make predictions on the test data
y_pred1 = best_svm_model1.predict(X_test)

## Evaluate the performance using common classification metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred1)}')
print(f'Precision: {precision_score(y_test, y_pred1, average="weighted")}')
print(f'Recall: {recall_score(y_test, y_pred1, average="weighted")}')
print(f'F1 Score: {f1_score(y_test, y_pred1, average="weighted")}')


# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred1)

# Create a DataFrame for the confusion matrix with row and column names
conf_matrix_df = pd.DataFrame(conf_matrix, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])


print("Confusion Matrix:")
print(conf_matrix_df)


# Compute the ROC curve and AUC
y_prob_1 = svm_model1.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_1)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()





# %% Exploratory Data Analysis

## Correctly show the Mandarin in the graphs.
matplotlib.rc('font', family='Microsoft JhengHei')

''' 
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
Data[6].columns.tolist()
Data[7].columns.tolist()
Data[8].columns.tolist()
Data[9].columns.tolist()
Data[10].columns.tolist()


# var_Demo = ['15-64歲人口數(人)', '青壯年人口比率(15-64歲)(％)', '性比例(女=100)',
#             '遷入人口數(人)', '遷出人口數(人)', '人口密度(人/平方公里)',
#             '粗結婚率(‰)', '粗離婚率(‰)']
var_LabI = ['年齡別失業率-15-24歲(％)', '年齡別失業率-25-44歲(％)', 
            '年齡別失業率-45-64歲(％)', '就業者之行業結構-工業(％)',
            '就業者之行業結構-服務業(％)', '就業者之行業結構-農林漁牧業(％)',
            '總工時(小時)']
var_EldC = ['老人長期照顧、安養機構數(所)', '老人長期照顧、安養機構每位工作人員服務老人數(人/人)',
            '中低收入老人生活津貼核定人數(人)', '中低收入老人生活津貼核發金額(元)',
            '老年基本保證年金核付人數(人)', '老年基本保證年金核付金額(元)',] 
var_Home = ['家庭收支-平均每戶全年經常性收入(元)', '家庭收支-平均每戶可支配所得(元)',
            '家庭收支-平均每戶儲蓄額(元)', '家庭收支-平均儲蓄傾向(％)',
            '每戶可支配所得中位數(元)', '平均每人每年可支配所得(元)',] 
var_Cars = ['小客車登記數(輛)',
            '汽車登記數(輛)',
            '機動車輛登記數(輛)',
            '機車登記數(輛)',
            '小貨車登記數(輛)',]
var_Heal = [ '醫療院所數(所)',
             '現有藥商家數(家)',
             '執業醫事人員數(人)',
             ]


## Construct the dataset with chosen variables.
EDA_LabI = pd.concat([Ys, Data[0][var_LabI]], axis=1)

EDA_Envi = pd.concat([Ys, Data[3][var_Envi]], axis=1)
EDA_Publ = pd.concat([Ys, Data[4][var_Publ]], axis=1)
EDA_Home = pd.concat([Ys, Data[5][var_Home]], axis=1)
EDA_Cars = pd.concat([Ys, Data[5][var_Cars]], axis=1)
EDA_Heal = pd.concat([Ys, Data[8][var_Heal]], axis=1)
EDA_Dist = pd.concat([Ys, Region.iloc[:, 2:]], axis=1)

var_GHeal = [ '△' + name for name in var_Heal ]
EDA_GHeal = pd.concat([Ys, Growth[7][var_GHeal]], axis=1)


## Sketch the pairplots.
if graph:
    sns.pairplot(EDA_Demo)
    sns.pairplot(EDA_LabI)
    sns.pairplot(EDA_Envi)
    sns.pairplot(EDA_Publ)
    sns.pairplot(EDA_Home)
    sns.pairplot(EDA_Cars)
    sns.pairplot(EDA_Heal)
    sns.pairplot(EDA_Dist)
    
    sns.pairplot(EDA_GHeal)


## Sketch the heatmap.
if graph:
    sns.heatmap(EDA_Demo.corr(), annot=True)
    sns.heatmap(EDA_LabI.corr(), annot=True)
    sns.heatmap(EDA_Heal.corr(), annot=True)
    sns.heatmap(EDA_Envi.corr(), annot=True)
    sns.heatmap(EDA_Publ.corr(), annot=True)
    sns.heatmap(EDA_Home.corr(), annot=True)
    sns.heatmap(EDA_Dist.corr(), annot=True)
    
## Show summary statistics.
SS_Demo = EDA_Demo.describe()
SS_LabI = Data[1].iloc[:, 2:].describe()
SS_Heal = Data[2][var_Heal].describe()
SS_Envi = Data[3][var_Envi].describe()
SS_Publ = Data[4][var_Publ].describe()
SS_Home = Data[5][var_Home].describe()




# Box plot.
## Draw the boxplot.
District = Region.iloc[:, :2]
District['District'] = ''
for _var in North:
    District.loc[:, 'District'][ District['County'] == _var ] = 'North'
for _var in Middle:
    District.loc[:, 'District'][ District['County'] == _var ] = 'Middle'
for _var in South:
    District.loc[:, 'District'][ District['County'] == _var ] = 'South'
for _var in East:
    District.loc[:, 'District'][ District['County'] == _var ] = 'East'

District = pd.concat([Ys, District], axis=1)

plt.figure(figsize = (12, 7))
ax = sns.boxplot(x = 'District', y = '粗離婚率(‰)', width = .85,
                 data = District)
sns.set_style('darkgrid')
plt.show()



# Create a plot

Plot = pd.concat([Ys, Region.iloc[:, 1]], axis=1)
Plot = Plot.groupby(['Year']).agg({'粗結婚率(‰)'  : 'sum',
                                   '粗離婚率(‰)'  : 'sum'})


plt.figure(figsize=(12, 6))

plt.plot(Plot.index, Plot['粗離婚率(‰)']/10, label='粗離婚率(‰)')
plt.plot(Plot.index, Plot['粗結婚率(‰)']/10, label='粗結婚率(‰)')

plt.xlabel('Year')
plt.ylabel('Percentage (%)')
plt.title('The Changes in Marriage Rate and Divorce Rate from 2003 to 2022')
plt.grid(True)
plt.xlim(2004, 2022)
plt.legend(loc=(1, 0), fontsize=20)
plt.show()



'''











