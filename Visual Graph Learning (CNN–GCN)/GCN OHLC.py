# Copyright (C) 2025 Gennis
'''
Initiated Date    : 2025/11/01
Last Updated Date : 2025/11/07
Aim: Visual Graph Learning with GCN for Financial Forecasting: CNN-Based OHLC Image Embeddings and Market Topology
Input Data: CSV, downloaded from Yahoo! Finance
'''
# %% Environment

from copy import deepcopy
# import curl_cffi
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
import numpy as np
import os
import pandas as pd
import requests
import scipy
# import sklearn as sk
# import sklearn.model_selection as skm
# import yfinance as yf
# %matplotlib inline

_Path = r'D:\03Programs_Clouds\Google Drive\NSYSU\05Deep Learning\Final Report'
os.chdir(_Path)




# %% Parameter Setting

# Length of total period
start_date  = '2000-01-01'
end_date    = '2025-11-05'
# Preserve 23-year length of data for CNN task.
cutoff_date = ['2002-11-01', '2025-10-31']

# Risk free rate
rf = 0.01

## Divide dataset.
# Resemble JIANG (2023) "Re‐Imagining Price Trends" for 8 years in training model.
train_size = 8
val_size   = 0.3

# OHLC graph's.
day_obs  = 20
height   = 64
# Prediction
day_inv  = 5


## Set the random seed.
_seed = 2025

## Set for controlling download the graph sketched.
plot = False

## Set for controlling download the data or read the data locally.
download = False
# Stock Price
_subfolder_s  = 'Data_Stocks'
_file_stock = 'Stock_Prices.csv'
_file_wiki  = 'Hang_Seng_Index.csv'
_file_HSI   = 'HSI_Price.csv'
# OHLC Graph and Target
_subfolder_o  = 'Data_OHLC'
# Network Graph
_subfolder_n  = 'Data_Network'




# %% Data Access - Stocks

## Access/Save the data in the subfolder.
_subfolder_s  = os.path.join(_Path, _subfolder_s)
_path_stock = os.path.join(_subfolder_s, _file_stock)
_path_wiki  = os.path.join(_subfolder_s, _file_wiki)
_path_HSI   = os.path.join(_subfolder_s, _file_HSI)


if download:
    ## Download the information regarding the Hang Seng Index from Wiki.
    wikiURL = 'https://en.wikipedia.org/wiki/Hang_Seng_Index'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    try:
        page = requests.get(wikiURL, headers=headers, timeout=10)
        
        ## Raise an exception if the request was unsuccessful (e.g., 404, 403, 500)
        page.raise_for_status()

        WikiTables = pd.read_html(page.text)

        ## Extract the list of the stock components included in the table from the information crawled.
        WikiTable = WikiTables[6]

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error occurred: {err}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


    ## Prepare for downloading stock price data.
    WikiTable['Ticker'] = WikiTable['Ticker'].apply( lambda x: int(x[6:]) )

    def Func_Ticker( num ):
        if num < 10:
            tick = '000' + str(num)
        elif num < 100:
            tick = '00'  + str(num)
        elif num < 1000:
            tick = '0'   + str(num)
        else:
            tick = str(num)
        
        return tick + '.hk'


    WikiTable['Ticker_yf']   = WikiTable['Ticker'].apply( lambda x: Func_Ticker(x) )
    WikiTable['Ticker_node'] = WikiTable['Ticker_yf'].apply( lambda x: x[:-3] )
    
    ## Save the Hang Seng Index information.
    WikiTable.to_csv(_path_wiki, index=False)

    
    ## Download stock prices.
    session = curl_cffi.requests.Session(impersonate="chrome")
    Stock = yf.download(tickers=WikiTable['Ticker_yf'].tolist(), 
                        start=start_date, end=end_date, progress=False, 
                        ignore_tz = False, threads = True, session=session)

    ## Save the stock prices.
    Stock.to_csv(_path_stock)
    
    
    ## Retrieve the Index.
    HSI = yf.download('^HSI', start=start_date, end=end_date)
    HSI.to_csv(_path_HSI)
    

else:
    ## Only read the stock data locally.
    WikiTable = pd.read_csv(_path_wiki)
    Stock     = pd.read_csv(_path_stock, header=[0, 1], index_col=0, parse_dates=True)
    HSI       = pd.read_csv(_path_HSI, header=[0, 1], index_col=0, parse_dates=True)




# %% x-date Return and MA20 Generation

Stock.columns.names = ['Price', 'Ticker']

## Compute 20-day moving average for each stock.
_temp_ma = Stock.xs('Close', level=0, axis=1).rolling(20).mean()
_temp_ma.columns = pd.MultiIndex.from_product([['MA20'], _temp_ma.columns])
Stock = pd.concat([Stock, _temp_ma], axis=1)


## Compute x-day return for each stock.
closePs   = Stock.xs('Close', level=0, axis=1)
openPs_5d = Stock.xs('Open', level=0, axis=1).shift(day_inv - 1)
_temp_r = (closePs - openPs_5d) / openPs_5d
_temp_r.columns = pd.MultiIndex.from_product([['r_Holding'], _temp_r.columns])
Stock = pd.concat([Stock, _temp_r], axis=1)




# %% Network Example Sketching

## Draw the MST network for all stocks available over whole period.
Returns = Stock.xs('Close', level='Price', axis=1).dropna()

## Construct the adjacency matrix and thus centrality by MST.
# If volatilty measure is adopted, rolling windows must be the input.
def Func_AdjM(dta, var='return', corr='Spear'):
    if corr == 'Spear':
        ## Construct the Spearman correlation matrix instead of Pearson.
        if var == 'return':
            Spear, _ = scipy.stats.spearmanr(dta)
        
        elif var == 'volatility':
            # Volatility must be collected period by period instead of daily data.
            rolling = pd.DataFrame()
            ## Collect the features from each rolling window.
            for period in range(len(dta)):
                rolling = pd.concat([rolling, dta[period].var()], axis=1)
            # rolling.columns = Last_dates[M-1 :]
            rolling = rolling.T
            
            Spear, _ = scipy.stats.spearmanr(rolling)
            
        ## Construct the distance matrix.
        Distance_mat = np.sqrt( 2 * (1 - Spear) )
            
    ## When Pearson correlation coefficient is adopted.
    elif corr == 'Pearson':
        if var == 'return':
            Distance_mat = np.sqrt( 2 * (1 - dta.corr()) )


    ## Construct the MST.
    MST = scipy.sparse.csgraph.minimum_spanning_tree(Distance_mat).toarray()

    ## Construct the adjacency matrix by MST.
    Adj_mat = (MST != 0) * 1
    Adj_mat = Adj_mat + Adj_mat.T


    ## Construct network.
    rows, cols = np.where(Adj_mat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    ## Compute centrality.
    EC = nx.eigenvector_centrality(gr, max_iter=2000)
    Centrality = [ value for value in EC.values() ] 

    return MST, Adj_mat, Centrality, gr


## The index of the inputs must be the same.
# Compute the MST through rank correlation among returns.
MST_r, Adj_mat_r, Centrality_r, gr_r = Func_AdjM(Returns, var='return')

## Sketch the network. Follow the color style of OHLC graphs.
G = nx.from_numpy_array(Adj_mat_r, create_using=nx.DiGraph)
layout = nx.spring_layout(G, seed=32)
fig, ax = plt.subplots()
# Set the background color of the plot area to black.
ax.set_facecolor('black')
fig.set_facecolor('black')
# Set the nodes and edges are white.
nx.draw(
    G,
    pos=layout,
    with_labels=False,
    arrows=False,
    node_size=100,
    node_color='white',
    edge_color='white',
    edgecolors='black'
    )
if plot:
    # Use a black background for the saved figure as well
    plt.savefig('MST_Example.jpeg', facecolor='black')
plt.show()




# %% Data Cleaning

## Find the first valid date for each stock
first_dates = Stock.groupby(level='Ticker', axis=1).apply(
    lambda df_group: df_group.apply(pd.Series.first_valid_index).min()
    )
first_dates = first_dates.sort_values().reset_index()
first_dates.columns = ['Ticker', 'First_Date']


## Remove all stocks whose first valid date is later than the cut-off date.
# Remove the trading dates earlier than the cut-off date.
def Func_Period(cutoff, data):
    cutoff = pd.to_datetime(cutoff)
    # Instablility of to_datetime()
    if cutoff.tz is None:
        cutoff = cutoff.tz_localize(data.index.tz)
    else:
        cutoff = cutoff.tz_convert(data.index.tz)
    
    return cutoff

cutoff_date = [Func_Period(date, Stock) for date in cutoff_date]
Stock = Stock[Stock.index >= cutoff_date[0]]
Stock = Stock[Stock.index <= cutoff_date[1]]

# Remove the price data accordingly.
_temp_nan = Stock.groupby(level='Ticker', axis=1).apply(
    lambda df_group: df_group.isnull().any().any()
    )
StockList = _temp_nan[~_temp_nan].index.tolist()

_temp_col = Stock.columns.get_level_values('Ticker').isin(StockList)
Stock = Stock.loc[:, _temp_col]




# %% Splitting Training and Test Set

## Extract first 8 years to be training set.
_temp_train = cutoff_date[0] + pd.DateOffset(years = train_size)
Train = Stock[Stock.index < _temp_train]
Test  = Stock[Stock.index >= _temp_train]




# %% OHLC Graph Generation

def Func_PixelMatrix(data, height):
    ## 3 pixel width per trading day.
    n_pred = len(data)
    width  = 3 * n_pred

    # Spacing with 1 pixel to seperate volumn bar and price bar.
    height_vol = round(height * 0.2) + 1
    

    ## Create a black pixel matrix initially.
    # uint8 canvas: 0 = black, 255 = white
    Canvas = np.zeros((height, width), dtype=np.uint8)
    WHITE  = 1
    
    # price → y-pixel mapping (0 = top row)
    pmin = min(data['Low'].min(),  data['MA20'].min())
    pmax = max(data['High'].max(), data['MA20'].max())
    vol_max = data['Volume'].max()
    
    ## Prevent no trading scenario.
    if pmax == pmin:
        # If all prices are the same, just add a tiny amount to pmax to prevent a ZeroDivisionError.
        pmax += 1e-6
    
    ## Compute the y-coordinate of each price.
    def Func_y_coord(price):
        return round( (pmax - price) / (pmax - pmin) * (height - height_vol - 1) )
    
    
    ## Draw OHLC with Volume.
    for day, row in enumerate(data.itertuples()):
        ## Draw prices.
        x_mid   = day * 3 + 1                  # middle column this day
    
        ## Compute y-coordinates of OHLC prices.
        y_High, y_Low   = Func_y_coord(row.High), Func_y_coord(row.Low)
        y_Open, y_Close = Func_y_coord(row.Open), Func_y_coord(row.Close)
    
        # Vertical high-low bar
        Canvas[y_High : y_Low+1, x_mid] = WHITE
        # Open tick (left)
        Canvas[y_Open, x_mid-1:x_mid]    = WHITE
        # Close tick (right)
        Canvas[y_Close, x_mid+1]        = WHITE
        
        ## Draw volume bars.
        # Prevent no trading scenario.
        if vol_max > 0:
            bar_height = int(round( row.Volume / vol_max * (height_vol-1) ))
            y_vol_top  = height - bar_height          # start row of bar
            Canvas[y_vol_top:, x_mid:x_mid+1] = WHITE
        
        
    ## Draw MA20 curve.
    y_MA20 = [Func_y_coord(price) for price in data['MA20']]
    x_MA20 = [day * 3 + 1 for day in range(n_pred)]  # use center col
    
    ## Bresenham-style line drawing between consecutive MA points.
    for day in range(n_pred - 1):
        x0, y0 = x_MA20[day],     y_MA20[day]
        x1, y1 = x_MA20[day+1],   y_MA20[day+1]
        
        ## Connect last MA20 value and current MA20 value.
        xs    = np.linspace(x0, x1, 4).astype(int)
        ys    = np.linspace(y0, y1, 4).astype(int)
        Canvas[ys, xs] = WHITE
        
    return Canvas




# %% CNN: Generation of Training Set and Validation Set

## Define the output path for figures, if plot.
_Path_figure = os.path.join(_Path, 'Data_OHLC')


## Collect training set and validation set samples.
X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []


## Stratified split the training and validation per stock.
for ticker in StockList:
    print(f"\n--- Processing stock: {ticker} ---")
    
    ## Extract this stock's data.
    OHLC = deepcopy(Train.xs(ticker, level='Ticker', axis=1))

    ## Collect OHLC graphs are x-day returns.
    current_stock_X = []
    current_stock_y_metadata = []

    ## Calculate total rolling windows.
    num_windows = (len(OHLC) - day_obs - day_inv)
    
    ## For each rolling window.
    for i_window in range(num_windows + 1):
        row_start = i_window
        row_end   = row_start + day_obs
        df = OHLC.iloc[row_start : row_end].copy()
        
        ## Generate the pixel matrix.
        matrix = Func_PixelMatrix(df, height)
        
        ## Extract corresponding x-day return for current pixel matrix.
        ret_5d = OHLC.iloc[row_end + day_inv - 1]['r_Holding']
        
        ## Store metadata.
        metadata = {'TradeDate': OHLC.index[row_end],
                    'Ticker': ticker,
                    'Return_5D': ret_5d
                    }
        
        ## Add data to this stock's temporary lists.
        current_stock_X.append(matrix)
        current_stock_y_metadata.append(metadata)

        ## Store the OHLC graphs, if plot.
        if plot:
            ticker_safe_name = ticker.replace('.', '_')
            file_name = (f'OHLC_{ticker_safe_name}_' + 
                         str(day_obs) + '_' +
                         df.index[0].strftime('%Y%m%d') + 'to' + 
                         df.index[-1].strftime('%Y%m%d') + '.png')
            
            save_path = os.path.join(_Path_figure, file_name)
            
            plt.imshow(matrix, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            
        if i_window % 500 == 0:
            print(f"     {day_obs}-Day: {i_window} / {num_windows}")


    ## Convert the temporary lists into file output format.
    X_ticker = np.array(current_stock_X)
    y_ticker = pd.DataFrame(current_stock_y_metadata)


    ## Split training and validation samples for current stock.
    X_train_ticker, X_val_ticker, y_train_ticker, y_val_ticker = skm.train_test_split(
        X_ticker, 
        y_ticker, 
        test_size = val_size, 
        random_state=_seed
        )


    ## Store the stock samples accordingly.
    X_train_list.append(X_train_ticker)
    X_val_list.append(X_val_ticker)
    y_train_list.append(y_train_ticker)
    y_val_list.append(y_val_ticker)
    
    print(f" Split {ticker}: {len(X_train_ticker)} train, {len(X_val_ticker)} validation.")


## In order to reduce the number of files generated, expand data dimension.
X_train = np.concatenate(X_train_list, axis=0)
X_val = np.concatenate(X_val_list, axis=0)

y_train = pd.concat(y_train_list, axis=0).reset_index(drop=True)
y_val = pd.concat(y_val_list, axis=0).reset_index(drop=True)


## Shuffle Training Set and Validation Set accordingly.
X_train, y_train = sk.utils.shuffle(X_train, y_train, random_state=_seed)
X_val, y_val = sk.utils.shuffle(X_val, y_val, random_state=_seed)




# %% Export - CNN: Training and Validation Sets

_subfolder_o  = os.path.join(_Path, _subfolder_o)

## Save the training set.
np.save(os.path.join(_subfolder_o, 'X_train_CNN.npy'), X_train)
y_train.to_csv(os.path.join(_subfolder_o, 'y_train_CNN.csv'), index=False)

## Save the validation set.
np.save(os.path.join(_subfolder_o, 'X_val_CNN.npy'), X_val)
y_val.to_csv(os.path.join(_subfolder_o, 'y_val_CNN.csv'), index=False)


## Save the trading dates.
Train.to_csv(os.path.join(_subfolder_o, 'Stocks_TrainAndVal.csv'), index=True)




# %% Generation of 5 Non-Overlapping Test Sets

## Create a test set for each starting trading date.
n_sets = day_inv

## Initialize a 4-D data structure: time x stock x OHLC(H x W) for each of test set.
X_lists_stock = [[] for _ in range(n_sets)]
y_lists_stock = [[] for _ in range(n_sets)]


## Loop over every stock in the test set.
for ticker in StockList:
    print(f"\n--- Processing stock: {ticker} ---")
    
    ## Extract current stock's data.
    OHLC = deepcopy(Test.xs(ticker, level='Ticker', axis=1))

    # --- NEW: Create 5 *temporary* lists, one for each set, for THIS stock ---
    ticker_X_sets = [[] for _ in range(n_sets)]
    ticker_y_sets = [[] for _ in range(n_sets)]

    ## Calculate total rolling windows.
    # +1 comes from we invest at 1st day's open and end at 5th day's close,
    # so there is only 4 day difference.
    num_windows = (len(OHLC) - day_obs - day_inv + 1)
    
    ## For each rolling window.
    for i_window in range(num_windows):
        
        ## Determine which test set this window belongs to (0, 1, 2, 3, or 4).
        set_index = i_window % n_sets
        
        row_start = i_window
        row_end   = row_start + day_obs
        df = OHLC.iloc[row_start : row_end].copy()
        
        ## Generate the pixel matrix.
        matrix = Func_PixelMatrix(df, height)
        
        ## Extract corresponding x-day return.
        # try:
        ret_5d = OHLC.iloc[row_end + day_inv - 1]['r_Holding']
        # except IndexError:
        #     continue

        ## Validate data
        # if pd.isna(ret_5d) or np.all(matrix == 0):
        #     continue
            
        ## Store metadata.
        metadata = {'TradeDate': OHLC.index[row_end],
                    'Ticker': ticker,
                    'Return_5D': ret_5d
                   }
        
        # Add data to the correct set's temporary list
        ticker_X_sets[set_index].append(matrix)
        ticker_y_sets[set_index].append(metadata)
        
        if i_window % 500 == 0:
            print(f"{day_obs}-Day: {i_window} / {num_windows}")
            

    ## After looping through windows, convert 5 temp lists to 3D arrays and append them to the 5 master lists.
    for i in range(n_sets):
        ## Convert this set's list to a 3D array.
        # Shape: (T_test, H, W)
        X_ticker_set_array = np.array(ticker_X_sets[i])
        
        ## Convert this set's metadata to a DataFrame.
        y_ticker_set_df = pd.DataFrame(ticker_y_sets[i])
        
        ## Append this stock's 3D array chunk to the correct master list.
        # Resulted Shape: (day_inv, T_test, S, H, W)
        X_lists_stock[i].append(X_ticker_set_array)
        y_lists_stock[i].append(y_ticker_set_df)


## These lists will hold your final different test sets
X_test_sets = []
y_test_sets = []

for i in range(n_sets):
    ## Take the list of 3D arrays (one per stock) and stack along axis=1
    # Final Shape: (Num_Dates, Num_Stocks, Image_Height, Image_Width)
    X_4D_set = np.stack(X_lists_stock[i], axis=1)
    
    
    ## Concatenate the list of all DataFrames for this set (one DataFrame per stock) into one big DataFrame.
    y_df_set = pd.concat(y_lists_stock[i], axis=0)
    
    ## Sort the transactions by TradeDate, then by Ticker.
    y_df_set.sort_values(by=['TradeDate', 'Ticker'], inplace=True)
    y_df_set.reset_index(drop=True, inplace=True)

    ## Collect as different test sets corresponded to different starting trading dates.
    X_test_sets.append(X_4D_set)
    y_test_sets.append(y_df_set)
    
    print(f"Test Set {i}: X shape={X_4D_set.shape}, y shape={y_df_set.shape}")




# %% Balanced Panel for Test Sets

## Check for unbalanced panel among all test sets.
for i in range(n_sets):
    print(f"Test Set {i}: X shape={X_test_sets[i].shape}, y shape={y_test_sets[i].shape}")


## Get balanced panel in order to average out the investment performance among the test sets.
X_test_sets[0] = X_test_sets[0][:-1]

_row_removed = X_test_sets[0].shape[1] 
y_test_sets[0] = y_test_sets[0].iloc[:-_row_removed]




# %% Export - Test Sets

## Save all features of all test sets into a single tensor file.
X_test_5D = np.stack(X_test_sets, axis=0)


_subfolder_o = os.path.join(_Path, _subfolder_o)
## Save the training set.
np.save(os.path.join(_subfolder_o, 'X_test.npy'), X_test_5D)

for i in range(n_sets):
    _FileName = 'y_test_' + str(i) + '.csv'
    y_test_sets[i].to_csv(os.path.join(_subfolder_o, _FileName), index=False)




# %% GCN: Generation of Training Set and Validation Set

# --- 1. Collect data per stock (chronologically) ---

# These lists will hold the 3D array (Dates, H, W) for each stock
X_ticker_list = []
# These lists will hold the DataFrame (Dates, 3 cols) for each stock
y_ticker_list = []


## Stratified split the training and validation per stock.
for ticker in StockList:
    print(f"\n--- Processing stock: {ticker} ---")
    
    ## Extract this stock's data.
    OHLC = deepcopy(Train.xs(ticker, level='Ticker', axis=1))

    ## Collect OHLC graphs are x-day returns.
    current_stock_X = []
    current_stock_y_metadata = []
    
    num_windows = (len(OHLC) - day_obs - day_inv + 1)
    
    ## For each rolling window.
    for i_window in range(num_windows): 
        row_start = i_window
        row_end   = row_start + day_obs
        df = OHLC.iloc[row_start : row_end].copy()
        
        ## Generate the pixel matrix.
        matrix = Func_PixelMatrix(df, height)
        
        ## Extract corresponding x-day return for current pixel matrix.
        ret_5d = OHLC.iloc[row_end + day_inv - 1]['r_Holding']
            
        ## Store metadata.
        metadata = {'TradeDate': OHLC.index[row_end],
                    'Ticker': ticker,
                    'Return_5D': ret_5d
                   }
        
        ## Add data to this stock's temporary lists.
        current_stock_X.append(matrix)
        current_stock_y_metadata.append(metadata)
            
        if i_window % 500 == 0:
            print(f"     {day_obs}-Day: {i_window} / {num_windows}")


    ## Convert the temporary lists into file output format.
    X_ticker = np.array(current_stock_X)
    y_ticker = pd.DataFrame(current_stock_y_metadata)

    # Shape: (S, T, H, W).
    X_ticker_list.append(X_ticker)
    y_ticker_list.append(y_ticker)
    


## Stack X along axis=1 (the "Stocks" dimension)
# Shape: (Num_Trade_Dates, Num_Stocks, Image_Height, Image_Width)
X_all_data_4D = np.stack(X_ticker_list, axis=1)

# Stack Y (by first converting DFs to numpy arrays)
y_all_data_3D = np.stack([df.values for df in y_ticker_list], axis=1)


## Shuffle and split the training and validation sets along time axis.
X_train, X_val, y_train, y_val = skm.train_test_split(
    X_all_data_4D,
    y_all_data_3D,
    test_size=val_size,
    random_state=_seed,
    shuffle=True
    )




# %% Export - GCN: Training and Validation Sets

## Stack the target data to output as a csv file.
y_train_flat_array = y_train.reshape(-1, y_train.shape[2])

_col = y_ticker_list[0].columns.tolist()
y_train_df = pd.DataFrame(y_train_flat_array, columns=_col)

y_val_flat_array = y_val.reshape(-1, y_val.shape[2])
y_val_df = pd.DataFrame(y_val_flat_array, columns=_col)


## Save to CSV.
_subfolder_o  = os.path.join(_Path, _subfolder_o)
y_train_df.to_csv(os.path.join(_subfolder_o, 'y_train_GCN.csv'), index=False)
y_val_df.to_csv(os.path.join(_subfolder_o, 'y_val_GCN.csv'), index=False)


## Save the features.
np.save(os.path.join(_subfolder_o, 'X_train_GCN.npy'), X_train)
np.save(os.path.join(_subfolder_o, 'X_val_GCN.npy'), X_val)


# A = np.load(os.path.join(_subfolder_o, 'X_train_GNN.npy'))
# B = np.load(os.path.join(_subfolder_o, 'X_val_GNN.npy'))




# %% Decision Point of Training Target

## Remove all stocks whose first valid date is later than the cut-off date.
# Remove the trading dates earlier than the cut-off date.
def Func_Period(cutoff, data):
    cutoff = pd.to_datetime(cutoff)
    # Instablility of to_datetime()
    if cutoff.tz is None:
        cutoff = cutoff.tz_localize(data.index.tz)
    else:
        cutoff = cutoff.tz_convert(data.index.tz)
    
    return cutoff

## Extract first 8 years to be training set.
cutoff_date = [Func_Period(date, Stock) for date in cutoff_date]
_temp_train = cutoff_date[0] + pd.DateOffset(years = train_size)

# Hang Seng Index.
## Compute x-day return for market index.
HSI = HSI.xs('^HSI', level=1, axis=1)
HSI['r_Holding'] = HSI['Close'].pct_change(periods=day_inv)

cutoff_date = ['2002-11-01', '2025-10-31']
cutoff_date = [Func_Period(time, HSI) for time in cutoff_date]
HSI = HSI[HSI.index >= cutoff_date[0]]
HSI = HSI[HSI.index <= cutoff_date[1]]


# The x-day return over whole period.
HSI['r_Holding'].mean()
# The x-day return over test set.
HSI['r_Holding'][HSI.index >= Func_Period(_temp_train, HSI)].mean()
HSI['r_Holding'][HSI.index < Func_Period(_temp_train, HSI)].mean()

## Annualized
( (1 + HSI['r_Holding'][HSI.index < Func_Period(_temp_train, HSI)].mean()) ** 52 - 1) / 52 * 100

temp = HSI['Close'][HSI.index < Func_Period(_temp_train, HSI)]
( temp.iloc[-1] / temp.iloc[0] ) ** (1/8)


# %% Visualization of Whole Period Return of HSI

HSI['Cumulative_Return'] = HSI.loc[:, 'Close'] / HSI.iloc[0, 0]


plt.figure(figsize=(12, 6))
plt.plot(HSI.index, HSI['Close'], linewidth=2, color='black')

plt.xlabel("Year", fontsize=17)
plt.ylabel("Hang Seng Index", fontsize=17)
# plt.title("HSI Cumulative Return Over Time", fontsize=14)

## Add vertical line for splitting traing and test set.
# line = plt.axvline(_temp_train, color='red', linestyle='--', linewidth=1)
# line.set_dashes([10, 12])

# Add wording next to the line
# plt.text(
#     _temp_train + pd.Timedelta(days=2300), 
#     HSI['Close'].min() + 1000,   # y-position (adjust if needed)
#     "Train–Test Split Point", 
#     # rotation=90,
#     fontsize=20,
#     color='red',
#     va='bottom',       # vertical alignment
#     ha='right'         # horizontal alignment
# )


plt.tick_params(axis='both', labelsize=15)
plt.margins(x=0)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()






fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(HSI.index, HSI['Close'], linewidth=2, color='k')

# Find the first available date for each odd year
odd_year_ticks = (
    HSI.groupby(HSI.index.year)
       .apply(lambda x: x.index[0])              # first date in that year
       .loc[lambda s: s.index % 2 == 1]          # keep odd years
       .tolist()
)

# Apply ticks and labels
ax.set_xticks(odd_year_ticks)
ax.set_xticklabels([d.strftime('%Y') for d in odd_year_ticks], fontsize=14)

ax.set_xlabel("Year", fontsize=16)
ax.set_ylabel("Hang Seng Index", fontsize=16)
ax.tick_params(axis='y', labelsize=14)
ax.margins(x=0)

plt.grid(alpha=0.5)
plt.show()

















