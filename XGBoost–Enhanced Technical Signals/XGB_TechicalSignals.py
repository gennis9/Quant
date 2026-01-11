# %% Environment

from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import talib
import xgboost as xgb
import optuna
import random
import subprocess
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm

# 抑制 Optuna 的詳細輸出，只顯示錯誤與結果 (Suppress Optuna verbose output)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# %matplotlib inline

_Path = r'D:\03Programs_Clouds\Google Drive\NSYSU\05Algorithm Trading\Final Report'
os.chdir(_Path)
    
def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # If using deep learning later:
    # tf.random.set_seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

set_seed(42)

# %% Data Access

Stock = pd.read_parquet("TWSE_before2013.parquet")

# %% Data Cleaning

## Keep codes that are exactly 4 digits, all numbers.
Common = Stock[
    Stock['Code'].astype(str).str.fullmatch(r'\d{4}') &        # 4 digits
    ~Stock['Code'].astype(str).str.startswith('00') &          # no ETF
    ~Stock['Code'].astype(str).str.startswith('01') &          # no CBBC
    ~Stock['Code'].astype(str).str.startswith('91')            # no TDR
    ].copy() # 加上 .copy() 避免 SettingWithCopyWarning

## 確保 Date 欄位是 datetime 格式並排序 (Ensure datetime format and sorting)
if 'Date' in Common.columns:
    Common['Date'] = pd.to_datetime(Common['Date'])
    Common = Common.sort_values(['Code', 'Date']).reset_index(drop=True)

## Check point.
print(f"Unique Codes: {len(Common['Code'].unique())}")

# ==============================================================================
#                 👇 以下為根據您的要求生成的全新策略架構 👇
# ==============================================================================

# %% [1] Configuration & Hyperparameters (Updated)

# 交易成本
TX_COST_RATE = 0.004
SHORT_COST_RATE = 0.002

# 回測視窗
TRAIN_YEARS = 4
TEST_YEARS = 1

# XGBoost 參數
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'random_state': 42
}

# 核心策略門檻
# 根據剛才的分析，建議提高做多門檻，做空門檻設高或降低權重
THRESHOLD_LONG = 0.60  
THRESHOLD_SHORT = 0.70

# 權重計算方式
WEIGHTING_METHOD = 'constant' 
OPTUNA_TRIALS = 100

# === Groupmate B 強制出場規則 ===
STOP_LOSS_RATE = 0.15      
TRAILING_DD_RATE = 0.15    
MAX_HOLD_DAYS = 80         

# === 🔥 NEW: Short Selling Factor 🔥 ===
# 放空調節係數：
# 1.0 = 正常做空 (100% 權重)
# 0.5 = 做空減半 (50% 權重)
# 0.0 = 禁止做空
SHORT_FACTOR = 0.001


# %% [2] Feature Engineering

def calculate_market_features(df_all):
    """計算全市場特徵 (含 Market Index, MA200 Bias, Breadth, Volatility)"""
    df = df_all.copy()

    # 1. 計算個別股票的季線狀態 (用於市場廣度)
    df['MA60'] = df.groupby('Code')['Close'].transform(lambda x: x.rolling(60).mean())
    df['Above_MA60'] = (df['Close'] > df['MA60']).astype(int)

    # 2. 聚合計算全市場指標
    # 注意：這裡使用 'mean' 作為等權重指數的代理
    mkt = df.groupby('Date').agg({
        'Code': 'count',
        'Above_MA60': 'mean',     # 市場廣度 (Market Breadth)
        'Close': ['std', 'mean'], # 市場離散度與均價 (Market Index Proxy)
        'Volume': 'sum'
    })

    # 3. 欄位重新命名
    mkt.columns = ['_'.join(col).strip() for col in mkt.columns.values]
    mkt = mkt.rename(columns={
        'Above_MA60_mean': 'Market_Breadth',
        'Close_std':        'Market_Dispersion',
        'Close_mean':       'Market_Index',      # 以全市場平均收盤價作為大盤指數
        'Volume_sum':       'Market_Volume'
    })

    # 4. 計算大盤年線與乖離率 (Market Bias) - 給 AI 判斷牛熊市的重要特徵
    # 正值 = 牛市 (做多有利), 負值 = 熊市 (做空有利)
    mkt['Market_MA200'] = mkt['Market_Index'].rolling(200).mean()
    mkt['Feat_Market_Bias_200'] = (mkt['Market_Index'] / mkt['Market_MA200']) - 1

    # 5. 計算市場成交量變異 (Volume Delta)
    mkt['Market_Vol_MA5'] = mkt['Market_Volume'].rolling(5).mean()
    mkt['Market_Vol_Delta'] = mkt['Market_Volume'] / mkt['Market_Vol_MA5'].replace(0, np.nan)

    # 6. 計算市場波動率指數 (Volatility Index)
    mkt['Market_Volatility_Idx'] = mkt['Market_Dispersion'] / mkt['Market_Index']

    # 7. 處理可能的 NaN (例如前 200 天沒有 MA200)
    # 使用 bfill 或填 0 避免訓練出錯，但要小心前段資料偏差
    mkt = mkt.fillna(0)

    # 回傳需要的特徵欄位
    return mkt[[
        'Market_Breadth', 
        'Market_Dispersion', 
        'Market_Vol_Delta', 
        'Market_Volatility_Idx',
        'Market_Index', 
        'Feat_Market_Bias_200' # 新增的關鍵特徵
    ]].reset_index()

def calculate_individual_features(df_stock):
    """計算個股特徵 (含 Groupmate A 的動能與乖離)"""
    df = df_stock.copy()
    
    # 1. 基礎技術特徵
    df['ATR14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ATR50'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=50)
    df['Feat_ATR_Ratio'] = df['ATR14'] / df['ATR50']
    
    u, m, l = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    df['Feat_BB_Width'] = (u - l) / m.replace(0, np.nan)
    
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Feat_Rel_Vol'] = df['Volume'] / df['Vol_MA20'].replace(0, np.nan)
    df['Feat_RSI_Strength'] = talib.RSI(df['Close'], timeperiod=14)
    
    # 2. Groupmate A 的特徵 (Momentum & Bias)
    df['Feat_Ret_1d'] = df['Close'].pct_change(1)
    df['Feat_Ret_5d'] = df['Close'].pct_change(5)
    df['Feat_Ret_20d'] = df['Close'].pct_change(20)
    df['Feat_Vol_20d'] = df['Feat_Ret_1d'].rolling(20).std()
    
    # 乖離率
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    df['Feat_Bias_5'] = (df['Close'] / df['MA5'].replace(0, np.nan)) - 1
    df['Feat_Bias_20'] = (df['Close'] / df['MA20'].replace(0, np.nan)) - 1
    df['Feat_Bias_60'] = (df['Close'] / df['MA60'].replace(0, np.nan)) - 1
    
    # 短期量能
    df['Vol_MA5'] = df['Volume'].rolling(5).mean()
    df['Feat_Vol_Ratio_5'] = df['Volume'] / df['Vol_MA5'].replace(0, np.nan)
    df['Feat_Vol_Change'] = df['Volume'].pct_change()
    
    # 清洗 Inf/NaN
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return df

# %% [3] Strategy Logic Class (Enhanced)

class StrategyLogic:
    @staticmethod
    def strategy_rsi_atr(df, rsi_period, rsi_buy, rsi_sell, atr_period, atr_mult):
        rsi = talib.RSI(df['Close'], timeperiod=int(rsi_period))
        atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=int(atr_period))
        
        signals = np.zeros(len(df))
        position = 0; stop_price = 0.0
        
        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            if np.isnan(rsi.iloc[i]) or np.isnan(atr.iloc[i]): continue
            
            # Exit Logic (ATR Trailing Stop)
            if position == 1:
                new_stop = price - (atr_mult * atr.iloc[i])
                stop_price = max(stop_price, new_stop)
                if price < stop_price:
                    position = 0; signals[i] = 2
            elif position == -1:
                new_stop = price + (atr_mult * atr.iloc[i])
                stop_price = min(stop_price, new_stop) if stop_price > 0 else new_stop
                if price > stop_price:
                    position = 0; signals[i] = 2
            
            # Entry Logic
            elif position == 0:
                if rsi.iloc[i] < rsi_buy:
                    position = 1; signals[i] = 1; stop_price = price - (atr_mult * atr.iloc[i])
                elif rsi.iloc[i] > rsi_sell:
                    position = -1; signals[i] = -1; stop_price = price + (atr_mult * atr.iloc[i])
        return signals

    @staticmethod
    def strategy_dma_adx(df, fast_ma, slow_ma, adx_period, adx_threshold):
        ma_fast = talib.SMA(df['Close'], timeperiod=int(fast_ma))
        ma_slow = talib.SMA(df['Close'], timeperiod=int(slow_ma))
        adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=int(adx_period))
        
        signals = np.zeros(len(df))
        position = 0
        
        for i in range(1, len(df)):
            if np.isnan(ma_slow.iloc[i]) or np.isnan(adx.iloc[i]): continue
            
            # Exit Logic (Crossover Reverse)
            if position == 1:
                if ma_fast.iloc[i] < ma_slow.iloc[i]:
                    position = 0; signals[i] = 2
            elif position == -1:
                if ma_fast.iloc[i] > ma_slow.iloc[i]:
                    position = 0; signals[i] = 2
            
            # Entry Logic (Trend Strength)
            elif position == 0:
                if adx.iloc[i] > adx_threshold:
                    if ma_fast.iloc[i] > ma_slow.iloc[i] and ma_fast.iloc[i-1] <= ma_slow.iloc[i-1]:
                        position = 1; signals[i] = 1
                    elif ma_fast.iloc[i] < ma_slow.iloc[i] and ma_fast.iloc[i-1] >= ma_slow.iloc[i-1]:
                        position = -1; signals[i] = -1
        return signals

    @staticmethod
    def strategy_macd(df, fast_period, slow_period, signal_period):
        """
        MACD Advanced: 包含 Slope 與 Divergence (背離) 邏輯
        """
        macd, signal_line, hist = talib.MACD(df['Close'], 
                                           fastperiod=int(fast_period), 
                                           slowperiod=int(slow_period), 
                                           signalperiod=int(signal_period))
        
        # 1. Slope (動能變化)
        hist_slope = hist - hist.shift(3)
        
        # 2. Divergence (背離)
        lookback = 5
        price_low = df['Close'] < df['Close'].shift(lookback)
        hist_higher = hist > hist.shift(lookback)
        bull_div = price_low & hist_higher # 底背離
        
        price_high = df['Close'] > df['Close'].shift(lookback)
        hist_lower = hist < hist.shift(lookback)
        bear_div = price_high & hist_lower # 頂背離
        
        signals = np.zeros(len(df))
        position = 0
        
        hist_arr = hist.values; slope_arr = hist_slope.values
        bull_div_arr = bull_div.values; bear_div_arr = bear_div.values
        
        for i in range(lookback, len(df)):
            if np.isnan(hist_arr[i]): continue
            
            # Exit Logic (Zero Line)
            if position == 1:
                if hist_arr[i] < 0: position = 0; signals[i] = 2
            elif position == -1:
                if hist_arr[i] > 0: position = 0; signals[i] = 2
            
            # Entry Logic
            elif position == 0:
                # Long: (Gold Cross + Slope Up) OR Bull Div
                is_gc = (hist_arr[i] > 0) and (hist_arr[i-1] <= 0)
                if (is_gc and slope_arr[i] > 0) or (bull_div_arr[i] and hist_arr[i] < 0):
                    position = 1; signals[i] = 1
                
                # Short: (Dead Cross + Slope Down) OR Bear Div
                is_dc = (hist_arr[i] < 0) and (hist_arr[i-1] >= 0)
                if (is_dc and slope_arr[i] < 0) or (bear_div_arr[i] and hist_arr[i] > 0):
                    position = -1; signals[i] = -1
        return signals

# %% [4] Helper Functions

def calculate_net_pnl(entry_price, exit_price, position_type):
    """計算單筆交易的淨損益 (扣除成本)"""
    if position_type == 1: # Long
        return (exit_price - entry_price) - (exit_price * TX_COST_RATE)
    elif position_type == -1: # Short
        return (entry_price - exit_price) - (entry_price * SHORT_COST_RATE) - (exit_price * TX_COST_RATE)
    return 0.0

def get_strategy_pnl(df, signals):
    """計算策略原始損益 (用於 Optuna 優化)"""
    total = 0.0; entry = 0.0; pos = 0
    prices = df['Close'].values
    for i in range(len(signals)):
        if signals[i] in [1, -1]:
            entry = prices[i]; pos = int(signals[i])
        elif signals[i] == 2 and pos != 0:
            total += calculate_net_pnl(entry, prices[i], pos)
            pos = 0; entry = 0.0
    return total

# Optuna Objectives
def obj_rsi(t, df):
    p_rsi = t.suggest_int('rsi_p', 10, 25); p_buy = t.suggest_int('rsi_b', 20, 35)
    p_sell = t.suggest_int('rsi_s', 65, 80); p_atr = t.suggest_float('atr_m', 2.0, 4.0, step=0.1)
    return get_strategy_pnl(df, StrategyLogic.strategy_rsi_atr(df, p_rsi, p_buy, p_sell, 14, p_atr))

def obj_dma(t, df):
    p_f = t.suggest_int('f_ma', 5, 20); p_s = t.suggest_int('s_ma', 21, 60)
    p_adx = t.suggest_int('adx_p', 10, 20); p_th = t.suggest_int('adx_t', 15, 30)
    return get_strategy_pnl(df, StrategyLogic.strategy_dma_adx(df, p_f, p_s, p_adx, p_th))

def obj_macd(t, df):
    p_f = t.suggest_int('f_p', 10, 15); p_s = t.suggest_int('s_p', 20, 30); p_sig = t.suggest_int('sig_p', 5, 10)
    return get_strategy_pnl(df, StrategyLogic.strategy_macd(df, p_f, p_s, p_sig))

# XGB Data Preparation (With Peer Signals)
def prepare_xgb_data(df, signals, feature_cols, strat_id, sig_rsi, sig_dma, sig_macd):
    X, y = [], []
    prices = df['Close'].values
    feat_data = df[feature_cols].values
    
    # Peer Signals Arrays (同儕訊號)
    s_rsi = sig_rsi; s_dma = sig_dma; s_macd = sig_macd
    
    entry_idx = -1; entry_price = 0.0; pos = 0
    
    for i in range(len(signals)):
        sig = signals[i]
        if sig in [1, -1]:
            if entry_idx == -1:
                entry_idx = i; entry_price = prices[i]; pos = int(sig)
        elif sig == 2 and entry_idx != -1:
            pnl = calculate_net_pnl(entry_price, prices[i], pos)
            
            # Features: [Base Features] + [Strategy ID] + [Peer Signals]
            feats = list(feat_data[entry_idx])
            feats.append(strat_id)
            feats.append(s_rsi[entry_idx])
            feats.append(s_dma[entry_idx])
            feats.append(s_macd[entry_idx])
            
            X.append(feats)
            y.append(1 if pnl > 0 else 0)
            entry_idx = -1; pos = 0
            
    return np.array(X), np.array(y)

# %% [7] Main Rolling Framework (Updated with SHORT_FACTOR)

def run_framework(common_df):
    print(f"Weighting Method: {WEIGHTING_METHOD}")
    print(f"Short Factor: {SHORT_FACTOR} (Short Sizing Scaler)")
    print(f"Forced Exit Rules: Stop {STOP_LOSS_RATE*100}%, Trail {TRAILING_DD_RATE*100}%, Time {MAX_HOLD_DAYS} days")
    
    print("Step 1: Calculating Market Features...")
    mkt_feats = calculate_market_features(common_df)
    common_df = common_df.merge(mkt_feats, on='Date', how='left')
    
    print("Step 2: Calculating Individual Features...")
    def _apply_feat(g): return calculate_individual_features(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_final = common_df.groupby('Code', group_keys=False).apply(_apply_feat)
    
    # 確保加入 Market Bias 特徵
    base_cols = [
        'Feat_Market_Bias_200', 'Market_Breadth', 
        'Feat_ATR_Ratio', 'Feat_BB_Width', 'Feat_RSI_Strength', 'Feat_Rel_Vol',
        'Feat_Ret_1d', 'Feat_Ret_5d', 'Feat_Ret_20d', 'Feat_Vol_20d',
        'Feat_Bias_5', 'Feat_Bias_20', 'Feat_Bias_60', 'Feat_Vol_Ratio_5', 'Feat_Vol_Change'
    ]
    
    years = sorted(df_final['Date'].dt.year.unique())
    print(f"Data Range: {years[0]} - {years[-1]}")
    
    df_Exe_list = []
    
    for i in range(len(years) - TRAIN_YEARS):
        train_years = years[i : i + TRAIN_YEARS]
        test_year = years[i + TRAIN_YEARS]
        
        print(f"\n=== Window: Train {train_years} | Test {test_year} ===")
        df_train_full = df_final[df_final['Date'].dt.year.isin(train_years)].copy()
        
        # --- A. Training (省略細節，與前版相同) ---
        X_train_list, y_train_list = [], []
        # (此處保留原本的 Optuna 與 訓練資料生成迴圈...)
        # 為節省篇幅，假設訓練資料已正確生成
        # 請確保這裡使用原本完整的訓練代碼
        for t in range(len(train_years) - 1):
            opt_yr = train_years[t]; gen_yr = train_years[t+1]
            df_opt = df_train_full[df_train_full['Date'].dt.year == opt_yr]
            df_gen = df_train_full[df_train_full['Date'].dt.year == gen_yr]
            top_stocks = df_opt.groupby('Code')['Volume'].mean().nlargest(10).index.tolist()
            df_opt_s = df_opt[df_opt['Code'].isin(top_stocks)]
            
            s_rsi = optuna.create_study(direction='maximize'); s_rsi.optimize(lambda t: sum([obj_rsi(t, df_opt_s[df_opt_s['Code']==c]) for c in top_stocks]), n_trials=OPTUNA_TRIALS)
            s_dma = optuna.create_study(direction='maximize'); s_dma.optimize(lambda t: sum([obj_dma(t, df_opt_s[df_opt_s['Code']==c]) for c in top_stocks]), n_trials=OPTUNA_TRIALS)
            s_macd = optuna.create_study(direction='maximize'); s_macd.optimize(lambda t: sum([obj_macd(t, df_opt_s[df_opt_s['Code']==c]) for c in top_stocks]), n_trials=OPTUNA_TRIALS)
            p_rsi = s_rsi.best_params; p_dma = s_dma.best_params; p_macd = s_macd.best_params
            
            gen_stocks = df_gen.groupby('Code')['Volume'].mean().nlargest(50).index.tolist()
            for code in gen_stocks:
                df_s = df_gen[df_gen['Code'] == code]
                if len(df_s) < 50: continue
                sig_r = StrategyLogic.strategy_rsi_atr(df_s, p_rsi['rsi_p'], p_rsi['rsi_b'], p_rsi['rsi_s'], 14, p_rsi['atr_m'])
                sig_d = StrategyLogic.strategy_dma_adx(df_s, p_dma['f_ma'], p_dma['s_ma'], p_dma['adx_p'], p_dma['adx_t'])
                sig_m = StrategyLogic.strategy_macd(df_s, p_macd['f_p'], p_macd['s_p'], p_macd['sig_p'])
                X_r, y_r = prepare_xgb_data(df_s, sig_r, base_cols, 0, sig_r, sig_d, sig_m)
                X_d, y_d = prepare_xgb_data(df_s, sig_d, base_cols, 1, sig_r, sig_d, sig_m)
                X_m, y_m = prepare_xgb_data(df_s, sig_m, base_cols, 2, sig_r, sig_d, sig_m)
                if len(X_r)>0: X_train_list.append(X_r); y_train_list.append(y_r)
                if len(X_d)>0: X_train_list.append(X_d); y_train_list.append(y_d)
                if len(X_m)>0: X_train_list.append(X_m); y_train_list.append(y_m)

        # --- B. Train XGBoost ---
        xgb_model = None
        if len(X_train_list) > 0:
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            if len(np.unique(y_train)) > 1:
                xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
                xgb_model.fit(X_train, y_train)

        # --- C. Optimize Params (同上) ---
        last_train_year = train_years[-1]
        df_last = df_train_full[df_train_full['Date'].dt.year == last_train_year]
        top_stocks_last = df_last.groupby('Code')['Volume'].mean().nlargest(10).index.tolist()
        df_last_s = df_last[df_last['Code'].isin(top_stocks_last)]
        
        s_rsi_t = optuna.create_study(direction='maximize'); s_rsi_t.optimize(lambda t: sum([obj_rsi(t, df_last_s[df_last_s['Code']==c]) for c in top_stocks_last]), n_trials=OPTUNA_TRIALS)
        s_dma_t = optuna.create_study(direction='maximize'); s_dma_t.optimize(lambda t: sum([obj_dma(t, df_last_s[df_last_s['Code']==c]) for c in top_stocks_last]), n_trials=OPTUNA_TRIALS)
        s_macd_t = optuna.create_study(direction='maximize'); s_macd_t.optimize(lambda t: sum([obj_macd(t, df_last_s[df_last_s['Code']==c]) for c in top_stocks_last]), n_trials=OPTUNA_TRIALS)
        tp_rsi = s_rsi_t.best_params; tp_dma = s_dma_t.best_params; tp_macd = s_macd_t.best_params

        # --- D. Testing Phase (Updated with SHORT_FACTOR) ---
        print(f"  > Running Test on ALL stocks in {test_year}...")
        df_test = df_final[df_final['Date'].dt.year == test_year].copy()
        test_codes = df_test['Code'].unique()
        
        # 訓練用特徵欄位 (排除非數值)
        train_cols = [c for c in base_cols if c in df_test.columns]

        for code in tqdm(test_codes, desc="Testing Stocks"):
            df_s = df_test[df_test['Code'] == code].copy()
            if len(df_s) < 10: continue
            
            s_r = StrategyLogic.strategy_rsi_atr(df_s, tp_rsi['rsi_p'], tp_rsi['rsi_b'], tp_rsi['rsi_s'], 14, tp_rsi['atr_m'])
            s_d = StrategyLogic.strategy_dma_adx(df_s, tp_dma['f_ma'], tp_dma['s_ma'], tp_dma['adx_p'], tp_dma['adx_t'])
            s_m = StrategyLogic.strategy_macd(df_s, tp_macd['f_p'], tp_macd['s_p'], tp_macd['sig_p'])
            
            feat_data = df_s[train_cols].values
            prices = df_s['Close'].values; highs = df_s['High'].values; lows = df_s['Low'].values
            
            curr_pos = 0; entry_price = 0.0; curr_w = 0.0; entry_prob = 0.0; rec_strat_name = "None"
            highest_price = 0.0; lowest_price = 0.0; days_held = 0
            
            exec_sig_list = []; rec_strat_list = []; prob_list = []; weight_list = []; exit_reason_list = []
            net_ret_list = []; gross_ret_list = []; confusion_list = []; conf_w_ret_list = []
            cum_net = 0.0; cum_gross = 0.0
            
            for t in range(len(df_s)):
                day_exec_sig = 0; day_prob = 0.0; day_strat = "None"; day_w = 0.0; day_exit_reason = "None"
                day_net = 0.0; day_gross = 0.0; conf = 0; conf_w_ret = np.nan
                
                # --- A. Check Exit ---
                if curr_pos != 0:
                    should_exit = False; exit_reason = ""; days_held += 1
                    
                    if curr_pos == 1:
                        highest_price = max(highest_price, highs[t])
                        dd = (prices[t] - highest_price) / highest_price
                        unrealized_ret = (prices[t] - entry_price) / entry_price
                    else:
                        lowest_price = min(lowest_price, lows[t]) if lowest_price > 0 else lows[t]
                        dd = (lowest_price - prices[t]) / lowest_price
                        unrealized_ret = (entry_price - prices[t]) / entry_price
                    
                    if unrealized_ret < -STOP_LOSS_RATE: should_exit = True; exit_reason = "HardStop"
                    elif dd < -TRAILING_DD_RATE: should_exit = True; exit_reason = "Trailing"
                    elif days_held >= MAX_HOLD_DAYS: should_exit = True; exit_reason = "Time"
                    else:
                        orig_signal = 0
                        if rec_strat_name == "RSI": orig_signal = s_r[t]
                        elif rec_strat_name == "DMA": orig_signal = s_d[t]
                        elif rec_strat_name == "MACD": orig_signal = s_m[t]
                        if orig_signal == 2: should_exit = True; exit_reason = "Strategy"
                            
                    if should_exit:
                        day_exec_sig = 2; day_exit_reason = exit_reason
                        # PnL 計算：因為 entry 時已經把 curr_w 乘以了 SHORT_FACTOR，
                        # 這裡直接用 curr_w 計算，損益就會自動縮放，無需再次乘係數。
                        gross_pnl = (prices[t] - entry_price) * curr_pos * curr_w
                        exit_cost = (prices[t] * curr_w * TX_COST_RATE)
                        net_pnl = gross_pnl - exit_cost
                        day_net += net_pnl; day_gross += gross_pnl
                        
                        if curr_pos == 1: conf = 1 if prices[t] > entry_price else (2 if prices[t] < entry_price else 3)
                        else: conf = -1 if prices[t] < entry_price else (-2 if prices[t] > entry_price else -3)
                        conf_w_ret = entry_prob * net_pnl
                        
                        curr_pos = 0; entry_price = 0.0; curr_w = 0.0; entry_prob = 0.0
                        highest_price = 0.0; lowest_price = 0.0; days_held = 0; rec_strat_name = "None"
                
                # --- B. Check Entry ---
                elif curr_pos == 0 and xgb_model:
                    candidates = []
                    # 簡單檢查特徵長度 (防止 NaN 導致錯誤)
                    try:
                        f_base = list(feat_data[t])
                    except: continue

                    # 收集候選訊號
                    if s_r[t] in [1, -1]:
                        f = f_base + [0, s_r[t], s_d[t], s_m[t]]
                        try: p = xgb_model.predict_proba(np.array([f]))[0][1]
                        except: p = 0
                        candidates.append((p, "RSI", s_r[t]))
                        
                    if s_d[t] in [1, -1]:
                        f = f_base + [1, s_r[t], s_d[t], s_m[t]]
                        try: p = xgb_model.predict_proba(np.array([f]))[0][1]
                        except: p = 0
                        candidates.append((p, "DMA", s_d[t]))

                    if s_m[t] in [1, -1]:
                        f = f_base + [2, s_r[t], s_d[t], s_m[t]]
                        try: p = xgb_model.predict_proba(np.array([f]))[0][1]
                        except: p = 0
                        candidates.append((p, "MACD", s_m[t]))
                        
                    if candidates:
                        candidates.sort(key=lambda x: x[0], reverse=True)
                        winner = candidates[0]
                        best_prob = winner[0]; day_strat = winner[1]; day_exec_sig = winner[2]
                        
                        # 應用不同的門檻
                        thresh = THRESHOLD_LONG if day_exec_sig == 1 else THRESHOLD_SHORT
                        
                        if best_prob > thresh:
                            # 計算基礎權重
                            if WEIGHTING_METHOD == 'non-linear': w = ((best_prob - thresh) / (1 - thresh)) ** 2
                            elif WEIGHTING_METHOD == 'linear': w = (best_prob - thresh) / (1 - thresh)
                            else: w = 1.0
                            
                            w = min(w, 1.0)
                            
                            # 🔥🔥🔥 關鍵修改：應用 SHORT_FACTOR 🔥🔥🔥
                            # 如果是放空，將權重乘以係數 (例如 0.5)
                            # 這會直接影響進場成本、出場成本與最終損益
                            if day_exec_sig == -1:
                                w = w * SHORT_FACTOR
                            
                            # 只有當調整後的權重 > 0 才進場 (若 FACTOR=0 則不進場)
                            if w > 0:
                                day_w = w
                                curr_pos = int(day_exec_sig)
                                entry_price = prices[t]
                                curr_w = day_w
                                entry_prob = best_prob
                                rec_strat_name = day_strat
                                highest_price = prices[t]; lowest_price = prices[t]; days_held = 0
                                
                                entry_cost = (prices[t] * curr_w * TX_COST_RATE) if day_exec_sig==1 else (prices[t] * curr_w * SHORT_COST_RATE)
                                day_net -= entry_cost
                                day_prob = best_prob

                # Update Logs
                cum_net += day_net; cum_gross += day_gross
                exec_sig_list.append(day_exec_sig); rec_strat_list.append(day_strat)
                prob_list.append(day_prob); weight_list.append(day_w); exit_reason_list.append(day_exit_reason)
                net_ret_list.append(cum_net); gross_ret_list.append(cum_gross)
                confusion_list.append(conf); conf_w_ret_list.append(conf_w_ret)
            
            df_s['Rec_Strat'] = rec_strat_list; df_s['Exec_Sig'] = exec_sig_list
            df_s['Prob'] = prob_list; df_s['Weight'] = weight_list
            df_s['Exit_Reason'] = exit_reason_list
            df_s['Net_Cum_Ret'] = net_ret_list; df_s['Gross_Cum_Ret'] = gross_ret_list
            df_s['Confusion'] = confusion_list; df_s['Conf_Weighted_Ret'] = conf_w_ret_list
            df_Exe_list.append(df_s)

    if df_Exe_list:
        df_Exe = pd.concat(df_Exe_list)
        print("\n✅ df_Exe Generation Complete.")
    else:
        return None, None
    
    # Generate df_Inv
    print("Generating df_Inv...")
    inv_data = []
    for code, g in df_Exe.groupby('Code'):
        tp = (g['Confusion'] == 1).sum()
        fp = (g['Confusion'] == 2).sum()
        tn = (g['Confusion'] == -1).sum()
        fn = (g['Confusion'] == -2).sum()
        total_trades = tp + fp + tn + fn
        
        final_net = g['Net_Cum_Ret'].iloc[-1]
        final_gross = g['Gross_Cum_Ret'].iloc[-1]
        
        equity = 1 + g['Net_Cum_Ret']
        mdd = ((equity - equity.cummax()) / equity.cummax()).min()
        
        daily_pnl = g['Net_Cum_Ret'].diff().fillna(0)
        std = daily_pnl.std()
        sharpe = (daily_pnl.mean() / std * np.sqrt(252)) if std != 0 else 0
            
        inv_data.append({
            'Code': code, 'Gross_Ret': final_gross, 'Net_Ret': final_net,
            'MDD': mdd, 'Sharpe': sharpe, 'Total_Trades': total_trades,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
        })
    df_Inv = pd.DataFrame(inv_data)
    print("✅ df_Inv Generation Complete.")
    
    return df_Exe, df_Inv


# %% [6] Execution Entry Point

if __name__ == "__main__":
    if 'Common' in locals() and not Common.empty:
        try:
            print("🚀 Starting Backtest Framework...")
            df_Exe, df_Inv = run_framework(Common)
            
            if df_Inv is not None:
                print("\n" + "="*40)
                print("FINAL REPORT SUMMARY")
                print("="*40)
                
                print("\n--- df_Inv Head (Top 5) ---")
                print(df_Inv.head())
                
                print("\n--- Overall Performance Averages ---")
                print(df_Inv.mean(numeric_only=True))
                
                # 輸出 CSV
                df_Exe.to_parquet("Transaction_Ledger_df_Exe.parquet", index=False)
                df_Inv.to_csv("Strategy_Performance_df_Inv.csv", index=False)
                print("\n💾 Files saved: Transaction_Ledger_df_Exe.csv, Strategy_Performance_df_Inv.csv")
            else:
                print("Result is empty.")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Error: Data 'Common' is not available.")
        

# 2. Calculate Annualized Return with safeguards
def calculate_cagr(net_ret):
    # Case A: Bankruptcy (Loss > 100%)
    if net_ret <= -1.0:
        return (1 + net_ret) ** int(365 / _dayOfTrade) - 1
    
    # Case B: Standard Calculation
    else:
        return (1 + net_ret) ** (365 / _dayOfTrade) - 1

_dayOfTrade = (df_Exe['Date'].max() - df_Exe['Date'].min()).days
df_Inv['A.Net_Ret'] = df_Inv['Net_Ret'].apply(calculate_cagr)

(1 + 7.5659) ** (365 / _dayOfTrade) - 1
7.5659 * 746

# %% [8] Match with Industry

print("🚀 正在從證交所 (TWSE) 網站抓取最新的完整產業清單...")

def fetch_twse_data(url, market_name):
    try:
        # 讀取網頁表格
        dfs = pd.read_html(url)
        df = dfs[0]
        
        # 設定標題列 (通常第一列是標題)
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        
        # 找到 "有價證券代號及名稱" 這一欄
        col_name = [c for c in df.columns if '有價證券代號及名稱' in str(c)][0]
        
        # 拆分代號與名稱 (格式通常是 "1101　台泥")
        # 使用 split 拆開，第一個是代號，第二個是名稱
        df['Code'] = df[col_name].astype(str).str.split(n=1).str[0]
        df['Name'] = df[col_name].astype(str).str.split(n=1).str[1]
        
        # 對應產業欄位
        if '產業別' in df.columns:
            df['Industry'] = df['產業別']
        else:
            df['Industry'] = 'Unknown'
            
        print(f"✅ {market_name} 資料下載成功: {len(df)} 筆")
        return df[['Code', 'Name', 'Industry']]
        
    except Exception as e:
        print(f"❌ {market_name} 資料下載失敗: {e}")
        return pd.DataFrame()

# 1. 下載 上市 (Mode=2) 與 上櫃 (Mode=4)
df_listed = fetch_twse_data("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", "上市")
df_otc    = fetch_twse_data("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", "上櫃")

# 2. 合併兩者
df_industry_web = pd.concat([df_listed, df_otc], ignore_index=True)

# 3. 資料清理
# 過濾掉非股票的代號 (有些是 warrants 或 ETF，長度不對或開頭特殊)
# 這裡簡單過濾：只留代號長度為 4 的 (普通股)
df_industry_web = df_industry_web[df_industry_web['Code'].str.len() == 4]

# 強制清理格式
df_industry_web['Code'] = df_industry_web['Code'].astype(str).str.strip()
df_Inv['Code'] = df_Inv['Code'].astype(str).str.strip()

# 4. 合併進 df_Inv
# 移除舊欄位
for col in ['Name', 'Industry']:
    if col in df_Inv.columns:
        df_Inv = df_Inv.drop(columns=[col])

# 合併
df_Inv = pd.merge(df_Inv, df_industry_web, on='Code', how='left')

# 填補空值
df_Inv['Name'] = df_Inv['Name'].fillna('Unknown')
df_Inv['Industry'] = df_Inv['Industry'].fillna('Unknown')

print("\n📊 最終合併結果 (前 5 筆):")
print(df_Inv[['Code', 'Name', 'Industry', 'Net_Ret']].head())

# 檢查 1101 是否成功
print("\n🔍 檢查 1101 (台泥):")
print(df_Inv[df_Inv['Code'] == '1101'][['Code', 'Name', 'Industry']])


        
# %% [9] Analysis: Profitability by Probability Bucket (Fixed Version)

def analyze_prob_performance_fixed(df_exe):
    print("\n" + "="*60)
    print("📊 Analysis: Net Return by ENTRY Probability (Fixed)")
    print("="*60)
    
    df = df_exe.copy()
    
    # 1. 修正：回溯「進場時」的機率 (Entry Probability)
    # 邏輯：建立一個新欄位，只在進場日填入 Prob，然後向下填充 (Forward Fill) 到出場日
    df['Entry_Prob_Fixed'] = np.nan
    
    # 標記進場點 (Exec_Sig 為 1 或 -1)
    entry_mask = df['Exec_Sig'].isin([1, -1])
    df.loc[entry_mask, 'Entry_Prob_Fixed'] = df.loc[entry_mask, 'Prob']
    
    # 針對每一檔股票進行 Forward Fill
    # 這樣出場日 (Exec_Sig=2) 就會拿到最近一次進場日的機率
    df['Entry_Prob_Fixed'] = df.groupby('Code')['Entry_Prob_Fixed'].ffill()
    
    # 2. 過濾：只保留有結算的出場日 (Conf_Weighted_Ret 非空)
    df_res = df.dropna(subset=['Conf_Weighted_Ret']).copy()
    
    if df_res.empty:
        print("No closed trades to analyze.")
        return

    # 3. 建立 1% 的區間 (Bucket) 使用修正後的進場機率
    df_res['Prob_Bucket'] = np.floor(df_res['Entry_Prob_Fixed'] * 100) / 100
    
    # 4. 分組統計
    bucket_stats = df_res.groupby('Prob_Bucket').agg({
        'Conf_Weighted_Ret': 'sum',          
        'Net_Cum_Ret': 'count',              
        'Confusion': lambda x: (x.abs()==1).sum() / x.count() 
    }).rename(columns={'Net_Cum_Ret': 'Trade_Count', 'Confusion': 'Win_Rate'})
    
    # 5. 輸出報表
    bucket_stats = bucket_stats.sort_index(ascending=True)
    
    print(f"{'Entry Prob':<15} | {'Trades':>8} | {'Total Conf_W_Ret':>18} | {'Win Rate':>10}")
    print("-" * 65)
    
    for prob, row in bucket_stats.iterrows():
        range_str = f"{prob:.2f} - {prob+0.01:.2f}"
        print(f"{range_str:<15} | {int(row['Trade_Count']):>8} | {row['Conf_Weighted_Ret']:>18.6f} | {row['Win_Rate']:>10.1%}")

    # 6. 繪圖
    if len(bucket_stats) > 0:
        plt.figure(figsize=(12, 6))
        # 顏色：紅(虧) 綠(賺)
        colors = ['red' if x < 0 else 'green' for x in bucket_stats['Conf_Weighted_Ret']]
        
        plt.bar(bucket_stats.index + 0.005, bucket_stats['Conf_Weighted_Ret'], width=0.008, color=colors, alpha=0.7)
        plt.xlabel('XGB Entry Probability (Confidence)')
        plt.ylabel('Total Conf_Weighted_Ret')
        plt.title('Profitability by Entry Confidence Level (Fixed)')
        plt.grid(True, alpha=0.3)
        plt.show()

# 執行修正後的分析
if 'df_Exe' in locals() and not df_Exe.empty:
    analyze_prob_performance_fixed(df_Exe)
else:
    print("df_Exe not found.")
    
    
    
    
# %% [10] Analysis: Weight Effect Analysis (Double Weighted)

def analyze_weight_effect(df_exe):
    print("\n" + "="*60)
    print("📊 Analysis: Weight Effect (Net_PnL * Weight) by Confidence")
    print("   Formula: (Conf_Weighted_Ret / Prob) * Weight")
    print("   Note: This effectively squares the weight impact on PnL.")
    print("="*60)
    
    df = df_exe.copy()
    
    # 1. 回溯「進場時」的機率與權重
    # 因為平倉日 (Exec_Sig=2) 的 Weight 是 0，必須從進場日帶過來
    df['Entry_Prob_Fixed'] = np.nan
    df['Entry_Weight_Fixed'] = np.nan
    
    # 標記進場點
    entry_mask = df['Exec_Sig'].isin([1, -1])
    df.loc[entry_mask, 'Entry_Prob_Fixed'] = df.loc[entry_mask, 'Prob']
    df.loc[entry_mask, 'Entry_Weight_Fixed'] = df.loc[entry_mask, 'Weight']
    
    # Forward Fill
    df['Entry_Prob_Fixed'] = df.groupby('Code')['Entry_Prob_Fixed'].ffill()
    df['Entry_Weight_Fixed'] = df.groupby('Code')['Entry_Weight_Fixed'].ffill()
    
    # 2. 過濾出平倉交易
    df_res = df.dropna(subset=['Conf_Weighted_Ret']).copy()
    
    if df_res.empty:
        print("No closed trades to analyze.")
        return

    # 3. 計算使用者要求的指標
    # New Metric = (Conf_Weighted_Ret / Prob) * Weight
    # 這裡的 Prob 和 Weight 都是進場當下的值
    df_res['Weight_Effect_Ret'] = (df_res['Conf_Weighted_Ret'] / df_res['Entry_Prob_Fixed']) * df_res['Entry_Weight_Fixed']
    
    # 4. 建立 1% 區間
    df_res['Prob_Bucket'] = np.floor(df_res['Entry_Prob_Fixed'] * 100) / 100
    
    # 5. 分組統計
    bucket_stats = df_res.groupby('Prob_Bucket').agg({
        'Weight_Effect_Ret': 'sum',
        'Net_Cum_Ret': 'count', # 交易次數
        'Confusion': lambda x: (x.abs()==1).sum() / x.count()
    }).rename(columns={'Net_Cum_Ret': 'Trade_Count', 'Confusion': 'Win_Rate'})
    
    # 6. 輸出報表
    bucket_stats = bucket_stats.sort_index(ascending=True)
    
    print(f"{'Entry Prob':<15} | {'Trades':>8} | {'Weight_Effect_Ret':>20} | {'Win Rate':>10}")
    print("-" * 65)
    
    for prob, row in bucket_stats.iterrows():
        range_str = f"{prob:.2f} - {prob+0.01:.2f}"
        print(f"{range_str:<15} | {int(row['Trade_Count']):>8} | {row['Weight_Effect_Ret']:>20.6f} | {row['Win_Rate']:>10.1%}")

    # 7. 繪圖
    if len(bucket_stats) > 0:
        plt.figure(figsize=(12, 6))
        colors = ['red' if x < 0 else 'purple' for x in bucket_stats['Weight_Effect_Ret']]
        
        plt.bar(bucket_stats.index + 0.005, bucket_stats['Weight_Effect_Ret'], width=0.008, color=colors, alpha=0.7)
        plt.xlabel('XGB Entry Probability')
        plt.ylabel('Total Weight_Effect_Ret')
        plt.title('Impact of Non-Linear Weighting by Confidence Level')
        plt.grid(True, alpha=0.3)
        plt.show()

# 執行分析
if 'df_Exe' in locals() and not df_Exe.empty:
    analyze_weight_effect(df_Exe)
else:
    print("df_Exe not found.")
    
    
# %% [11] Analysis: Portfolio Cumulative Returns (Equal Weighted)

def plot_portfolio_gross_net_styled(df_exe, total_stocks_count=746):
    print("🚀 繪製投資組合圖表 (加大軸字體 + 統一色塊風格)...")
    
    if df_exe is None or df_exe.empty:
        print("Error: df_Exe is empty.")
        return

    # 1. 資料處理
    df = df_exe[['Date', 'Code', 'Net_Cum_Ret', 'Gross_Cum_Ret', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.sort_values(['Code', 'Date'])
    
    # Net Processing
    df['Daily_Net_PnL'] = df.groupby('Code')['Net_Cum_Ret'].diff().fillna(0)
    mask_net = (df['Net_Cum_Ret'] != 0) & (df['Daily_Net_PnL'] == 0)
    df.loc[mask_net, 'Daily_Net_PnL'] = df.loc[mask_net, 'Net_Cum_Ret']
    
    # Gross Processing
    df['Daily_Gross_PnL'] = df.groupby('Code')['Gross_Cum_Ret'].diff().fillna(0)
    mask_gross = (df['Gross_Cum_Ret'] != 0) & (df['Daily_Gross_PnL'] == 0)
    df.loc[mask_gross, 'Daily_Gross_PnL'] = df.loc[mask_gross, 'Gross_Cum_Ret']
    
    # Convert to Percentage Contribution
    df['Daily_Net_Contrib'] = df['Daily_Net_PnL'] / df['Close']
    df['Daily_Gross_Contrib'] = df['Daily_Gross_PnL'] / df['Close']
    
    # Aggregate
    daily_stats = df.groupby('Date')[['Daily_Net_Contrib', 'Daily_Gross_Contrib']].sum()
    daily_stats /= total_stocks_count
    cum_stats = daily_stats.cumsum()
    
    # 2. 找出關鍵點
    net_min_val = cum_stats['Daily_Net_Contrib'].min()
    net_min_date = cum_stats['Daily_Net_Contrib'].idxmin()
    gross_min_val = cum_stats['Daily_Gross_Contrib'].min()
    gross_min_date = cum_stats['Daily_Gross_Contrib'].idxmin()
    
    net_final_val = cum_stats['Daily_Net_Contrib'].iloc[-1]
    gross_final_val = cum_stats['Daily_Gross_Contrib'].iloc[-1]
    last_date = cum_stats.index[-1]

    # --- 繪圖 ---
    plt.figure(figsize=(14, 8))
    
    # 畫線
    plt.plot(cum_stats.index, cum_stats['Daily_Gross_Contrib'], 
             label='Gross Return', color='blue', linewidth=2, alpha=0.6, linestyle='--')
    
    plt.plot(cum_stats.index, cum_stats['Daily_Net_Contrib'], 
             label='Net Return', color='red', linewidth=2.5)
    
    plt.fill_between(cum_stats.index, cum_stats['Daily_Gross_Contrib'], cum_stats['Daily_Net_Contrib'], 
                     color='gray', alpha=0.1, label='Transaction Costs')
    
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    # 樣式設定
    style_net = dict(boxstyle="round,pad=0.3", fc="red", ec="darkred", alpha=0.9)
    style_gross = dict(boxstyle="round,pad=0.3", fc="blue", ec="navy", alpha=0.8)

    # 標記最低點
    plt.scatter(net_min_date, net_min_val, color='black', s=100, marker='v', zorder=10)
    plt.annotate(f"Lowest Net: {net_min_val:.2%}", 
                 xy=(net_min_date, net_min_val), 
                 xytext=(10, -20), textcoords='offset points', 
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=12, fontweight='bold', color='white',
                 bbox=style_net)

    plt.scatter(gross_min_date, gross_min_val, color='black', s=80, marker='v', zorder=10)
    plt.annotate(f"Lowest Gross: {gross_min_val:.2%}", 
                 xy=(gross_min_date, gross_min_val), 
                 xytext=(0, 30), textcoords='offset points', 
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=12, fontweight='bold', color='white',
                 bbox=style_gross)

    # 標記最終報酬
    plt.annotate(f"Final Net: {net_final_val:.2%}", 
                 xy=(last_date, net_final_val), 
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=12, fontweight='bold', color='white',
                 verticalalignment='center',
                 bbox=style_net)

    plt.annotate(f"Final Gross: {gross_final_val:.2%}", 
                 xy=(last_date, gross_final_val), 
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=12, fontweight='bold', color='white',
                 verticalalignment='center',
                 bbox=style_gross)

    # ==========================================
    # 🔥🔥🔥 關鍵修改：加大字體 🔥🔥🔥
    # ==========================================
    
    # 1. 加大標題 (Title) -> 20
    plt.title(f'Portfolio Cumulative Return (Equal Weight, N={total_stocks_count})', fontsize=20, fontweight='bold', pad=15)
    
    # 2. 加大軸標籤 (Labels) -> 16
    plt.xlabel('Date', fontsize=18, fontweight='bold')
    plt.ylabel('Cumulative Return (%)', fontsize=18, fontweight='bold')
    
    # 3. 加大軸刻度數字 (Ticks) -> 14
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Y軸格式
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # 加大圖例字體
    plt.legend(loc='upper left', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ 計算完成。")
    
    return net_final_val, gross_final_val

# 執行
final_net, final_gross = plot_portfolio_gross_net_styled(df_Exe)
print(f"外部接收到的數值 -> Net: {final_net}, Gross: {final_gross}")
costs_incurred = final_gross - final_net

_dayOfTrade = (df_Exe['Date'].max() - df_Exe['Date'].min()).days
(1 + final_gross) ** (365 / _dayOfTrade) - 1
(1 + final_net) ** (365 / _dayOfTrade) - 1
(1 + final_gross) ** (365 / _dayOfTrade) - (1 + final_net) ** (365 / _dayOfTrade)


    
# %% [12] Analysis: Plot Cumulative Gross Return for the Median Stock

def plot_median_stock_performance(df_inv, df_exe):
    print("\n" + "="*60)
    print("📊 Analysis: Median Performer Deep Dive")
    print("   Finding the stock with median Net_Ret in df_Inv")
    print("="*60)
    
    if df_inv is None or df_inv.empty or df_exe is None or df_exe.empty:
        print("Dataframes are empty. Cannot perform analysis.")
        return

    # 1. 找出中位數股票 (Find Median Stock)
    # 根據 Net_Ret 排序
    df_sorted = df_inv.sort_values(by='Net_Ret').reset_index(drop=True)
    
    # 取得中位數索引
    median_idx = len(df_sorted) // 2
    median_stock_info = df_sorted.iloc[median_idx]
    
    target_code = median_stock_info['Code']
    target_net_ret = median_stock_info['Net_Ret']
    target_gross_ret = median_stock_info['Gross_Ret']
    
    print(f"🎯 Median Stock Found: {target_code}")
    print(f"   Net Return:   {target_net_ret:.4f}")
    print(f"   Gross Return: {target_gross_ret:.4f}")
    print(f"   Total Trades: {median_stock_info['Total_Trades']}")

    # 2. 從 df_Exe 提取該股票的詳細數據 (Extract Data)
    stock_data = df_exe[df_exe['Code'] == target_code].copy()
    
    # 確保日期格式正確
    if not np.issubdtype(stock_data['Date'].dtype, np.datetime64):
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
    # 3. 繪圖 (Plot)
    plt.figure(figsize=(12, 6))
    
    # 繪製 Gross Return
    plt.plot(stock_data['Date'], stock_data['Gross_Cum_Ret'], 
             label=f'Gross Cum Ret ({target_code})', color='blue', linewidth=1.5)
    
    # 也可以順便畫出 Net Return 供比較 (虛線)
    plt.plot(stock_data['Date'], stock_data['Net_Cum_Ret'], 
             label=f'Net Cum Ret ({target_code})', color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.title(f'Performance of Median Stock: {target_code} (Net Ret: {target_net_ret:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# 執行分析
if 'df_Inv' in locals() and 'df_Exe' in locals():
    plot_median_stock_performance(df_Inv, df_Exe)
else:
    print("df_Inv or df_Exe not found. Please run the main backtest first.")


# %% [Analysis] Re-check Median Stock 1608 Performance
if 'df_Inv' in locals() and 'df_Exe' in locals():
    # 指定看 1608 這支股票 (原本的中位數/大賠股票)
    target_code = '1608' 
    
    stock_data = df_Exe[df_Exe['Code'] == target_code].copy()
    if not stock_data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data['Date'], stock_data['Gross_Cum_Ret'], label=f'Gross Cum Ret ({target_code})', color='blue')
        plt.plot(stock_data['Date'], stock_data['Net_Cum_Ret'], label=f'Net Cum Ret ({target_code})', color='red', linestyle='--')
        
        # 標示出場點 (如果有)
        exits = stock_data[stock_data['Exec_Sig'] == 2]
        plt.scatter(exits['Date'], exits['Net_Cum_Ret'], color='black', marker='x', s=100, label='Exit')
        
        plt.title(f'Performance Verification: Stock {target_code} (With Forced Exit Rules)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Final Net Return for {target_code}: {stock_data['Net_Cum_Ret'].iloc[-1]:.4f}")
    else:
        print(f"Stock {target_code} not found in results.")


# %% [14] Analysis: Sum Conf_Weighted_Ret by Year

def analyze_yearly_performance(df_exe):
    print("\n" + "="*60)
    print("📊 Analysis: Total Conf_Weighted_Ret by Year")
    print("="*60)
    
    if df_exe is None or df_exe.empty:
        print("df_Exe is empty. Cannot perform analysis.")
        return

    # 1. 複製資料並確保日期格式正確
    df = df_exe.copy()
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
        df['Date'] = pd.to_datetime(df['Date'])
        
    # 2. 提取年份
    df['Year'] = df['Date'].dt.year
    
    # 3. 分組加總 (Group by Year & Sum)
    yearly_sum = df.groupby('Year')['Conf_Weighted_Ret'].sum()
    
    # 4. 輸出報表
    print(f"{'Year':<6} | {'Total Conf_Weighted_Ret':>25}")
    print("-" * 35)
    for year, value in yearly_sum.items():
        print(f"{year:<6} | {value:>25.6f}")
        
    # 5. 繪圖 (Bar Chart)
    plt.figure(figsize=(10, 6))
    # 正報酬為綠色，負報酬為紅色
    colors = ['red' if v < 0 else 'green' for v in yearly_sum.values]
    bars = plt.bar(yearly_sum.index, yearly_sum.values, color=colors, alpha=0.7)
    
    plt.title('Total Confidence-Weighted Return by Year')
    plt.xlabel('Year')
    plt.ylabel('Sum of Conf_Weighted_Ret')
    plt.xticks(yearly_sum.index) # 確保每年都顯示
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 在柱狀圖上標示數值
    for bar in bars:
        height = bar.get_height()
        offset = 5 if height >= 0 else -15
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                 f'{height:.0f}',
                 ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
                 
    plt.show()

# 執行分析
if 'df_Exe' in locals() and not df_Exe.empty:
    analyze_yearly_performance(df_Exe)
else:
    print("df_Exe not found. Please run the main backtest first.")
    
    
# %% [Analysis] Total Error Rates Analysis (FPR & FNR)

def analyze_total_error_rates(df_inv):
    print("\n" + "="*60)
    print("📊 Analysis: Portfolio Total Error Rates (FPR & FNR)")
    print("="*60)
    
    if df_inv is None or df_inv.empty:
        print("df_Inv is empty. Cannot perform analysis.")
        return

    # 1. 加總所有個股的混淆矩陣數值
    total_tp = df_inv['TP'].sum() # Long Win (市場漲，做對了)
    total_fp = df_inv['FP'].sum() # Long Loss (市場跌，做錯了)
    total_tn = df_inv['TN'].sum() # Short Win (市場跌，做對了)
    total_fn = df_inv['FN'].sum() # Short Loss (市場漲，做錯了)
    
    total_trades = total_tp + total_fp + total_tn + total_fn
    
    # 2. 計算比率
    # FPR: 在所有"該跌"的時候(TN+FP)，我們誤判做多(FP)的機率
    actual_negatives = total_fp + total_tn
    fpr = total_fp / actual_negatives if actual_negatives > 0 else 0.0
    
    # FNR: 在所有"該漲"的時候(TP+FN)，我們誤判做空(FN)的機率
    actual_positives = total_tp + total_fn
    fnr = total_fn / actual_positives if actual_positives > 0 else 0.0
    
    # Precision (查準率): 做多時，真的漲的機率
    precision_long = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    
    # Precision (Short): 做空時，真的跌的機率
    precision_short = total_tn / (total_tn + total_fn) if (total_tn + total_fn) > 0 else 0.0

    # 3. 輸出報表
    print(f"{'Metric':<25} | {'Value':<15} | {'Description'}")
    print("-" * 75)
    print(f"{'Total Trades':<25} | {int(total_trades):<15} | Total executed signals")
    print(f"{'Total Long Wins (TP)':<25} | {int(total_tp):<15} | Correctly bought dip/trend")
    print(f"{'Total Long Loss (FP)':<25} | {int(total_fp):<15} | Bought but price fell")
    print(f"{'Total Short Wins (TN)':<25} | {int(total_tn):<15} | Correctly shorted top")
    print(f"{'Total Short Loss (FN)':<25} | {int(total_fn):<15} | Shorted but price rose")
    print("-" * 75)
    
    # 重點指標
    print(f"{'False Positive Rate':<25} | {fpr:>14.2%} | Long Error Rate (Bad Longs / All Downtrends)")
    print(f"{'False Negative Rate':<25} | {fnr:>14.2%} | Short Error Rate (Bad Shorts / All Uptrends)")
    print("-" * 75)
    print(f"{'Long Precision':<25} | {precision_long:>14.2%} | Win Rate when Long")
    print(f"{'Short Precision':<25} | {precision_short:>14.2%} | Win Rate when Short")
    print("="*60)
    
    # 4. 繪製混淆矩陣圖 (Confusion Matrix Visualization)
    try:
        import seaborn as sns
        
        # 建立矩陣數據 (2x2)
        #           Predicted Long    Predicted Short
        # Actual Up      TP                FN
        # Actual Down    FP                TN
        # 注意：這裡的標籤是「預測方向」，行是「實際方向」
        
        matrix_data = np.array([[total_tp, total_fn], 
                                [total_fp, total_tn]])
        
        plt.figure(figsize=(8, 6))
        
        # 使用百分比註釋
        group_names = ['TP (Long Win)', 'FN (Short Loss)', 'FP (Long Loss)', 'TN (Short Win)']
        group_counts = ["{0:0.0f}".format(value) for value in matrix_data.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in matrix_data.flatten()/np.sum(matrix_data)]
        
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        sns.heatmap(matrix_data, annot=labels, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Pred Long', 'Pred Short'],
                    yticklabels=['Actual Up', 'Actual Down'])
        
        plt.title('Portfolio Total Confusion Matrix')
        plt.ylabel('Actual Market Direction')
        plt.xlabel('Model Action')
        plt.show()
        
    except ImportError:
        print("Seaborn not installed, skipping heatmap.")

# 執行分析
if 'df_Inv' in locals() and not df_Inv.empty:
    analyze_total_error_rates(df_Inv)
else:
    print("df_Inv not found. Please run the backtest framework first.")


# %% [10] Analysis: Profitability by Bucket (Split Long vs Short)

def analyze_long_short_profitability(df_exe):
    print("\n" + "="*60)
    print("📊 Analysis: Long vs Short Profitability by Probability")
    print("="*60)
    
    df = df_exe.copy()
    
    # 1. 資料前處理：回溯「進場時」的 機率 與 方向
    # 建立新欄位
    df['Entry_Prob_Fixed'] = np.nan
    df['Entry_Type'] = np.nan # 1: Long, -1: Short
    
    # 標記進場點 (Exec_Sig 為 1 或 -1)
    entry_mask = df['Exec_Sig'].isin([1, -1])
    
    # 填入進場當下的資訊
    df.loc[entry_mask, 'Entry_Prob_Fixed'] = df.loc[entry_mask, 'Prob']
    df.loc[entry_mask, 'Entry_Type'] = df.loc[entry_mask, 'Exec_Sig']
    
    # Forward Fill: 讓出場日 (Exec_Sig=2) 拿到該筆交易的進場機率與方向
    df['Entry_Prob_Fixed'] = df.groupby('Code')['Entry_Prob_Fixed'].ffill()
    df['Entry_Type'] = df.groupby('Code')['Entry_Type'].ffill()
    
    # 2. 過濾：只保留有結算的出場日
    df_res = df.dropna(subset=['Conf_Weighted_Ret']).copy()
    
    if df_res.empty:
        print("No closed trades to analyze.")
        return

    # 3. 建立機率區間 (Bucket)
    df_res['Prob_Bucket'] = np.floor(df_res['Entry_Prob_Fixed'] * 100) / 100
    
    # 4. 拆分 Long 與 Short 資料集
    df_long = df_res[df_res['Entry_Type'] == 1].copy()
    df_short = df_res[df_res['Entry_Type'] == -1].copy()
    
    # --- 內部函式：統計與列印 ---
    def process_and_print(sub_df, title_prefix):
        if sub_df.empty:
            print(f"\nNo {title_prefix} trades found.")
            return None
            
        stats = sub_df.groupby('Prob_Bucket').agg({
            'Conf_Weighted_Ret': 'sum',
            'Net_Cum_Ret': 'count', # Trade Count
            # Win Rate 計算: Long(Confusion=1), Short(Confusion=-1)
            'Confusion': lambda x: (x.abs() == 1).sum() / x.count()
        }).rename(columns={'Net_Cum_Ret': 'Trade_Count', 'Confusion': 'Win_Rate'})
        
        stats = stats.sort_index()
        
        print(f"\n--- {title_prefix} Performance by Probability ---")
        print(f"{'Entry Prob':<15} | {'Trades':>8} | {'Total Conf_W_Ret':>18} | {'Win Rate':>10}")
        print("-" * 65)
        for prob, row in stats.iterrows():
            range_str = f"{prob:.2f} - {prob+0.01:.2f}"
            print(f"{range_str:<15} | {int(row['Trade_Count']):>8} | {row['Conf_Weighted_Ret']:>18.6f} | {row['Win_Rate']:>10.1%}")
            
        return stats

    # 5. 執行統計
    stats_long = process_and_print(df_long, "LONG (Buy)")
    stats_short = process_and_print(df_short, "SHORT (Sell)")
    
    # 6. 繪圖 (雙子圖比較)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot Long
    if stats_long is not None and not stats_long.empty:
        colors_l = ['green' if x > 0 else 'red' for x in stats_long['Conf_Weighted_Ret']]
        axes[0].bar(stats_long.index + 0.005, stats_long['Conf_Weighted_Ret'], width=0.008, color=colors_l, alpha=0.7)
        axes[0].set_title('LONG Strategy Profitability by Confidence')
        axes[0].set_ylabel('Total Conf_Weighted_Ret')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(0, color='black', linewidth=0.8)
    
    # Plot Short
    if stats_short is not None and not stats_short.empty:
        colors_s = ['green' if x > 0 else 'red' for x in stats_short['Conf_Weighted_Ret']]
        axes[1].bar(stats_short.index + 0.005, stats_short['Conf_Weighted_Ret'], width=0.008, color=colors_s, alpha=0.7)
        axes[1].set_title('SHORT Strategy Profitability by Confidence')
        axes[1].set_xlabel('XGB Entry Probability (Confidence)')
        axes[1].set_ylabel('Total Conf_Weighted_Ret')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='black', linewidth=0.8)
        
    plt.tight_layout()
    plt.show()

# 執行分析
if 'df_Exe' in locals() and not df_Exe.empty:
    analyze_long_short_profitability(df_Exe)
else:
    print("df_Exe not found.")


# %% [11] Analysis: Profitability by Year (Split Long vs Short) - Fixed Layout

def analyze_long_short_profitability_by_year(df_exe):
    print("\n" + "="*60)
    print("📊 Analysis: Long vs Short Profitability by Year")
    print("="*60)
    
    df = df_exe.copy()
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
        df['Date'] = pd.to_datetime(df['Date'])
        
    # 1. 資料前處理
    df['Entry_Type'] = np.nan 
    entry_mask = df['Exec_Sig'].isin([1, -1])
    df.loc[entry_mask, 'Entry_Type'] = df.loc[entry_mask, 'Exec_Sig']
    df['Entry_Type'] = df.groupby('Code')['Entry_Type'].ffill()
    df['Year'] = df['Date'].dt.year
    
    # 2. 過濾結算交易
    df_res = df.dropna(subset=['Conf_Weighted_Ret']).copy()
    
    if df_res.empty:
        print("No closed trades to analyze.")
        return

    df_long = df_res[df_res['Entry_Type'] == 1].copy()
    df_short = df_res[df_res['Entry_Type'] == -1].copy()
    
    # --- 內部統計函式 ---
    def get_stats(sub_df):
        if sub_df.empty: return None
        return sub_df.groupby('Year').agg({
            'Conf_Weighted_Ret': 'sum',
            'Net_Cum_Ret': 'count',
            'Confusion': lambda x: (x.abs() == 1).sum() / x.count()
        })

    stats_long = get_stats(df_long)
    stats_short = get_stats(df_short)

    # 3. 繪圖 (優化版 Layout)
    # 增加高度以避免跑版
    fig, axes = plt.subplots(2, 1, figsize=(12, 12)) 
    
    # 畫圖函式
    def plot_bars(ax, stats, title, color_logic):
        if stats is None or stats.empty:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            return
            
        colors = [color_logic(x) for x in stats['Conf_Weighted_Ret']]
        bars = ax.bar(stats.index, stats['Conf_Weighted_Ret'], color=colors, alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Weighted Return', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.axhline(0, color='black', linewidth=1)
        
        # 強制設定 X 軸為整數年份，避免出現 2016.5 這種小數
        ax.set_xticks(stats.index)
        ax.set_xticklabels(stats.index, fontsize=11)
        
        # 標示數值 (優化位置)
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        for bar in bars:
            height = bar.get_height()
            # 根據正負值調整 offset 方向
            offset = y_range * 0.02 if height >= 0 else -y_range * 0.05
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=10, fontweight='bold', color='black')

    # Plot Long
    plot_bars(axes[0], stats_long, 'LONG Strategy Profitability by Year', 
              lambda x: 'forestgreen' if x > 0 else 'firebrick')
    
    # Plot Short
    plot_bars(axes[1], stats_short, 'SHORT Strategy Profitability by Year', 
              lambda x: 'limegreen' if x > 0 else 'indianred')
        
    plt.tight_layout(pad=3.0) # 增加圖表間距
    plt.show()
    
    # 4. 列印報表
    for name, stats in [("LONG (Buy)", stats_long), ("SHORT (Sell)", stats_short)]:
        if stats is not None:
            print(f"\n--- {name} Performance by Year ---")
            print(f"{'Year':<6} | {'Trades':>8} | {'Total Return':>15} | {'Win Rate':>10}")
            print("-" * 50)
            for year, row in stats.iterrows():
                print(f"{year:<6} | {int(row['Net_Cum_Ret']):>8} | {row['Conf_Weighted_Ret']:>15.4f} | {row['Confusion']:>10.1%}")

# 執行分析
if 'df_Exe' in locals() and not df_Exe.empty:
    analyze_long_short_profitability_by_year(df_Exe)
else:
    print("df_Exe not found.")

# %% [12] Analysis: Profitability by Year & Probability Bucket (Detailed)

def analyze_year_bucket_distribution(df_exe):
    print("\n" + "="*60)
    print("📊 Analysis: Profitability by Year & 5% Probability Buckets")
    print("="*60)
    
    df = df_exe.copy()
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
        df['Date'] = pd.to_datetime(df['Date'])

    # 1. 資料前處理：回溯進場資訊
    df['Entry_Prob_Fixed'] = np.nan
    df['Entry_Type'] = np.nan 
    
    entry_mask = df['Exec_Sig'].isin([1, -1])
    df.loc[entry_mask, 'Entry_Prob_Fixed'] = df.loc[entry_mask, 'Prob']
    df.loc[entry_mask, 'Entry_Type'] = df.loc[entry_mask, 'Exec_Sig']
    
    df['Entry_Prob_Fixed'] = df.groupby('Code')['Entry_Prob_Fixed'].ffill()
    df['Entry_Type'] = df.groupby('Code')['Entry_Type'].ffill()
    
    # 建立年份
    df['Year'] = df['Date'].dt.year
    
    # 只保留已結算交易
    df_res = df.dropna(subset=['Conf_Weighted_Ret']).copy()
    
    if df_res.empty:
        print("No trades found.")
        return

    # 2. 建立 5% Buckets (0.50, 0.55, 0.60 ...)
    # 將機率無條件捨去到小數第二位，並以 0.05 為單位
    # 例如 0.63 -> 0.60, 0.68 -> 0.65
    df_res['Prob_Bucket'] = (np.floor(df_res['Entry_Prob_Fixed'] / 0.05) * 0.05).round(2)
    
    # 定義我們要觀察的 Bucket 範圍 (從 0.50 到 0.95)
    all_buckets = np.arange(0.50, 1.00, 0.05).round(2)
    years = sorted(df_res['Year'].unique())
    
    # 3. 繪圖設定
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # 設定長條圖寬度邏輯
    # 假設一年佔據 X 軸長度為 1.0
    # 我們留 0.15 的間隙，剩 0.85 給 Bar
    # 總共有 10 個 buckets (0.50 ~ 0.95)
    total_width = 0.85
    bar_width = total_width / len(all_buckets)
    
    # --- 內部繪圖函式 ---
    def plot_layer(ax, entry_type, title):
        # 篩選資料
        sub_df = df_res[df_res['Entry_Type'] == entry_type]
        if sub_df.empty:
            ax.text(0.5, 0.5, 'No Trades', ha='center', transform=ax.transAxes)
            return

        # 聚合數據: [Year, Bucket] -> Sum Return
        agg = sub_df.groupby(['Year', 'Prob_Bucket'])['Conf_Weighted_Ret'].sum()
        
        # 繪製基準線
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        
        # 迴圈繪製
        # 外層：年份
        for year in years:
            # 內層：每一個 Bucket
            for i, bucket in enumerate(all_buckets):
                if (year, bucket) in agg.index:
                    val = agg.loc[(year, bucket)]
                    
                    # 計算 X 座標
                    # Year 是中心點，我們先移到最左邊，然後加上 bucket 偏移量
                    # offset = (i - len/2) * w
                    x_center = year
                    x_offset = (i - len(all_buckets)/2 + 0.5) * bar_width
                    x_pos = x_center + x_offset
                    
                    color = 'forestgreen' if val > 0 else 'firebrick'
                    edge_color = 'white' if abs(val) > 0 else 'none'
                    
                    # 畫 Bar (align='center')
                    # linewidth=0.5 讓 bar 之間有一條極細的白線區隔，避免視覺糊在一起
                    ax.bar(x_pos, val, width=bar_width, color=color, edgecolor=edge_color, linewidth=0.3, alpha=0.85)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Profit / Loss', fontsize=12)
        ax.grid(True, axis='y', alpha=0.2, linestyle='--')
        
        # 標記年份分隔線 (虛線)
        for y in years[:-1]:
            ax.axvline(y + 0.5, color='gray', linestyle=':', alpha=0.3)

    # 4. 執行繪圖
    plot_layer(axes[0], 1, 'LONG Strategy: PnL by Year & Confidence Bucket (Left=Low Conf, Right=High Conf)')
    plot_layer(axes[1], -1, 'SHORT Strategy: PnL by Year & Confidence Bucket')
    
    # 5. 設定 X 軸標籤
    axes[1].set_xticks(years)
    axes[1].set_xticklabels(years, fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Year (Internal Bars: 50% -> 95% Confidence)', fontsize=12)
    
    # 增加圖例說明 Bar 的意義
    # 手動建立一個 Legend 說明 Bar 的排列順序
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=0, label='Bar Order within Year:'),
        Line2D([0], [0], color='gray', lw=0, label='Left: 50% Prob'),
        Line2D([0], [0], color='gray', lw=0, label='Right: 95%+ Prob')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()

# 執行
if 'df_Exe' in locals() and not df_Exe.empty:
    analyze_year_bucket_distribution(df_Exe)
else:
    print("df_Exe not found.")



# %% [13] Visualization: Net Return Distribution

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("🚀 繪製分佈圖 (直接鎖定 'Net_Ret')...")

# 1. 資料清洗與截斷 (直接使用 Net_Ret)
# ------------------------------------------------
# 移除無限大與空值
clean_data = df_Inv['Net_Ret'].replace([np.inf, -np.inf], np.nan).dropna()

# 強制截斷：小於 -1.0 的都視為 -1.0 (為了堆疊在最左邊)
# 大於 3.0 的視為 3.0 (避免極端值拉長圖表)
plot_data = clean_data.clip(lower=-1.0, upper=3.0)

# 統計數據
median_ret = clean_data.median()
win_rate = (clean_data > 0).mean()

# 2. 繪圖設定
# ------------------------------------------------
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# (A) 直方圖
# binrange=(-1.0, 3.0): 強制第一個 bin 準確從 -1.0 開始
ax = sns.histplot(plot_data, bins=40, binrange=(-1.0, 3.0), kde=False, 
                  color='teal', edgecolor='white', alpha=0.85)

# (B) 密度曲線 (KDE)
if len(plot_data) > 10:
    try:
        sns.kdeplot(plot_data, color='darkslategray', linewidth=1.5, ax=ax, cut=0)
    except:
        pass

# (C) 關鍵修正：強制設定 X 軸刻度與標籤
# ------------------------------------------------
# 1. 設定 X 軸範圍 (左邊多留一點空隙，讓 -1.0 的 bar 不會被切掉)
ax.set_xlim(left=-1.15, right=3.1)

# 2. 強制指定刻度位置 (必須包含 -1.0)
custom_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ax.set_xticks(custom_ticks)

# 3. 建立自訂標籤 (處理 <-100%)
custom_labels = []
for t in custom_ticks:
    if t == -1.0:
        custom_labels.append("<-100%") 
    else:
        custom_labels.append(f"{t:.0%}")
        
# 4. 應用標籤並加大字體
ax.set_xticklabels(custom_labels, fontweight='bold', fontsize=14)

# (D) 參考線與標註
# ------------------------------------------------
plt.axvline(0, color='black', linewidth=1.5, linestyle='-', label='Break-even')
plt.axvline(median_ret, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_ret:.2%}')

# 統計框
stats_text = (f"Total Stocks: {len(clean_data)}\n"
              f"Median: {median_ret:.2%}\n"
              f"Win Rate: {win_rate:.2%}\n"
              f"(Bankruptcy clipped at -100%)")

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
plt.gca().text(0.98, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', horizontalalignment='right', bbox=props)

# (E) 加大字體與標題
plt.title('Distribution of Net Returns (Net_Ret)', fontsize=20, fontweight='bold', pad=15)
plt.xlabel('Net Return', fontsize=16, fontweight='bold')
plt.ylabel('Frequency (Stock Count)', fontsize=16, fontweight='bold')
plt.tick_params(axis='y', labelsize=14)

plt.legend(loc='upper right', fontsize=14)
plt.tight_layout()
plt.show()

print("✅ 繪圖完成")
# clean_data = clean_data[clean_data.abs() > 1e-6]


# %% [14] Sharpe, Win Rate, and Max Drawdown


def compute_portfolio_metrics(df_exe, total_stocks_count=746, risk_free_rate=0.0):
    print("🚀 計算等權重投資組合績效指標 (Win Rate, Sharpe, MDD)...")
    
    if df_exe is None or df_exe.empty:
        print("Error: df_Exe is empty.")
        return

    # ==========================================
    # 1. 建立投資組合每日報酬率序列
    # ==========================================
    df = df_exe[['Date', 'Code', 'Net_Cum_Ret', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 排序
    df = df.sort_values(['Code', 'Date'])
    
    # 還原每日損益 (Daily PnL in Points)
    df['Daily_PnL'] = df.groupby('Code')['Net_Cum_Ret'].diff().fillna(0)
    
    # 修正第一筆交易 (diff 會是 0，需補回)
    mask = (df['Net_Cum_Ret'] != 0) & (df['Daily_PnL'] == 0)
    df.loc[mask, 'Daily_PnL'] = df.loc[mask, 'Net_Cum_Ret']
    
    # 轉為百分比貢獻 (Contribution %)
    # 公式: (當日賺的點數 / 股價)
    df['Daily_Contrib_Pct'] = df['Daily_PnL'] / df['Close']
    
    # 聚合：算出「投資組合」每一天的總報酬率
    # 這裡除以 total_stocks_count (746) 是等權重的關鍵
    portfolio_daily_ret = df.groupby('Date')['Daily_Contrib_Pct'].sum() / total_stocks_count
    
    # ==========================================
    # 2. 計算績效指標 (Metrics)
    # ==========================================
    
    # --- A. Win Rate (日勝率) ---
    # 統計報酬率 > 0 的天數佔比
    winning_days = (portfolio_daily_ret > 0).sum()
    total_days = len(portfolio_daily_ret)
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # --- B. Sharpe Ratio (夏普值) ---
    # 年化係數通常設為 252 (交易日)
    mean_ret = portfolio_daily_ret.mean()
    std_ret = portfolio_daily_ret.std()
    
    if std_ret == 0:
        sharpe_ratio = 0
    else:
        # (平均日報酬 - 無風險利率) / 日波動率 * sqrt(252)
        # 這裡假設 risk_free_rate 為年化，需轉為日化 (或直接忽略，視為 0)
        daily_rf = risk_free_rate / 252
        sharpe_ratio = ((mean_ret - daily_rf) / std_ret) * np.sqrt(252)
        
    # --- C. Maximum Drawdown (最大回落) ---
    # 1. 計算累積報酬曲線 (Cumulative Return)
    cum_ret = portfolio_daily_ret.cumsum()
    # 2. 計算歷史最高點 (Running Max)
    running_max = cum_ret.cummax()
    # 3. 計算回落 (Drawdown)
    drawdown = cum_ret - running_max
    # 4. 取最小值 (最深的回落)
    max_drawdown = drawdown.min()
    
    # --- D. 其他輔助指標 ---
    total_return = cum_ret.iloc[-1]
    annualized_return = total_return * (252 / total_days) # 簡單估算
    
    # ==========================================
    # 3. 輸出結果
    # ==========================================
    print("-" * 40)
    print(f"📊 Portfolio Performance Metrics (Equal Weight, N={total_stocks_count})")
    print("-" * 40)
    print(f"Daily Win Rate:      {win_rate:.2%}")
    print(f"Sharpe Ratio:        {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown:    {max_drawdown:.2%}")
    print("-" * 40)
    print(f"Total Return:        {total_return:.2%}")
    print(f"Daily Volatility:    {std_ret:.2%}")
    print("-" * 40)
    
    return {
        'Win_Rate': win_rate,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Total_Return': total_return,
        'Portfolio_Daily_Returns': portfolio_daily_ret
    }

# 執行計算
metrics = compute_portfolio_metrics(df_Exe, total_stocks_count=746)




# %% [15] Execution of Strategy

# df_Exe = pd.read_parquet("Transaction_Ledger_Constant.parquet")
# df_Inv = pd.read_csv("Strategy_Performance_XGB.csv")

def plot_stock_execution(df_exe, stock_code):
    """
    繪製特定股票的策略執行圖 (雙 Y 軸)。
    
    Args:
        df_exe (pd.DataFrame): 包含交易紀錄的 DataFrame (必須包含 Date, Close, Net_Cum_Ret, Exec_Sig)
        stock_code (str or int): 要繪製的股票代碼
    """
    # 1. 資料過濾
    # 確保代碼格式一致 (轉成字串比較最安全)
    df_plot = df_exe[df_exe['Code'].astype(str) == str(stock_code)].copy()
    
    if df_plot.empty:
        print(f"Error: No data found for stock code {stock_code}")
        return
    
    # 確保日期格式
    if not pd.api.types.is_datetime64_any_dtype(df_plot['Date']):
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])
        
    df_plot = df_plot.sort_values('Date')
    
    # 2. 設定畫布與雙軸
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # --- 左軸：股價 (Price) ---
    ax1.set_xlabel('Date', fontsize=18)
    ax1.set_ylabel('Stock Price', color='black', fontsize=18)
    # 使用灰色線條顯示股價，作為背景參考
    ax1.plot(df_plot['Date'], df_plot['Close'], color='black', alpha=1, linewidth=1, label='Close Price')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    
    # --- 右軸：累積報酬率 (Cumulative Return) ---
    ax2 = ax1.twinx()  # 建立共享 X 軸的第二個 Y 軸
    ax2.set_ylabel('Cumulative Net Return', color='blue', fontsize=18)
    # 使用藍色實線顯示績效
    ax2.plot(df_plot['Date'], df_plot['Net_Cum_Ret'], color='royalblue', linewidth=3, label='Net Cum Ret', alpha=1)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=15)
    
    # 3. 標記進出場點 (標記在右軸的報酬率曲線上)
    # Exec_Sig 定義: 1=Long Entry, -1=Short Entry, 2=Exit
    
    # (A) 做多進場 (Long Entry) - 綠色向上三角形
    long_entry = df_plot[df_plot['Exec_Sig'] == 1]
    ax2.scatter(long_entry['Date'], long_entry['Net_Cum_Ret'], 
                color='green', marker='^', s=100, zorder=5, label='Long Entry', alpha=0.5)

    # (B) 做空進場 (Short Entry) - 紅色向下三角形
    short_entry = df_plot[df_plot['Exec_Sig'] == -1]
    ax2.scatter(short_entry['Date'], short_entry['Net_Cum_Ret'], 
                color='red', marker='v', s=100, zorder=5, label='Short Entry', alpha=0.5)

    # (C) 出場 (Exit) - 黑色 X
    # 注意：這裡假設 Exec_Sig == 2 代表平倉
    exit_points = df_plot[df_plot['Exec_Sig'] == 2]
    ax2.scatter(exit_points['Date'], exit_points['Net_Cum_Ret'], 
                color='black', marker='X', s=80, zorder=5, label='Exit', alpha=0.9)

    # 4. 格式美化
    plt.title(f'Strategy Execution & Performance: {stock_code}', fontsize=20, fontweight='bold')
    
    # 合併兩個軸的 Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # 這裡只顯示右軸 (報酬率與交易點) 的圖例會比較清楚，因為左軸只是背景
    ax2.legend(lines_2, labels_2, loc='upper left', frameon=True, fancybox=True, framealpha=0.9, fontsize=18)
    
    # ==========================================
    # 🔥🔥🔥 關鍵修改：對齊零軸 (Align Zeros) 🔥🔥🔥
    # ==========================================
    
    # 1. 取得目前右軸 (報酬率) 的範圍
    y2_min, y2_max = ax2.get_ylim()
    
    # 2. 計算右軸的「上方比例」與「下方比例」
    #    報酬率通常有正有負，0 在中間
    up2 = max(y2_max, 0)
    down2 = max(-y2_min, 0) # 取絕對值
    
    # 防呆：如果右軸全是正的 (down2=0)，或者全是負的 (up2=0)
    if up2 == 0: up2 = 0.01 # 避免除以零
    
    # 算出比例 ratio = 下方長度 / 上方長度
    ratio = down2 / up2
    
    # 3. 取得左軸 (股價) 的範圍
    #    股價通常都是正的，所以 0 在最下面
    y1_min, y1_max = ax1.get_ylim()
    up1 = max(y1_max, 0) # 股價上方空間 (就是最高價)
    
    # 4. 強制設定左軸的下方空間，使其比例與右軸一致
    #    new_down1 / up1 = down2 / up2  =>  new_down1 = up1 * ratio
    new_down1 = up1 * ratio
    
    # 5. 設定新的左軸範圍
    #    這樣左軸的 0 就會被「推」到跟右軸 0 一樣的高度
    ax1.set_ylim(-new_down1, up1)
    
    # 畫一條水平零線作為參考
    ax2.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)

    # ==========================================
    
    
    # 設定 X 軸日期格式
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # 每3個月顯示一次
    # 🔥🔥🔥 關鍵修改：X 軸只顯示年份 🔥🔥🔥
    # 設定格式為 %Y (只有年份)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # 設定刻度為「每年」顯示一次 (避免太多重複的年份)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    plt.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[0]['Code'])
# plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[1]['Code'])
plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[2]['Code'])
plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[3]['Code'])
plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[4]['Code'])
plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[5]['Code'])
plot_stock_execution(df_Exe, best_stock)




best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[-1]['Code'])
plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[-2]['Code'])
plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[-3]['Code'])
plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[-4]['Code'])
plot_stock_execution(df_Exe, best_stock)

best_stock = int(df_Inv.sort_values('Net_Ret', ascending=False).iloc[-5]['Code'])
plot_stock_execution(df_Exe, best_stock)



# %% Win Rates

(df_Inv['Net_Ret'] > 0).sum() - 2


(df_Inv['Net_Ret'] < 0).sum()


(df_Inv['Net_Ret'] == 0).sum()



268 + 183 + 295







