import logging
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier



# Suppress warnings for clean terminal output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from nsepython import nse_eq_symbols
from niftystocks import ns

tickers = ns.get_nifty50_with_ns()  # returns list like ['RELIANCE.NS', 'TCS.NS', ...]
print(tickers)
# =====================================================================
# CONSTANTS & UTILITIES
# =====================================================================
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "HCLTECH.NS",
    "BAJFINANCE.NS", "SUNPHARMA.NS", "M&M.NS", "ADANIENT.NS",
    "NTPC.NS", "KOTAKBANK.NS", "AXISBANK.NS", "ONGC.NS", "TITAN.NS",
    "MARUTI.NS", "POWERGRID.NS", "ULTRACEMCO.NS", "COALINDIA.NS", "ADANIPORTS.NS",
    "BAJAJFINSV.NS", "BPCL.NS", "WIPRO.NS", "ASIANPAINT.NS", "JIOFIN.NS",
    "HINDALCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "LTI.NS", "GRASIM.NS",
    "NESTLEIND.NS", "TECHM.NS", "HDFCLIFE.NS", "SBILIFE.NS", "CIPLA.NS",
    "EICHERMOT.NS", "TATACONSUM.NS", "BRITANNIA.NS", "INDUSINDBK.NS", "DRREDDY.NS",
    "DIVISLAB.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS"
]
NIFTY50_TICKERS=tickers

SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Financial Services",
    "BHARTIARTL.NS": "Telecom", "ICICIBANK.NS": "Financial Services", "INFY.NS": "IT",
    "SBIN.NS": "Financial Services", "ITC.NS": "FMCG", "LT.NS": "Construction",
    "HCLTECH.NS": "IT", "BAJFINANCE.NS": "Financial Services", "SUNPHARMA.NS": "Healthcare"
}

# =====================================================================
# DATA LOADERS
# =====================================================================
class DataLoader:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def fetch_stock_data(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """Download historical OHLCV data from Yahoo Finance."""
        logging.info(f"Starting data fetch for {len(tickers)} tickers over {period}...")
        data_store = {}

        try:
            df_batch = yf.download(tickers, period=period, interval="1d", group_by='ticker', progress=False)

            for ticker in tickers:
                if ticker in df_batch.columns.levels[0]:
                    df = df_batch[ticker].dropna(subset=['Close'])
                    if not df.empty:
                        df = df.copy()
                        df.columns = [str(col).capitalize() for col in df.columns]
                        data_store[ticker] = df
        except Exception as e:
            logging.error(f"Error downloading batch market datasets: {e}")

        logging.info(f"Successfully loaded data for {len(data_store)} stocks.")
        return data_store

    @staticmethod
    @st.cache_data(show_spinner=False)
    def fetch_benchmark(period: str = "2y") -> pd.DataFrame:
        """Download the NIFTY benchmark and normalize labels."""
        try:
            df = yf.download("^NSEI", period=period, interval="1d", progress=False, multi_level_index=False)
            df = df.copy()
            df.columns = [str(col).split('_')[-1].capitalize() if isinstance(col, tuple) else str(col).capitalize() for col in df.columns]
            return df
        except Exception as e:
            logging.error(f"Failed to fetch NIFTY index data: {e}")
            return pd.DataFrame()


# =====================================================================
# INDICATOR ENGINE
# =====================================================================
class IndicatorEngine:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Append indicator series and candlestick pattern signals.
        Tuned for swing trading (1 day – 2 weeks hold, cash-and-carry only).
        """
        df = df.copy()
        if len(df) < 200:
            return pd.DataFrame()

        try:
            close = df['Close']
            high  = df['High']
            low   = df['Low']
            open_ = df['Open']
            vol   = df['Volume']

            # ── Trend ──────────────────────────────────────────────────────
            df['EMA_20']  = ta.ema(close, length=20)
            df['EMA_50']  = ta.ema(close, length=50)
            df['EMA_200'] = ta.ema(close, length=200)
            df['SMA_50']  = ta.sma(close, length=50)
            df['SMA_200'] = ta.sma(close, length=200)

            # Regime: price above both EMAs = bull, below both = bear
            df['Trend_Bull'] = (
                (close > df['EMA_20']) &
                (close > df['EMA_50']) &
                (df['EMA_20'] > df['EMA_50'])
            ).astype(int)

            # Supertrend — primary swing entry/exit signal
            supertrend = ta.supertrend(high, low, close, length=10, multiplier=3)
            if supertrend is not None:
                df['Supertrend']     = supertrend.get('SUPERT_10_3.0', np.nan)
                df['Supertrend_Dir'] = supertrend.get('SUPERTd_10_3.0', np.nan)  # 1 = bullish, -1 = bearish
            else:
                df['Supertrend'] = df['Supertrend_Dir'] = np.nan

            # ── Momentum ───────────────────────────────────────────────────
            macd = ta.macd(close, fast=12, slow=26, signal=9)
            if macd is not None:
                df['MACD']      = macd.get('MACD_12_26_9', np.nan)
                df['MACD_Sig']  = macd.get('MACDs_12_26_9', np.nan)
                df['MACD_Hist'] = macd.get('MACDh_12_26_9', np.nan)
                # Histogram turning positive = early momentum shift
                df['MACD_Cross_Up'] = (
                    (df['MACD_Hist'] > 0) & (df['MACD_Hist'].shift(1) <= 0)
                ).astype(int)
            else:
                df['MACD'] = df['MACD_Sig'] = df['MACD_Hist'] = np.nan
                df['MACD_Cross_Up'] = 0

            df['RSI_14'] = ta.rsi(close, length=14)
            # RSI coming out of oversold (30–50 zone) is the sweet spot for swing entry
            df['RSI_Swing_Zone'] = (
                (df['RSI_14'] > 40) & (df['RSI_14'] < 65)
            ).astype(int)

            # Rate of Change — short-term momentum burst detection
            df['ROC_5']  = ta.roc(close, length=5)   # 1-week momentum
            df['ROC_10'] = ta.roc(close, length=10)  # 2-week momentum

            # Williams %R — faster overbought/oversold than Stochastic for short holds
            willr = ta.willr(high, low, close, length=14)
            df['WillR'] = willr if willr is not None else np.nan
            # -80 to -20 is the actionable swing range
            df['WillR_Oversold']  = (df['WillR'] < -80).astype(int)
            df['WillR_Recovering'] = (
                (df['WillR'] > -80) & (df['WillR'].shift(1) <= -80)
            ).astype(int)

            stoch = ta.stoch(high, low, close)
            if stoch is not None:
                df['Stoch_K'] = stoch.get('STOCHk_14_3_3', np.nan)
                df['Stoch_D'] = stoch.get('STOCHd_14_3_3', np.nan)
                df['Stoch_Cross_Up'] = (
                    (df['Stoch_K'] > df['Stoch_D']) &
                    (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)) &
                    (df['Stoch_K'] < 50)  # only meaningful below midline
                ).astype(int)
            else:
                df['Stoch_K'] = df['Stoch_D'] = np.nan
                df['Stoch_Cross_Up'] = 0

            df['CCI'] = ta.cci(high, low, close, length=20)
            df['OBV'] = ta.obv(close, vol)

            # ── Trend Strength ─────────────────────────────────────────────
            adx = ta.adx(high, low, close, length=14)
            if adx is not None:
                df['ADX'] = adx.get('ADX_14', np.nan)
                df['DMP'] = adx.get('DMP_14', np.nan)
                df['DMN'] = adx.get('DMN_14', np.nan)
                # ADX > 20 with +DI > -DI = trending up, worth swinging
                df['ADX_Trending'] = (
                    (df['ADX'] > 20) & (df['DMP'] > df['DMN'])
                ).astype(int)
            else:
                df['ADX'] = df['DMP'] = df['DMN'] = np.nan
                df['ADX_Trending'] = 0

            # ── Volatility & Bands ─────────────────────────────────────────
            df['ATR'] = ta.atr(high, low, close, length=14)

            bbands = ta.bbands(close, length=20, std=2)
            if bbands is not None:
                df['BB_Upper']     = bbands.get('BBU_20_2.0', np.nan)
                df['BB_Lower']     = bbands.get('BBL_20_2.0', np.nan)
                df['BB_Middle']    = bbands.get('BBM_20_2.0', np.nan)
                df['BB_Width']     = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
                df['BB_Pct']       = bbands.get('BBP_20_2.0', np.nan)  # 0=lower band, 1=upper band
            else:
                df['BB_Upper'] = df['BB_Lower'] = df['BB_Middle'] = np.nan
                df['BB_Width'] = df['BB_Pct'] = np.nan

            # Keltner Channel for squeeze detection
            kc = ta.kc(high, low, close, length=20, scalar=1.5)
            if kc is not None:
                df['KC_Upper'] = kc.get('KCUe_20_1.5', np.nan)
                df['KC_Lower'] = kc.get('KCLe_20_1.5', np.nan)
                # Squeeze: BB inside KC = compressed volatility, explosive move coming
                df['Squeeze'] = (
                    (df['BB_Upper'] < df['KC_Upper']) &
                    (df['BB_Lower'] > df['KC_Lower'])
                ).astype(int)
                # Squeeze release = volatility expanding after compression
                df['Squeeze_Release'] = (
                    (df['Squeeze'] == 0) & (df['Squeeze'].shift(1) == 1)
                ).astype(int)
            else:
                df['KC_Upper'] = df['KC_Lower'] = np.nan
                df['Squeeze'] = df['Squeeze_Release'] = 0

            # ── Volume ─────────────────────────────────────────────────────
            df['Vol_MA']    = ta.sma(vol, length=20)
            df['VWAP']      = ta.vwap(high, low, close, vol)
            df['Vol_Ratio'] = vol / df['Vol_MA']  # > 1.5 = above-average volume
            # Volume surge on an up day = institutional accumulation signal
            df['Vol_Surge_Up'] = (
                (df['Vol_Ratio'] > 1.5) & (close > open_)
            ).astype(int)

            # ── Support / Resistance / Breakout ────────────────────────────
            df['Support']    = low.rolling(window=20).min()
            df['Resistance'] = high.rolling(window=20).max()
            # Confirmed breakout: close above resistance + volume surge
            df['Breakout'] = (
                (close > df['Resistance'].shift(1)) &
                (df['Vol_Ratio'] > 1.2)
            ).astype(int)

            # Pivot Points (standard floor trader pivots from previous day)
            df['Pivot']  = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
            df['PP_R1']  = 2 * df['Pivot'] - low.shift(1)
            df['PP_S1']  = 2 * df['Pivot'] - high.shift(1)
            df['PP_R2']  = df['Pivot'] + (high.shift(1) - low.shift(1))
            df['PP_S2']  = df['Pivot'] - (high.shift(1) - low.shift(1))

            # ── ATR-based Risk Levels ───────────────────────────────────────
            # For each bar: suggested stop (1.5×ATR below close) and target (3×ATR above)
            # Risk:Reward = 1:2 built in
            df['Stop_Loss']  = close - (1.5 * df['ATR'])
            df['Target_1']   = close + (2.0 * df['ATR'])
            df['Target_2']   = close + (3.5 * df['ATR'])

            # ── Composite Swing Score (0–6) ─────────────────────────────────
            # A quick at-a-glance score; use to rank/filter, not as a hard signal
            df['Swing_Score'] = (
                df['Trend_Bull'] +          # price in uptrend
                df['ADX_Trending'] +        # trend has strength
                df['MACD_Cross_Up'] +       # momentum just turned
                df['RSI_Swing_Zone'] +      # RSI in healthy entry zone
                df['Vol_Surge_Up'] +        # volume confirms
                df['Squeeze_Release']       # volatility just expanded
            )

            # ── Candlestick Patterns ────────────────────────────────────────
            for pattern, col in [
                ("engulfing", "CDL_Engulfing"),
                ("hammer",    "CDL_Hammer"),
                ("doji",      "CDL_Doji"),
                ("morningstar", "CDL_MorningStar"),
                ("shootingstar", "CDL_ShootingStar"),  # tells you what to AVOID
            ]:
                result = ta.cdl_pattern(open_, high, low, close, name=pattern)
                df[col] = result.iloc[:, 0] if result is not None and not result.empty else 0

            # ── Extension / Overbought Guardrails ──────────────────────────────
            # How far is price stretched above its own trend, in ATR units?
            df['Ext_From_EMA20_ATR'] = (close - df['EMA_20']) / df['ATR']

            # Recent short-term move — used to avoid chasing a stock that already popped
            df['Ret_1D'] = close.pct_change(1)
            df['Ret_3D'] = close.pct_change(3)

            # Flag stocks that are too extended to be a fresh swing entry
            df['Overextended'] = (
                (df['Ext_From_EMA20_ATR'] > 2.5) |   # > 2.5 ATR above EMA20
                (df['Ret_3D'] > 0.08)                 # already up >8% in 3 sessions
            ).astype(int)

            # RSI should be RISING into the zone, not already sitting high for days
            df['RSI_Rising_Into_Zone'] = (
                (df['RSI_14'] > 40) & (df['RSI_14'] < 60) &
                (df['RSI_14'] > df['RSI_14'].shift(3))  # momentum building, not stalling
            ).astype(int)

            # Distinguish accumulation volume from a climax/blow-off day
            df['Candle_Range'] = high - low
            df['Range_Pct_of_ATR'] = df['Candle_Range'] / df['ATR']
            df['Close_Near_High'] = (close - low) / (df['Candle_Range'].replace(0, np.nan))

            df['Climax_Day'] = (
                (df['Vol_Ratio'] > 2.0) &
                (df['Range_Pct_of_ATR'] > 2.0) &
                (df['Close_Near_High'] > 0.85)
            ).astype(int)  # exhaustion signature — exclude, don't reward

            # Healthy volume confirmation = above-average but NOT a climax
            df['Vol_Surge_Up'] = (
                (df['Vol_Ratio'] > 1.3) & (df['Vol_Ratio'] < 2.0) &
                (close > open_) & (df['Climax_Day'] == 0)
            ).astype(int)

            # Risk:Reward gate — only score setups with at least 1:1.5 built in
            df['RR_OK'] = (
                (df['Target_1'] - close) / (close - df['Stop_Loss']) > 1.5
            ).astype(int)

        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return pd.DataFrame()

        return df

# =====================================================================
# SCORING ENGINE
# =====================================================================
class ScoringEngine:
    @staticmethod
    def score_stock(df: pd.DataFrame) -> Tuple[float, float, float]:
        """Score the latest bar using trend, momentum, volume, volatility and patterns.
        Penalizes overextended/climax conditions instead of rewarding them —
        this engine is tuned to find fresh swing entries, not stocks that already ran.
        """
        if df.empty or 'EMA_200' not in df.columns or pd.isna(df.iloc[-1]['EMA_200']):
            return 0.0, 0.5, 1.0

        row = df.iloc[-1]

        def safe_value(key, default=np.nan):
            return row[key] if key in row else default

        # ── Trend ──────────────────────────────────────────────────────────
        t_score = 0
        if row['Close'] > row['EMA_20'] > row['EMA_50'] > row['EMA_200']:
            t_score += 2
        elif row['Close'] > row['EMA_200']:
            t_score += 1
        elif row['Close'] < row['EMA_20'] < row['EMA_50'] < row['EMA_200']:
            t_score -= 2
        elif row['Close'] < row['EMA_200']:
            t_score -= 1

        if not pd.isna(safe_value('MACD')) and not pd.isna(safe_value('MACD_Sig')) and row['MACD'] > row['MACD_Sig']:
            t_score += 1
        else:
            t_score -= 1

        if not pd.isna(safe_value('ADX')) and row['ADX'] > 25 and safe_value('DMP') > safe_value('DMN'):
            t_score += 1
        elif not pd.isna(safe_value('ADX')) and row['ADX'] > 25 and safe_value('DMN') > safe_value('DMP'):
            t_score -= 1

        trend_final = np.clip(t_score / 2, -2, 2)

        # ── Momentum (FIXED: overbought no longer scores bullish) ──────────
        m_score = 0
        if not pd.isna(safe_value('RSI_14')):
            rsi = row['RSI_14']
            if 45 <= rsi <= 60:
                m_score += 2          # sweet spot — room to run, not stretched
            elif 40 <= rsi < 45 or 60 < rsi <= 68:
                m_score += 1          # acceptable but less ideal
            elif rsi > 75:
                m_score -= 2          # genuinely overbought — penalize, don't chase
            elif rsi < 30:
                m_score -= 2          # oversold/falling knife risk for swing longs

        # Reward RSI that's rising into the zone (fresh momentum), not stalled there
        if safe_value('RSI_Rising_Into_Zone') == 1:
            m_score += 1

        if not pd.isna(safe_value('Stoch_K')) and not pd.isna(safe_value('Stoch_D')):
            if row['Stoch_K'] > row['Stoch_D'] and row['Stoch_K'] < 80:
                m_score += 1
            elif row['Stoch_K'] >= 80:
                m_score -= 1          # stochastic already pinned high
            else:
                m_score -= 1

        if not pd.isna(safe_value('CCI')):
            if 0 < row['CCI'] <= 100:
                m_score += 1          # building momentum, not extreme
            elif row['CCI'] > 150:
                m_score -= 1          # extreme reading = exhaustion risk, not strength
            elif row['CCI'] < -100:
                m_score -= 1

        momentum_final = np.clip(m_score / 2, -2, 2)

        # ── Volume ────────────────────────────────────────────────────────
        v_score = 0
        if not pd.isna(safe_value('Vol_Surge_Up')) and row['Vol_Surge_Up'] == 1:
            v_score += 1               # healthy confirmation volume (already excludes climax)
        elif not pd.isna(safe_value('Volume')) and not pd.isna(safe_value('Vol_MA')) and row['Volume'] > row['Vol_MA']:
            v_score += 0.5

        if not pd.isna(safe_value('VWAP')) and row['Close'] > row['VWAP']:
            v_score += 1
        else:
            v_score -= 1

        if safe_value('Climax_Day') == 1:
            v_score -= 2                # blow-off volume = exhaustion, not strength

        volume_final = np.clip(v_score, -2, 2)

        # ── Volatility / Bands ───────────────────────────────────────────
        vol_score = 0
        if not pd.isna(safe_value('BB_Upper')) and row['Close'] > row['BB_Upper']:
            vol_score -= 1               # riding outside upper band = stretched, not a buy signal
        elif not pd.isna(safe_value('BB_Lower')) and row['Close'] < row['BB_Lower']:
            vol_score -= 1
        elif not pd.isna(safe_value('BB_Pct')) and 0.3 <= row['BB_Pct'] <= 0.75:
            vol_score += 1                # comfortably inside the band, room to expand
        vol_final = np.clip(vol_score, -2, 2)

        # ── Patterns / Breakout (quality-gated, not flat reward) ───────────
        p_score = 0
        if safe_value('Breakout') == 1 and safe_value('Overextended') == 0:
            p_score += 2                  # breakout only counts if not already extended
        elif safe_value('Breakout') == 1:
            p_score += 0                  # extended breakout = no credit, likely late entry

        if safe_value('CDL_Engulfing') > 0:
            p_score += 1
        elif safe_value('CDL_Engulfing') < 0:
            p_score -= 1
        if safe_value('CDL_Hammer') > 0:
            p_score += 1
        if safe_value('CDL_ShootingStar') > 0:
            p_score -= 1                  # bearish reversal candle near highs
        p_final = np.clip(p_score, -2, 2)

        # ── Combine ──────────────────────────────────────────────────────
        weights = {'trend': 0.40, 'momentum': 0.30, 'volume': 0.15, 'volatility': 0.10, 'patterns': 0.05}
        final_score = (
            (trend_final * weights['trend']) +
            (momentum_final * weights['momentum']) +
            (volume_final * weights['volume']) +
            (vol_final * weights['volatility']) +
            (p_final * weights['patterns'])
        )

        # ── Hard guardrails: these override raw score, same intent as
        #    IndicatorEngine.Overextended / Climax_Day / RR_OK ─────────────
        overextended = safe_value('Overextended', 0) == 1
        climax = safe_value('Climax_Day', 0) == 1
        rr_ok = safe_value('RR_OK', 1) == 1   # default to OK if column missing

        if overextended:
            final_score -= 1.0            # direct penalty, stacks with momentum/vol penalties above
        if climax:
            final_score -= 1.0
        if not rr_ok:
            final_score -= 0.5            # poor risk:reward shouldn't score as a clean setup

        final_score = float(np.clip(final_score, -2, 2))

        bullish_probability = (final_score + 2) / 4
        # Extension also tightens probability — even if score is decent, an extended
        # stock has a thinner margin before mean reversion kicks in
        if overextended or climax:
            bullish_probability = max(0.0, bullish_probability - 0.15)

        risk_score = (
            np.clip(row['ATR'] / row['Close'] * 100, 0.5, 10.0)
            if not pd.isna(safe_value('ATR')) and row['ATR'] > 0 else 1.0
        )
        # Bump risk_score for extended/climax setups since downside risk is understated
        # by ATR alone in these cases (mean-reversion risk isn't captured by ATR)
        if overextended:
            risk_score = min(10.0, risk_score * 1.3)
        if climax:
            risk_score = min(10.0, risk_score * 1.5)

        return float(final_score), float(bullish_probability), float(risk_score)
# =====================================================================
# STRATEGY TRADE LEVELS ENGINE
# =====================================================================
class StrategyEngine:
    @staticmethod
    def calculate_levels(price: float, atr: float) -> Dict[str, float]:
        """Generate tactical execution levels from price and ATR."""
        if pd.isna(atr) or atr <= 0:
            atr = price * 0.01

        entry = price
        stop_loss = price - (1.5 * atr)
        target_1 = price + (1.5 * atr)
        target_2 = price + (3.0 * atr)
        rr_ratio = (target_1 - entry) / (entry - stop_loss) if (entry - stop_loss) > 0 else 0

        return {
            "Entry": round(entry, 2),
            "StopLoss": round(stop_loss, 2),
            "Target1": round(target_1, 2),
            "Target2": round(target_2, 2),
            "RiskReward": round(rr_ratio, 2)
        }


# =====================================================================
# MACHINE LEARNING FILTER
# =====================================================================
class MLFilter:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
        self.is_trained = False
        self.feature_cols = ['RSI_14', 'CCI', 'ADX', 'DMP', 'DMN', 'Breakout']

    def prepare_data(self, data_store: Dict[str, pd.DataFrame], df_nifty: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X_list, y_list = [], []
        if df_nifty.empty:
            return pd.DataFrame(), pd.Series(dtype='int')

        df_nifty = df_nifty.copy()
        nifty_close = df_nifty['Close'].squeeze()
        nifty_fwd = nifty_close.shift(-10) / nifty_close - 1
        nifty_fwd_df = pd.DataFrame({'Nifty_Fwd_10': nifty_fwd}, index=df_nifty.index)

        for ticker, df in data_store.items():
            if df.empty or len(df) < 50:
                continue
            df = df.copy()
            stock_close = df['Close'].squeeze()
            df['Stock_Fwd_10'] = stock_close.shift(-10) / stock_close - 1
            df = df.join(nifty_fwd_df, how='inner')
            df['Target'] = (df['Stock_Fwd_10'] > df['Nifty_Fwd_10']).astype(int)
            df_clean = df.dropna(subset=self.feature_cols + ['Target'])
            if not df_clean.empty:
                X_list.append(df_clean[self.feature_cols])
                y_list.append(df_clean['Target'])

        if X_list and y_list:
            return pd.concat(X_list, axis=0), pd.concat(y_list, axis=0)
        return pd.DataFrame(), pd.Series(dtype='int')

    def train(self, data_store: Dict[str, pd.DataFrame], df_nifty: pd.DataFrame):
        X, y = self.prepare_data(data_store, df_nifty)
        if isinstance(X, pd.DataFrame) and not X.empty and len(np.unique(y)) > 1:
            logging.info(f"Training Machine Learning alpha filters on shape matrix: {X.shape}")
            self.model.fit(X, y)
            self.is_trained = True
        else:
            logging.warning("Insufficient data vectors to complete structural ML Training.")

    def predict_probability(self, df: pd.DataFrame) -> float:
        if not self.is_trained or df.empty:
            return 0.50
        last_row = df.dropna(subset=self.feature_cols).iloc[-1:]
        if last_row.empty:
            return 0.50
        features = last_row[self.feature_cols]
        prob = self.model.predict_proba(features)[0][1]
        return float(prob)


# =====================================================================
# STREAMLIT APPLICATION
# =====================================================================
st.set_page_config(page_title="NIFTY Swing Scanner", layout="wide")
st.title("NSE Swing Scanner")

with st.sidebar:
    st.header("Scanner Settings")
    selected_tickers = st.multiselect("Select tickers to scan", NIFTY50_TICKERS, default=NIFTY50_TICKERS)
    selected_period = st.selectbox("History period", ["1y", "2y", "3y"], index=1)
    min_bars = st.slider("Minimum bars to process", min_value=150, max_value=300, value=200, step=10)
    score_threshold = st.slider("Minimum score", min_value=-2.0, max_value=2.0, value=0.1, step=0.1)
    show_details = st.checkbox("Show ticker details", value=True)
    run_scan = st.button("Run swing scan")

if not selected_tickers:
    st.warning("Please select one or more tickers to scan.")
else:
    if run_scan:
        with st.spinner("Downloading data and running the scanner..."):
            data_store = DataLoader.fetch_stock_data(selected_tickers, period=selected_period)
            df_nifty = DataLoader.fetch_benchmark(period=selected_period)

            processed_store: Dict[str, pd.DataFrame] = {}
            for ticker, df in data_store.items():
                processed_df = IndicatorEngine.calculate_all(df)
                if not processed_df.empty and len(processed_df) >= min_bars:
                    processed_store[ticker] = processed_df

            ml_filter = MLFilter()
            ml_filter.train(processed_store, df_nifty)

            records = []
            for ticker, df in processed_store.items():
                score, prob, risk = ScoringEngine.score_stock(df)
                ml_prob = ml_filter.predict_probability(df)
                blended_prob = (prob * 0.70) + (ml_prob * 0.30)

                last_price = float(df.iloc[-1]['Close'])
                last_atr = float(df.iloc[-1]['ATR']) if 'ATR' in df.columns else 0.0
                levels = StrategyEngine.calculate_levels(last_price, last_atr)
                sector = SECTOR_MAP.get(ticker, "NIFTY Index Component")

                records.append({
                    "Ticker": ticker,
                    "Sector": sector,
                    "Price": round(last_price, 2),
                    "Score": round(score, 4),
                    "Bullish_Prob": round(blended_prob * 100, 2),
                    "Risk_Score": round(risk, 2),
                    **levels,
                })

            ranking_df = pd.DataFrame(records)
            if ranking_df.empty:
                st.error("No valid quantitative metrics were generated. Try changing the ticker selection or history period.")
            else:
                ranking_df = ranking_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
                ranking_df.index += 1
                ranking_df.index.name = "Rank"

                top_10 = ranking_df.head(10)
                signals_df = ranking_df[ranking_df['Score'] > score_threshold][[
                    'Ticker', 'Price', 'Bullish_Prob', 'Entry', 'StopLoss', 'Target1', 'Target2', 'RiskReward'
                ]]

                st.success("Scanner complete.")
                st.markdown("### Top 10 Swing Opportunities")
                st.dataframe(top_10, use_container_width=True)

                st.markdown("### Scan Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Ticker count", len(selected_tickers))
                col2.metric("Processed stocks", len(processed_store))
                col3.metric("Top score", f"{top_10.iloc[0]['Score']:.4f}" if not top_10.empty else "N/A")

                csv = ranking_df.to_csv(index=True).encode('utf-8')
                st.download_button("Download full ranking CSV", csv, file_name="swing_rankings.csv", mime="text/csv")

                if not signals_df.empty:
                    st.markdown("### Signal Table")
                    st.dataframe(signals_df, use_container_width=True)
                    signals_csv = signals_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download signals CSV", signals_csv, file_name="swing_signals.csv", mime="text/csv")

                if show_details and not top_10.empty:
                    ticker_options = top_10['Ticker'].tolist()
                    selected_detail = st.selectbox("Show details for", ticker_options)
                    if selected_detail:
                        detail_df = processed_store[selected_detail].copy()
                        detail_df = detail_df[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_20', 'EMA_50', 'EMA_200']].dropna()
                        st.markdown(f"### {selected_detail} price and moving averages")
                        st.line_chart(detail_df[['Close', 'EMA_20', 'EMA_50', 'EMA_200']])
                        latest = detail_df.iloc[-1:].T.rename(columns={detail_df.index[-1]: 'Value'})
                        st.markdown(f"### {selected_detail} latest indicators")
                        st.dataframe(latest, use_container_width=True)
