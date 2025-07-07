# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# --- Do not remove these libs ---
from datetime import datetime
from typing import Optional

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair


class CloudTrendRider(IStrategy):
    """
    Cloud Trend Rider v1.0

    A multi-timeframe trend following strategy that combines:
    - Daily EMA clouds for trend direction
    - RSI divergence detection to avoid reversals
    - Hourly EMA crossovers for precise entries
    - Dual stop-loss system (Fixed 2% → Chandelier Exit)
    - Partial profit taking at 2R

    Designed for 3-5 trades per week with 2:1 minimum R:R
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy
    timeframe = '1h'

    # Enable short trades
    can_short = True

    # --- EXIT MECHANISM CONFIGURATION ---
    use_custom_stoploss = True
    use_exit_signal = True  # We need this for partial exits
    trailing_stop = False
    stoploss = -0.02  # 2% initial stop loss

    # Minimal ROI - used for partial profit taking
    minimal_roi = {
        "0": 0.04,  # 4% (2R) - will trigger partial exit
    }

    # --- RISK MANAGEMENT PARAMETERS ---
    risk_per_trade = 0.02  # Risk 2% of wallet per trade
    first_tp_ratio = 2.0   # First take profit at 2R
    first_tp_percentage = 0.5  # Take 50% at first TP
    max_profit_target = 0.08  # 8% (4R) max target

    # --- CHANDELIER EXIT PARAMETERS ---
    ce_params = {
        'BTC/USDT:USDT': {'lookback': 24, 'mult': 4.0},
        'ETH/USDT:USDT': {'lookback': 24, 'mult': 4.0},
        'SOL/USDT:USDT': {'lookback': 20, 'mult': 4.5},
        'BNB/USDT:USDT': {'lookback': 20, 'mult': 4.2},
        'ADA/USDT:USDT': {'lookback': 24, 'mult': 4.2}
    }
    ce_default = {'lookback': 14, 'mult': 2.5}

    # --- RSI DIVERGENCE PARAMETERS ---
    rsi_period = 14
    divergence_lookback = 20  # Bars to look back for pivot points

    # --- General Strategy Settings ---
    process_only_new_candles = True
    startup_candle_count: int = 250  # Need extra for daily data

    def get_ce_params_for_pair(self, pair: str) -> dict:
        """ Helper to get Chandelier Exit parameters for a specific pair. """
        return self.ce_params.get(pair, self.ce_default)

    # ========================================
    # INFORMATIVE PAIRS - MULTI-TIMEFRAME
    # ========================================
    def informative_pairs(self):
        """
        Define additional pairs/timeframes for multi-timeframe analysis
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1d') for pair in pairs]
        return informative_pairs

    # ========================================
    # LEVERAGE
    # ========================================
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade.
        Start with 1x, can be increased once strategy is proven.
        """
        return 1.0

    # ========================================
    # CUSTOM STAKE AMOUNT - FIXED
    # ========================================
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        """
        Calculate position size based on 2% risk per trade
        
        With a 2% stop loss and 2% risk, we use 100% of our risk allocation
        Position Size = (Risk Amount / Stop Loss %) * Leverage
        """
        wallet_balance = self.wallets.get_total_stake_amount()
        
        # Risk 2% of total wallet
        risk_amount = wallet_balance * self.risk_per_trade
        
        # With 2% stop loss, to risk 2% of wallet:
        # If stop loss = 2%, then position size = risk_amount / 0.02
        # This equals wallet * 0.02 / 0.02 = wallet * 1.0
        # But we limit it to avoid using entire wallet
        position_size = risk_amount / self.stoploss  # stoploss is negative, so we divide by abs
        position_size = abs(position_size)
        
        # Apply leverage
        position_size = position_size * leverage
        
        # Safety check: never use more than 2% of wallet per trade
        position_size = min(position_size, wallet_balance * 0.02)
        
        # Respect min/max stakes
        if min_stake is not None:
            position_size = max(position_size, min_stake)
        position_size = min(position_size, max_stake)
        
        return position_size

    # ========================================
    # RSI DIVERGENCE DETECTION
    # ========================================
    def detect_rsi_divergence(self, dataframe: DataFrame,
                              lookback: int = 20) -> tuple[pd.Series, pd.Series]:
        """
        Detect bullish and bearish RSI divergences
        Returns: (bearish_divergence, bullish_divergence) boolean series
        """
        # Initialize divergence columns
        bearish_div = pd.Series(False, index=dataframe.index)
        bullish_div = pd.Series(False, index=dataframe.index)

        if len(dataframe) < lookback * 2:
            return bearish_div, bullish_div

        # Find pivot highs and lows
        for i in range(lookback, len(dataframe) - 1):
            # Check for pivot high (for bearish divergence)
            if i >= lookback:
                # Current high vs previous high
                curr_idx = i
                prev_high_idx = dataframe['high'].iloc[i-lookback:i].idxmax()

                # Bearish divergence: price HH but RSI LH
                if (dataframe['high'].iloc[curr_idx] > dataframe['high'].iloc[prev_high_idx] and
                        dataframe['rsi'].iloc[curr_idx] < dataframe['rsi'].iloc[prev_high_idx]):
                    bearish_div.iloc[curr_idx] = True

            # Check for pivot low (for bullish divergence)
            if i >= lookback:
                # Current low vs previous low
                curr_idx = i
                prev_low_idx = dataframe['low'].iloc[i-lookback:i].idxmin()

                # Bullish divergence: price LL but RSI HL
                if (dataframe['low'].iloc[curr_idx] < dataframe['low'].iloc[prev_low_idx] and
                        dataframe['rsi'].iloc[curr_idx] > dataframe['rsi'].iloc[prev_low_idx]):
                    bullish_div.iloc[curr_idx] = True

        # Extend divergence signal for a few bars
        bearish_div = bearish_div.rolling(5).max().fillna(False).astype(bool)
        bullish_div = bullish_div.rolling(5).max().fillna(False).astype(bool)

        return bearish_div, bullish_div

    # ========================================
    # POPULATE INDICATORS - HOURLY (FIXED)
    # ========================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators for the main (hourly) timeframe
        """
        pair = metadata.get('pair', '')

        # --- HOURLY INDICATORS ---

        # EMAs for entry signals
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)

        # Volume analysis
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_spike'] = dataframe['volume'] > dataframe['volume_mean']

        # ATR for Chandelier Exit
        ce_params = self.get_ce_params_for_pair(pair)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=ce_params['lookback'])

        # --- DAILY INDICATORS (from informative) ---

        # Get daily data
        informative_1d = self.dp.get_pair_dataframe(pair=pair, timeframe='1d')

        # Calculate indicators on daily timeframe
        informative_1d['ema_12'] = ta.EMA(informative_1d, timeperiod=12)
        informative_1d['ema_26'] = ta.EMA(informative_1d, timeperiod=26)
        informative_1d['ema_50'] = ta.EMA(informative_1d, timeperiod=50)
        informative_1d['ema_200'] = ta.EMA(informative_1d, timeperiod=200)

        # Calculate RSI BEFORE divergence detection
        informative_1d['rsi'] = ta.RSI(informative_1d, timeperiod=self.rsi_period)

        # Detect divergences on daily
        informative_1d['bearish_div'], informative_1d['bullish_div'] = self.detect_rsi_divergence(
            informative_1d, self.divergence_lookback
        )

        # Cloud colors (True = Green/Bullish, False = Red/Bearish)
        informative_1d['fast_cloud_bull'] = informative_1d['ema_12'] > informative_1d['ema_26']
        informative_1d['slow_cloud_bull'] = informative_1d['ema_50'] > informative_1d['ema_200']

        # Merge daily data to hourly using FreqTrade's built-in function
        dataframe = merge_informative_pair(dataframe, informative_1d, self.timeframe, '1d', ffill=True)

        # Create trend conditions using the merged columns
        dataframe['bull_trend'] = (
            (dataframe['slow_cloud_bull_1d'] == 1) &  # Slow cloud green
            (dataframe['rsi_1d'] > 40) &  # RSI not oversold
            (dataframe['bearish_div_1d'] == 0)  # No bearish divergence
        )

        dataframe['bear_trend'] = (
            (dataframe['slow_cloud_bull_1d'] == 0) &  # Slow cloud red
            (dataframe['rsi_1d'] < 60) &  # RSI not overbought
            (dataframe['bullish_div_1d'] == 0)  # No bullish divergence
        )

        # Entry crosses
        dataframe['ema_8_cross_above_9'] = qtpylib.crossed_above(
            dataframe['ema_8'], dataframe['ema_9']
        )
        dataframe['ema_8_cross_below_9'] = qtpylib.crossed_below(
            dataframe['ema_8'], dataframe['ema_9']
        )

        return dataframe

    # ========================================
    # ENTRY LOGIC
    # ========================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals based on multi-timeframe analysis
        """
        # LONG ENTRIES
        dataframe.loc[
            (
                # Daily trend conditions
                (dataframe['bull_trend'] == 1) &

                # Hourly entry conditions
                (dataframe['ema_8_cross_above_9'] == 1) &
                (dataframe['close'] > dataframe['ema_21']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'cloud_trend_long')

        # SHORT ENTRIES
        dataframe.loc[
            (
                # Daily trend conditions
                (dataframe['bear_trend'] == 1) &

                # Hourly entry conditions
                (dataframe['ema_8_cross_below_9'] == 1) &
                (dataframe['close'] < dataframe['ema_21']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_short', 'enter_tag']] = (1, 'cloud_trend_short')

        return dataframe

    # ========================================
    # EXIT LOGIC
    # ========================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals - mainly for trend changes
        """
        # Exit long if trend turns bearish
        dataframe.loc[
            (dataframe['bear_trend'] == 1),
            ['exit_long', 'exit_tag']] = (1, 'trend_change')

        # Exit short if trend turns bullish
        dataframe.loc[
            (dataframe['bull_trend'] == 1),
            ['exit_short', 'exit_tag']] = (1, 'trend_change')

        return dataframe

    # ========================================
    # CUSTOM STOP LOSS - FIXED IMPLEMENTATION
    # ========================================
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool = False, **kwargs) -> float:
        """
        Implement dual stop loss system:
        1. Fixed 2% until 2R profit
        2. Move to breakeven at 2R (handled by minimal_roi)
        3. Switch to Chandelier Exit after 2R
        """
        # Phase 1: Before hitting 2R (4% profit)
        if current_profit < 0.04:
            # Keep original 2% stop loss
            return -0.02

        # Phase 2: After 2R profit - move to breakeven
        elif current_profit >= 0.04 and current_profit < 0.06:
            # Stop at breakeven (small positive to ensure profit)
            return 0.001

        # Phase 3: After 2R, use Chandelier Exit
        else:
            # Get current data
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return 0.001  # Default to breakeven if no data

            last_candle = dataframe.iloc[-1].squeeze()

            # Get Chandelier parameters
            ce_params = self.get_ce_params_for_pair(pair)
            multiplier = ce_params['mult']

            # Get ATR value
            current_atr = last_candle.get('atr', 0)
            if current_atr == 0:
                return 0.001  # Keep at breakeven if no ATR

            # Calculate trailing stop based on highest high since entry
            trade_candles = dataframe.loc[dataframe['date'] >= trade.open_date_utc]
            if trade_candles.empty:
                return 0.001

            if trade.is_short:
                # For shorts: stop trails up from lowest low
                lowest_low = trade_candles['low'].min()
                chandelier_stop = lowest_low + (current_atr * multiplier)
                stop_distance = (chandelier_stop - current_rate) / current_rate
            else:
                # For longs: stop trails down from highest high
                highest_high = trade_candles['high'].max()
                chandelier_stop = highest_high - (current_atr * multiplier)
                stop_distance = (current_rate - chandelier_stop) / current_rate

            # Ensure stop loss is at least at breakeven
            return max(0.001, -stop_distance)

    # ========================================
    # POSITION ADJUSTMENT - PARTIAL EXITS
    # ========================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None:
        """
        Handle partial profit taking at 2R
        This is the proper way to handle partial exits in FreqTrade
        """
        # Check if we've already taken partial profit
        if trade.nr_of_successful_exits == 0:
            # Take 50% profit at 2R (4% gain)
            if current_profit >= 0.04:
                # Return negative value to indicate selling
                # -0.5 means sell 50% of the position
                return -(trade.stake_amount * 0.5)
        
        return None