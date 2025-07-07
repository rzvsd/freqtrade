# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# --- Do not remove these libs ---
from datetime import datetime

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter, DecimalParameter


class HyperoptCloudTrendRider(IStrategy):
    """
    Cloud Trend Rider v2.0 - HYPEROPT VERSION
    
    This is the COMPLETE hyperopt version that includes ALL features:
    - Daily EMA clouds for trend direction (optimizable)
    - RSI divergence detection to avoid reversals (optimizable)
    - Hourly EMA crossovers for precise entries (optimizable)
    - Dual stop-loss system (Fixed → Chandelier Exit) (partially optimizable)
    - Partial profit taking at 2R (fixed - proven to work)
    - Custom position sizing based on 2% risk (fixed - risk management)
    
    What's optimizable:
    - EMA periods (entry and trend)
    - RSI parameters
    - Chandelier Exit parameters
    - Initial stop loss
    - Volume spike threshold
    
    What's fixed (proven risk management):
    - 2% risk per trade
    - 50% partial profit at 2R
    - Breakeven stop after 2R
    - Position sizing logic
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
    
    # Default stop loss (will be optimized)
    stoploss = -0.02  # 2% initial stop loss

    # Minimal ROI - used for partial profit taking (FIXED - proven to work)
    minimal_roi = {
        "0": 0.04,  # 4% (2R) - will trigger partial exit evaluation
    }

    # --- RISK MANAGEMENT PARAMETERS (FIXED - DON'T OPTIMIZE) ---
    risk_per_trade = 0.02  # Risk 2% of wallet per trade
    first_tp_ratio = 2.0   # First take profit at 2R
    first_tp_percentage = 0.5  # Take 50% at first TP
    max_profit_target = 0.08  # 8% (4R) max target

    # --- General Strategy Settings ---
    process_only_new_candles = True
    startup_candle_count: int = 250  # Need extra for daily data

    # ========================================
    # HYPEROPT PARAMETERS
    # ========================================
    
    # --- Stop Loss (optimizable) ---
    stop_loss = DecimalParameter(-0.05, -0.01, default=-0.02, decimals=3, space="sell", optimize=True)
    
    # --- Entry EMAs (1h) ---
    ema_short_period = IntParameter(5, 12, default=8, space="buy", optimize=True)
    ema_long_period = IntParameter(7, 15, default=9, space="buy", optimize=True)
    ema_signal_period = IntParameter(15, 30, default=21, space="buy", optimize=True)
    
    # --- Trend EMAs (1d) ---
    ema_fast_1d = IntParameter(8, 20, default=12, space="buy", optimize=True)
    ema_fast2_1d = IntParameter(20, 35, default=26, space="buy", optimize=True)
    ema_slow_1d = IntParameter(40, 60, default=50, space="buy", optimize=True)
    ema_slow2_1d = IntParameter(180, 220, default=200, space="buy", optimize=True)
    
    # --- RSI Parameters ---
    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    rsi_bull_threshold = IntParameter(35, 45, default=40, space="buy", optimize=True)
    rsi_bear_threshold = IntParameter(55, 65, default=60, space="sell", optimize=True)
    divergence_lookback = IntParameter(15, 30, default=20, space="buy", optimize=True)
    
    # --- Volume Analysis ---
    volume_ma_period = IntParameter(15, 30, default=20, space="buy", optimize=True)
    volume_spike_threshold = DecimalParameter(1.0, 2.0, default=1.0, decimals=1, space="buy", optimize=True)
    
    # --- Chandelier Exit Parameters (global, not per-pair for hyperopt) ---
    ce_lookback = IntParameter(10, 30, default=14, space="sell", optimize=True)
    ce_multiplier = DecimalParameter(2.0, 5.0, default=2.5, decimals=1, space="sell", optimize=True)

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
    # LEVERAGE (FIXED)
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
    # CUSTOM STAKE AMOUNT (FIXED - Risk Management)
    # ========================================
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        """
        Calculate position size based on 2% risk per trade
        This is FIXED - we don't optimize risk management
        """
        wallet_balance = self.wallets.get_total_stake_amount()
        
        # Risk 2% of total wallet
        risk_amount = wallet_balance * self.risk_per_trade
        
        # Calculate position size based on stop loss
        position_size = risk_amount / abs(self.stop_loss.value)
        
        # Apply leverage
        position_size = position_size * leverage
        
        # Safety check: never use more than 20% of wallet per trade
        position_size = min(position_size, wallet_balance * 0.2)
        
        # Respect min/max stakes
        if min_stake is not None:
            position_size = max(position_size, min_stake)
        position_size = min(position_size, max_stake)
        
        return position_size

    # ========================================
    # RSI DIVERGENCE DETECTION
    # ========================================
    def detect_rsi_divergence(self, dataframe: DataFrame, period: int, 
                              lookback: int) -> tuple[pd.Series, pd.Series]:
        """
        Detect bullish and bearish RSI divergences
        Returns: (bearish_divergence, bullish_divergence) boolean series
        """
        # Initialize divergence columns
        bearish_div = pd.Series(False, index=dataframe.index)
        bullish_div = pd.Series(False, index=dataframe.index)

        if len(dataframe) < lookback * 2:
            return bearish_div, bullish_div

        # Calculate RSI if not present
        if 'rsi' not in dataframe.columns:
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=period)

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
    # POPULATE INDICATORS
    # ========================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate indicators for the main (hourly) timeframe
        """
        pair = metadata.get('pair', '')

        # --- HOURLY INDICATORS ---

        # EMAs for entry signals (using hyperopt values)
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.ema_short_period.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.ema_long_period.value)
        dataframe['ema_signal'] = ta.EMA(dataframe, timeperiod=self.ema_signal_period.value)

        # Volume analysis (using hyperopt values)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=self.volume_ma_period.value).mean()
        dataframe['volume_spike'] = (dataframe['volume'] > (dataframe['volume_mean'] * self.volume_spike_threshold.value)).astype(bool)

        # ATR for Chandelier Exit (using hyperopt values)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.ce_lookback.value)

        # --- DAILY INDICATORS (from informative) ---

        # Get daily data
        informative_1d = self.dp.get_pair_dataframe(pair=pair, timeframe='1d')

        # Calculate indicators on daily timeframe (using hyperopt values)
        informative_1d['ema_fast'] = ta.EMA(informative_1d, timeperiod=self.ema_fast_1d.value)
        informative_1d['ema_fast2'] = ta.EMA(informative_1d, timeperiod=self.ema_fast2_1d.value)
        informative_1d['ema_slow'] = ta.EMA(informative_1d, timeperiod=self.ema_slow_1d.value)
        informative_1d['ema_slow2'] = ta.EMA(informative_1d, timeperiod=self.ema_slow2_1d.value)

        # Calculate RSI BEFORE divergence detection
        informative_1d['rsi'] = ta.RSI(informative_1d, timeperiod=self.rsi_period.value)

        # Detect divergences on daily (using hyperopt values)
        informative_1d['bearish_div'], informative_1d['bullish_div'] = self.detect_rsi_divergence(
            informative_1d, self.rsi_period.value, self.divergence_lookback.value
        )

        # Cloud colors (True = Green/Bullish, False = Red/Bearish)
        informative_1d['fast_cloud_bull'] = (informative_1d['ema_fast'] > informative_1d['ema_fast2']).astype(bool)
        informative_1d['slow_cloud_bull'] = (informative_1d['ema_slow'] > informative_1d['ema_slow2']).astype(bool)

        # Merge daily data to hourly using FreqTrade's built-in function
        dataframe = merge_informative_pair(dataframe, informative_1d, self.timeframe, '1d', ffill=True)

        # Ensure boolean columns are properly typed after merge
        dataframe['slow_cloud_bull_1d'] = dataframe['slow_cloud_bull_1d'].fillna(False).astype(bool)
        dataframe['bearish_div_1d'] = dataframe['bearish_div_1d'].fillna(False).astype(bool)
        dataframe['bullish_div_1d'] = dataframe['bullish_div_1d'].fillna(False).astype(bool)

        # Create trend conditions using the merged columns (using hyperopt values)
        dataframe['bull_trend'] = (
            (dataframe['slow_cloud_bull_1d'] == True) &  # Slow cloud green
            (dataframe['rsi_1d'] > self.rsi_bull_threshold.value) &  # RSI not oversold
            (dataframe['bearish_div_1d'] == False)  # No bearish divergence
        )

        dataframe['bear_trend'] = (
            (dataframe['slow_cloud_bull_1d'] == False) &  # Slow cloud red
            (dataframe['rsi_1d'] < self.rsi_bear_threshold.value) &  # RSI not overbought
            (dataframe['bullish_div_1d'] == False)  # No bullish divergence
        )

        # Entry crosses
        dataframe['ema_cross_up'] = qtpylib.crossed_above(
            dataframe['ema_short'], dataframe['ema_long']
        )
        dataframe['ema_cross_down'] = qtpylib.crossed_below(
            dataframe['ema_short'], dataframe['ema_long']
        )

        return dataframe

    # ========================================
    # ENTRY LOGIC
    # ========================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals based on multi-timeframe analysis
        """
        # BALANCED ENTRY CONDITIONS
        # LONG ENTRIES
        dataframe.loc[
            (
                # Daily trend must be bullish
                (dataframe['slow_cloud_bull_1d'] == True) &
                (dataframe['rsi_1d'] > self.rsi_bull_threshold.value) &
                
                # Hourly EMA cross
                (dataframe['ema_cross_up'] == True) &
                
                # Price must be above signal EMA (trend filter)
                (dataframe['close'] > dataframe['ema_signal']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']] = (1, 'cloud_trend_long')

        # SHORT ENTRIES
        dataframe.loc[
            (
                # Daily trend must be bearish
                (dataframe['slow_cloud_bull_1d'] == False) &
                (dataframe['rsi_1d'] < self.rsi_bear_threshold.value) &
                
                # Hourly EMA cross
                (dataframe['ema_cross_down'] == True) &
                
                # Price must be below signal EMA (trend filter)
                (dataframe['close'] < dataframe['ema_signal']) &
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
            (dataframe['bear_trend'] == True),
            ['exit_long', 'exit_tag']] = (1, 'trend_change')

        # Exit short if trend turns bullish
        dataframe.loc[
            (dataframe['bull_trend'] == True),
            ['exit_short', 'exit_tag']] = (1, 'trend_change')

        return dataframe

    # ========================================
    # CUSTOM STOP LOSS (Dual System)
    # ========================================
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool = False, **kwargs) -> float:
        """
        Implement dual stop loss system:
        1. Fixed stop (optimized) until 2R profit
        2. Move to breakeven at 2R
        3. Switch to Chandelier Exit after 2R
        """
        # Phase 1: Before hitting 2R (4% profit)
        if current_profit < 0.04:
            # Use optimized stop loss
            return self.stop_loss.value

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

            # Use optimized Chandelier parameters
            multiplier = self.ce_multiplier.value

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
    # POSITION ADJUSTMENT - PARTIAL EXITS (FIXED)
    # ========================================
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None:
        """
        Handle partial profit taking at 2R
        This is FIXED - proven to work well
        """
        # Check if we've already taken partial profit
        if trade.nr_of_successful_exits == 0:
            # Take 50% profit at 2R (4% gain)
            if current_profit >= 0.04:
                # Return negative value to indicate selling
                # -0.5 means sell 50% of the position
                return -(trade.stake_amount * 0.5)
        
        return None

    # ========================================
    # HYPEROPT CONFIGURATION
    # ========================================
    @property
    def protections(self):
        """
        Enable protections to avoid bad market conditions
        """
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 3
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 200,
                "trade_limit": 1,
                "stop_duration_candles": 10,
                "max_allowed_drawdown": 0.15
            }
        ]