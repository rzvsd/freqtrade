# Filename: Hyperopt_CitadelV3.py
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement, unused-argument

from datetime import datetime
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from typing import Optional

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open, IntParameter, DecimalParameter


class Hyperopt_CitadelV3(IStrategy):
    """
    Citadel System - Version 3.0 HYPEROPT - LONG ONLY
    
    This is the hyperopt version of CitadelV3_LONG_Strategy.
    All key parameters are now optimizable to find the best combinations.
    
    Optimizable parameters:
    - BBW squeeze threshold
    - ADX threshold for strong/weak classification
    - EMA periods
    - Volume Z-score threshold
    - Stop loss percentages
    - Take profit levels
    - Risk allocation percentages
    """
    
    # --- Base Strategy Config ---
    INTERFACE_VERSION = 3
    timeframe = '1h'
    can_short = False  # This strategy is LONG ONLY
    process_only_new_candles = True
    startup_candle_count: int = 200  # Increased for longer EMAs

    # --- EXIT MECHANISM CONFIGURATION ---
    # These will be overridden by hyperopt
    stoploss = -0.10
    use_custom_stoploss = True
    minimal_roi = {"0": 0.05}
    use_exit_signal = False

    # ========================================
    # HYPEROPT PARAMETERS
    # ========================================
    
    # --- Macro Lens Parameters (4H) ---
    # BBW Squeeze threshold (0.3 = 30% of average, 0.7 = 70% of average)
    bbw_squeeze_threshold = DecimalParameter(0.3, 0.7, default=0.5, decimals=2, space="buy", optimize=True)
    
    # ADX threshold for strong vs weak trend
    adx_threshold = IntParameter(20, 35, default=25, space="buy", optimize=True)
    
    # --- Entry Parameters (1H) ---
    # EMA periods
    ema_fast_period = IntParameter(5, 15, default=10, space="buy", optimize=True)
    ema_slow_period = IntParameter(20, 35, default=25, space="buy", optimize=True)
    
    # Volume Z-score threshold
    volume_zscore_threshold = DecimalParameter(0.5, 2.5, default=1.5, decimals=1, space="buy", optimize=True)
    
    # Volume lookback period for Z-score calculation
    volume_lookback = IntParameter(50, 200, default=100, step=10, space="buy", optimize=True)
    
    # --- Risk Management Parameters ---
    # Base risk allocation (% of wallet)
    base_risk_pct = DecimalParameter(0.01, 0.03, default=0.02, decimals=3, space="buy", optimize=True)
    
    # Weak bull risk modifier (0.5 = 50% of base, 0.9 = 90% of base)
    weak_bull_modifier = DecimalParameter(0.5, 0.9, default=0.7, decimals=2, space="buy", optimize=True)
    
    # Stop loss
    stop_loss = DecimalParameter(-0.15, -0.05, default=-0.10, decimals=3, space="sell", optimize=True)
    
    # Tight trailing stop when trend changes
    tight_trail_pct = DecimalParameter(0.01, 0.04, default=0.02, decimals=3, space="sell", optimize=True)
    
    # Take profit levels
    take_profit = DecimalParameter(0.03, 0.10, default=0.05, decimals=3, space="sell", optimize=True)

    def informative_pairs(self):
        """Define the 4h timeframe for the Macro Lens."""
        pairs = self.dp.current_whitelist()
        return [(pair, '4h') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators for all timeframes."""
        # --- MACRO LENS (4h Data) ---
        informative_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')

        # Bollinger Bands & BBW
        bollinger = ta.BBANDS(informative_4h['close'], timeperiod=20)
        informative_4h['bbw'] = ((bollinger[0] - bollinger[2]) / bollinger[1])
        informative_4h['bbw_avg_100'] = informative_4h['bbw'].rolling(100).mean()

        # Ichimoku Cloud
        period9_high = informative_4h['high'].rolling(window=9).max()
        period9_low = informative_4h['low'].rolling(window=9).min()
        informative_4h['tenkan_sen'] = (period9_high + period9_low) / 2
        period26_high = informative_4h['high'].rolling(window=26).max()
        period26_low = informative_4h['low'].rolling(window=26).min()
        informative_4h['kijun_sen'] = (period26_high + period26_low) / 2
        informative_4h['senkou_a'] = ((informative_4h['tenkan_sen'] + informative_4h['kijun_sen']) / 2).shift(26)
        period52_high = informative_4h['high'].rolling(window=52).max()
        period52_low = informative_4h['low'].rolling(window=52).min()
        informative_4h['senkou_b'] = ((period52_high + period52_low) / 2).shift(26)

        # ADX for trend strength
        informative_4h['adx'] = ta.ADX(informative_4h, timeperiod=14)

        dataframe = merge_informative_pair(dataframe, informative_4h, self.timeframe, '4h', ffill=True)

        # --- MACRO LENS VERDICT ---
        # Using hyperopt parameters
        base_go_long = (
            (dataframe['bbw_4h'] < (dataframe['bbw_avg_100_4h'] * self.bbw_squeeze_threshold.value)) &
            (dataframe['close'] > dataframe['senkou_a_4h'])
        )
        dataframe['strong_bull'] = base_go_long & (dataframe['adx_4h'] > self.adx_threshold.value)
        dataframe['weak_bull'] = base_go_long & (dataframe['adx_4h'] <= self.adx_threshold.value)
        dataframe['go_long'] = dataframe['strong_bull'] | dataframe['weak_bull']

        # --- ENTRY INDICATORS (1h Data) ---
        # Using hyperopt EMA periods
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast_period.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_period.value)

        # Volume Z-Score with hyperopt lookback
        volume_mean = dataframe['volume'].rolling(window=self.volume_lookback.value).mean()
        volume_std = dataframe['volume'].rolling(window=self.volume_lookback.value).std()
        dataframe['volume_zscore'] = (dataframe['volume'] - volume_mean) / volume_std
        dataframe['volume_spike'] = dataframe['volume_zscore'] > self.volume_zscore_threshold.value

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry signals based on Macro Lens and 1h triggers."""
        dataframe.loc[
            (
                dataframe['go_long'] &
                dataframe['volume_spike'] &
                qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit logic is handled by custom_stoploss and minimal_roi."""
        return dataframe

    @property
    def protections(self):
        """
        Enable protections to avoid trading in bad conditions.
        These are optional but recommended.
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

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        """Dynamically size position based on market regime strength."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return 0.0
        last_candle = dataframe.iloc[-1].squeeze()
        
        wallet_balance = self.wallets.get_total_stake_amount()
        risk_amount = 0.0

        if last_candle['strong_bull']:
            risk_amount = wallet_balance * self.base_risk_pct.value
        elif last_candle['weak_bull']:
            risk_amount = wallet_balance * (self.base_risk_pct.value * self.weak_bull_modifier.value)
        
        if risk_amount == 0.0:
            return 0.0

        # Calculate position size based on risk amount and stoploss
        position_size = risk_amount / abs(self.stop_loss.value)
        
        # Ensure we don't exceed max_stake
        return min(position_size, max_stake)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Advanced stoploss that tightens when trend changes."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stop_loss.value
            
        last_candle = dataframe.iloc[-1].squeeze()

        # Check if the macro 'go_long' condition is still true
        if not last_candle['go_long']:
            # If trend flips, activate a tight trailing stop loss
            return stoploss_from_open(self.tight_trail_pct.value, current_profit,
                                      is_short=trade.is_short, leverage=trade.leverage)

        # If trend is still valid, use the optimized stoploss
        return self.stop_loss.value

    # Override the minimal_roi to use hyperopt parameter
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Update minimal_roi with hyperopt value
        if hasattr(self, 'take_profit'):
            self.minimal_roi = {"0": self.take_profit.value}