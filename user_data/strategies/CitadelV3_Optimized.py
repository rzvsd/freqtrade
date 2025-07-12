# Filename: CitadelV3_Optimized.py
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement, unused-argument

from datetime import datetime
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open


class CitadelV3_Optimized(IStrategy):
    """
    Citadel System - Version 3.0 - OPTIMIZED LONG ONLY
    
    This strategy uses the EXACT parameters from the successful hyperopt run.
    Parameters are hardcoded from Hyperopt_CitadelV3.json for production use.
    
    Expected Performance:
    - ~6.95% profit
    - 55.8% win rate
    - Max drawdown ~6.12%
    """
    
    # --- Base Strategy Config ---
    INTERFACE_VERSION = 3
    timeframe = '1h'
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 200

    # --- EXIT MECHANISM (from JSON) ---
    stoploss = -0.062  # From JSON stoploss.stoploss
    use_custom_stoploss = True
    minimal_roi = {"0": 0.042}  # From JSON roi
    use_exit_signal = False

    # --- OPTIMIZED PARAMETERS (from JSON buy/sell params) ---
    
    # Macro Lens Parameters (4H)
    bbw_squeeze_threshold = 0.42  # From buy_params
    adx_threshold = 27  # From buy_params
    
    # Entry Parameters (1H)  
    ema_fast_period = 8  # From buy_params
    ema_slow_period = 30  # From buy_params
    volume_lookback = 70  # From buy_params
    volume_zscore_threshold = 0.8  # From buy_params
    
    # Risk Management
    base_risk_allocation = 0.029  # From buy_params.base_risk_pct
    weak_bull_modifier = 0.8  # From buy_params.weak_bull_modifier
    
    # Custom Stop Parameters
    stop_loss_value = -0.062  # From sell_params.stop_loss
    tight_trailing_stop_pct = 0.017  # From sell_params.tight_trail_pct

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
        
        # Merge 4h data to 1h
        dataframe = merge_informative_pair(dataframe, informative_4h, self.timeframe, '4h', ffill=True)
        
        # --- MACRO LENS VERDICT ---
        base_go_long = (
            (dataframe['bbw_4h'] < (dataframe['bbw_avg_100_4h'] * self.bbw_squeeze_threshold)) &
            (dataframe['close'] > dataframe['senkou_a_4h'])
        )
        dataframe['strong_bull'] = base_go_long & (dataframe['adx_4h'] > self.adx_threshold)
        dataframe['weak_bull'] = base_go_long & (dataframe['adx_4h'] <= self.adx_threshold)
        dataframe['go_long'] = dataframe['strong_bull'] | dataframe['weak_bull']
        
        # --- ENTRY INDICATORS (1h Data) ---
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast_period)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow_period)
        
        # Volume Z-Score
        volume_mean = dataframe['volume'].rolling(window=self.volume_lookback).mean()
        volume_std = dataframe['volume'].rolling(window=self.volume_lookback).std()
        dataframe['volume_zscore'] = (dataframe['volume'] - volume_mean) / volume_std
        dataframe['volume_spike'] = dataframe['volume_zscore'] > self.volume_zscore_threshold
        
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
            risk_amount = wallet_balance * self.base_risk_allocation
        elif last_candle['weak_bull']:
            risk_amount = wallet_balance * (self.base_risk_allocation * self.weak_bull_modifier)
        
        if risk_amount == 0.0:
            return 0.0
        
        # Calculate position size based on risk amount and stoploss
        position_size = risk_amount / abs(self.stop_loss_value)
        
        return min(position_size, max_stake)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Advanced stoploss that tightens when trend changes."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stop_loss_value
            
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Check if the macro 'go_long' condition is still true
        if not last_candle['go_long']:
            # If trend flips, activate a tight trailing stop loss
            return stoploss_from_open(self.tight_trailing_stop_pct, current_profit,
                                      is_short=trade.is_short, leverage=trade.leverage)
        
        # If trend is still valid, use the standard stoploss
        return self.stop_loss_value

    @property
    def protections(self):
        """
        Enable protections to avoid trading in bad conditions.
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