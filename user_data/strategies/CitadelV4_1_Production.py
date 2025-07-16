# Filename: CitadelV4_1_Production.py
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement, unused-argument

from datetime import datetime
import pandas as pd
import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_from_open

class CitadelV4_1_Production(IStrategy):
    """
    Citadel System - Version 4.1 - PRODUCTION
    
    This is the final, optimized version of the multi-lens sniper strategy.
    All parameters have been fine-tuned by a 500-epoch hyperopt run.
    
    EXPECTED PERFORMANCE (2-year backtest):
    - Total Profit: +106.29%
    - Max Drawdown: 8.31%
    - Win Rate: 74.2%
    - Total Trades: 62
    """
    
    # --- Base Strategy Config ---
    timeframe = '15m'
    can_short = False
    process_only_new_candles = True
    startup_candle_count: int = 200
    INTERFACE_VERSION = 3

    # --- EXIT MECHANISM (Optimized) ---
    use_custom_stoploss = True
    minimal_roi = {"0": 0.086} # From sell_params.roi_tp
    use_exit_signal = False

    # --- OPTIMIZED PARAMETERS (from Epoch 38) ---
    # Macro Lens (4H)
    bbw_squeeze_threshold = 0.61
    adx_threshold = 28
    # Meso Lens (1H)
    volume_lookback = 50
    volume_zscore_threshold = 1.0
    # Micro Lens (15m)
    bos_lookback_candles = 8
    # Risk Management
    base_risk_allocation = 0.021
    weak_bull_modifier = 0.77
    # Custom Stop
    tight_trailing_stop_pct = 0.035
    max_structural_stop_pct = 0.066

    def informative_pairs(self):
        """Define all timeframes needed by the strategy."""
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs] + [(pair, '4h') for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators for all timeframes."""
        
        inf_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')
        inf_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')

        # --- 4H Indicators ---
        inf_4h['adx'] = ta.ADX(inf_4h, timeperiod=14)
        bollinger_4h = ta.BBANDS(inf_4h['close'], timeperiod=20)
        inf_4h['bb_upperband'] = bollinger_4h[0]
        inf_4h['bb_middleband'] = bollinger_4h[1]
        inf_4h['bb_lowerband'] = bollinger_4h[2]
        inf_4h['bbw'] = (inf_4h['bb_upperband'] - inf_4h['bb_lowerband']) / inf_4h['bb_middleband']
        inf_4h['bbw_avg_100'] = inf_4h['bbw'].rolling(100).mean()
        period9_high = inf_4h['high'].rolling(window=9).max()
        period9_low = inf_4h['low'].rolling(window=9).min()
        inf_4h['tenkan_sen'] = (period9_high + period9_low) / 2
        period26_high = inf_4h['high'].rolling(window=26).max()
        period26_low = inf_4h['low'].rolling(window=26).min()
        inf_4h['kijun_sen'] = (period26_high + period26_low) / 2
        inf_4h['senkou_a'] = ((inf_4h['tenkan_sen'] + inf_4h['kijun_sen']) / 2).shift(26)
        
        # --- 1H Indicators ---
        inf_1h['volume_spike'] = (inf_1h['volume'] - inf_1h['volume'].rolling(self.volume_lookback).mean()) / inf_1h['volume'].rolling(self.volume_lookback).std() > self.volume_zscore_threshold
        inf_1h['smc_liquidity_grab'] = inf_1h['low'] < inf_1h['low'].shift(1)
        inf_1h['smc_bullish_engulfing'] = inf_1h['close'] > inf_1h['high'].shift(1)
        inf_1h['smc_signal'] = inf_1h['smc_liquidity_grab'] & inf_1h['smc_bullish_engulfing']

        # --- Merge Data ---
        dataframe = merge_informative_pair(dataframe, inf_4h, self.timeframe, '4h', ffill=True)
        dataframe = merge_informative_pair(dataframe, inf_1h, self.timeframe, '1h', ffill=True)

        # --- Verdicts (on main 15m dataframe) ---
        base_go_long = (
            (dataframe['bbw_4h'] < (dataframe['bbw_avg_100_4h'] * self.bbw_squeeze_threshold)) &
            (dataframe['close_4h'] > dataframe['senkou_a_4h'])
        )
        dataframe['strong_bull'] = base_go_long & (dataframe['adx_4h'] > self.adx_threshold)
        dataframe['weak_bull'] = base_go_long & (dataframe['adx_4h'] <= self.adx_threshold)
        dataframe['go_long'] = dataframe['strong_bull'] | dataframe['weak_bull']
        dataframe['zone_of_interest'] = (dataframe['go_long'] & (dataframe['volume_spike_1h'] | dataframe['smc_signal_1h']))

        # --- 15m Indicators ---
        dataframe['bos_swing_high'] = dataframe['high'].rolling(self.bos_lookback_candles).max().shift(1)
        dataframe['bos_confirmed'] = dataframe['close'] > dataframe['bos_swing_high']
        dataframe['fvg_high'] = dataframe['high'].shift(2)
        dataframe['fvg_low'] = dataframe['low'].shift(1)
        dataframe['fvg_exists'] = dataframe['fvg_high'] > dataframe['fvg_low']
        dataframe['fvg_retested'] = (dataframe['low'] <= dataframe['fvg_high'].shift(1)) & dataframe['fvg_exists'].shift(1)
        dataframe['sniper_entry'] = dataframe['bos_confirmed'] & dataframe['fvg_retested']
        dataframe['structural_low'] = dataframe['low'].rolling(self.bos_lookback_candles).min()
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['dynamic_stop_loss_pct'] = np.nan
        stop_dist_pct = (dataframe['close'] - dataframe['structural_low']) / dataframe['close']
        entry_condition = (
            dataframe['zone_of_interest'] &
            dataframe['sniper_entry'] &
            (stop_dist_pct < self.max_structural_stop_pct) &
            (stop_dist_pct > 0)
        )
        dataframe.loc[entry_condition, 'enter_long'] = 1
        dataframe.loc[entry_condition, 'dynamic_stop_loss_pct'] = stop_dist_pct
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty: return 0.0
        last_candle = dataframe.iloc[-1].squeeze()
        stop_loss_distance = last_candle.get('dynamic_stop_loss_pct')
        if pd.isna(stop_loss_distance) or stop_loss_distance <= 0: return 0.0

        wallet_balance = self.wallets.get_total_stake_amount()
        risk_amount = 0.0
        if last_candle['strong_bull']:
            risk_amount = wallet_balance * self.base_risk_allocation
        elif last_candle['weak_bull']:
            risk_amount = wallet_balance * (self.base_risk_allocation * self.weak_bull_modifier)
        
        if risk_amount == 0.0: return 0.0
        
        position_size = risk_amount / stop_loss_distance
        return min(position_size, max_stake)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty: return self.max_structural_stop_pct
        last_candle = dataframe.iloc[-1].squeeze()
        
        if not last_candle['go_long']:
            return stoploss_from_open(self.tight_trailing_stop_pct, current_profit,
                                      is_short=trade.is_short, leverage=trade.leverage)
        
        trade_entry_row = dataframe.loc[dataframe['date'] == trade.open_date]
        if not trade_entry_row.empty:
            dynamic_stop_pct = trade_entry_row.iloc[0].get('dynamic_stop_loss_pct')
            if pd.notna(dynamic_stop_pct) and dynamic_stop_pct > 0:
                return -dynamic_stop_pct

        return self.max_structural_stop_pct

    @property
    def protections(self):
        return [
            { "method": "CooldownPeriod", "stop_duration_candles": 3 },
            { "method": "MaxDrawdown", "lookback_period_candles": 200, "trade_limit": 1,
              "stop_duration_candles": 10, "max_allowed_drawdown": 0.15 }
        ]