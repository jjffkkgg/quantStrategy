import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from utils.backtest import run_backtest, defer_pre_history_allocations
from utils.bond_proxy import treasury_total_return
from utils.macro_data import unemployment_asof, load_unemployment_vintages
from utils.trading_calendar import completed_month_ends
from strategies import laaMA4, laaMA_sandbox


class CorrectionsTest(unittest.TestCase):
    def test_signal_sleeve_never_rebalances_at_month_end(self):
        idx = pd.bdate_range('2024-01-29', periods=10)
        p = pd.DataFrame({'QQQ': [100., 100., 200., 200., 300., 300., 300., 300., 300., 300.],
                          'IEF': np.linspace(100., 110., 10), 'IAU': 100.,
                          'IWD': 100., 'SGOV': 100.}, index=idx)
        w = pd.DataFrame({'QQQ': .25, 'IEF': .25, 'IAU': .25, 'IWD': .25, 'SGOV': 0.}, index=idx)
        w.loc[idx[5:7], 'QQQ'] = 0.
        w.loc[idx[5:7], 'SGOV'] = .25
        result = self.simulate(p, w, signal_only_asset='QQQ')
        q = result.trade_log.query("ticker == 'QQQ'")
        self.assertEqual(list(q.index), [idx[1], idx[6], idx[8]])
        np.testing.assert_allclose(q.amount, [25., -75., 75.])
        core = result.trade_log.query("sleeve == 'monthly_core'")
        self.assertEqual(set(core.index), {idx[1], pd.Timestamp('2024-02-01')})
        self.assertTrue(result.equity_curve.gt(0).all())

    def test_monthly_core_ignores_midmonth_target_changes(self):
        idx = pd.bdate_range('2024-01-29', periods=7)
        p = pd.DataFrame(100., index=idx, columns=['QQQ', 'IEF', 'IAU', 'IWD', 'SGOV'])
        w = pd.DataFrame({'QQQ': 0., 'IEF': .25, 'IAU': .25, 'IWD': .25, 'SGOV': .25}, index=idx)
        w.loc[idx[4]:, 'IAU'] = 0.
        w.loc[idx[4]:, 'SGOV'] = .5
        result = self.simulate(p, w, signal_only_asset='QQQ')
        self.assertFalse((result.trade_log.action == 'SELL').any())
        np.testing.assert_allclose(result.equity_curve, 100.)

    def test_signal_sleeve_cash_is_reserved_through_month_end(self):
        idx = pd.bdate_range('2024-01-29', periods=7)
        p = pd.DataFrame({'QQQ': 100., 'IEF': [100., 100., 200., 200., 200., 200., 200.],
                          'IAU': 100., 'IWD': 100., 'SGOV': 100.}, index=idx)
        w = pd.DataFrame({'QQQ': 0., 'IEF': .25, 'IAU': .25, 'IWD': .25, 'SGOV': .25}, index=idx)
        w.loc[idx[4]:, 'QQQ'] = .25
        w.loc[idx[4]:, 'SGOV'] = 0.
        result = self.simulate(p, w, signal_only_asset='QQQ')
        q = result.trade_log.query("ticker == 'QQQ'")
        self.assertEqual(list(q.index), [idx[5]])
        self.assertAlmostEqual(q.amount.iloc[0], 25.)
        np.testing.assert_allclose(result.equity_curve.iloc[2:], 125.)

    def test_pre_history_cash_and_delayed_entry(self):
        idx = pd.bdate_range('1985-05-15', periods=5)
        p = pd.DataFrame({'QQQ': [np.nan, np.nan, 100., 110., 121.],
                          'SGOV': [100., 101., 102., 103., 104.]}, index=idx)
        w = pd.DataFrame({'QQQ': 1., 'SGOV': 0.}, index=idx)
        adjusted = defer_pre_history_allocations(p, w)
        np.testing.assert_allclose(adjusted.QQQ, [0., 0., 1., 1., 1.])
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.)
        self.assertTrue(w.QQQ.eq(1.).all())
        result = self.simulate(p, adjusted, rebalance='changes')
        buys = result.trade_log.query("ticker == 'QQQ' and action == 'BUY'")
        self.assertEqual(list(buys.index), [idx[3]])
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 100 * 103/101 * 1.1)

    def test_post_inception_missing_price_still_errors(self):
        idx = pd.bdate_range('1985-05-15', periods=3)
        p = pd.DataFrame({'QQQ': [100., np.nan, 101.], 'SGOV': 100.}, index=idx)
        w = pd.DataFrame({'QQQ': 1., 'SGOV': 0.}, index=idx)
        adjusted = defer_pre_history_allocations(p, w)
        with self.assertRaisesRegex(ValueError, 'QQQ'):
            self.simulate(p, adjusted)

    def test_entire_download_missing_is_not_masked(self):
        idx = pd.bdate_range('1985-05-15', periods=3)
        p = pd.DataFrame({'QQQ': np.nan, 'SGOV': 100.}, index=idx)
        w = pd.DataFrame({'QQQ': 1., 'SGOV': 0.}, index=idx)
        with self.assertRaisesRegex(ValueError, 'check download'):
            defer_pre_history_allocations(p, w)

    def test_missing_execution_defers_without_stale_price_fill(self):
        idx = pd.bdate_range('2004-01-01', periods=4)
        p = pd.DataFrame({'IAU': [100., np.nan, 110., 121.]}, index=idx)
        w = pd.DataFrame({'IAU': 1.}, index=idx)
        result = self.simulate(p, w, rebalance='changes', missing_execution='defer')
        self.assertEqual(list(result.trade_log.index), [idx[2]])
        np.testing.assert_allclose(result.equity_curve, [100., 100., 100., 110.])

    def test_new_signal_cancels_pending_buy(self):
        idx = pd.bdate_range('2004-01-01', periods=4)
        p = pd.DataFrame({'IAU': [100., np.nan, 110., 121.]}, index=idx)
        w = pd.DataFrame({'IAU': [1., 0., 0., 0.]}, index=idx)
        result = self.simulate(p, w, rebalance='changes', missing_execution='defer')
        self.assertTrue(result.trade_log.empty)
        np.testing.assert_allclose(result.equity_curve, 100.)

    def test_vintage_release_and_revision(self):
        v = pd.DataFrame({
            'date': pd.to_datetime(['2020-03-01'] * 2),
            'realtime_start': pd.to_datetime(['2020-04-03', '2021-01-08']),
            'value': [4.4, 4.5]})
        self.assertTrue(unemployment_asof(v, '2020-03-31').empty)
        self.assertEqual(unemployment_asof(v, '2020-04-03').iloc[-1], 4.4)
        self.assertEqual(unemployment_asof(v, '2021-01-08').iloc[-1], 4.5)

    def test_no_silent_revised_data_fallback(self):
        load_unemployment_vintages.cache_clear()
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'FRED_API_KEY'):
                load_unemployment_vintages()

    def test_vintage_api_pagination(self):
        load_unemployment_vintages.cache_clear()
        from unittest.mock import Mock
        responses = [Mock(status_code=200), Mock(status_code=200)]
        for response, release, value in zip(responses, ['2020-04-03', '2021-01-08'], ['4.4', '4.5']):
            response.json.return_value = dict(count=2, observations=[
                dict(date='2020-03-01', realtime_start=release, value=value)])
        with patch.dict('os.environ', {'FRED_API_KEY': 'test-only'}, clear=True):
            with patch('utils.macro_data.requests.get', side_effect=responses) as request:
                v = load_unemployment_vintages()
                self.assertEqual(request.call_args_list[1].kwargs['params']['offset'], 1)
                self.assertEqual(unemployment_asof(v, '2020-04-03').iloc[0], 4.4)
        load_unemployment_vintages.cache_clear()

    def test_weekend_month_end_in_live_prefix(self):
        idx = pd.bdate_range('2021-07-01', '2021-07-30')
        self.assertEqual(list(completed_month_ends(idx)), [pd.Timestamp('2021-07-30')])
        self.assertTrue(completed_month_ends(idx[:-1]).empty)

    def test_holiday_month_end(self):
        idx = pd.bdate_range('2024-03-01', '2024-03-28')  # Good Friday March 29
        self.assertEqual(list(completed_month_ends(idx)), [pd.Timestamp('2024-03-28')])

    def test_gold_prefix_invariance(self):
        idx = pd.bdate_range('2021-07-01', '2021-08-10')
        s = pd.Series(True, index=idx)
        full = laaMA4._completed_month_end_signal(s, idx)
        short = laaMA4._completed_month_end_signal(s.loc[:'2021-07-30'], idx[idx <= '2021-07-30'])
        pd.testing.assert_series_equal(full, short)

    def test_regime_applies_weekend_signal_in_both_strategies(self):
        idx = pd.bdate_range('2020-01-01', '2021-08-05')
        prices = pd.DataFrame({'SPY': np.linspace(200, 100, len(idx))}, index=idx)
        months = pd.date_range('2018-01-01', '2021-06-01', freq='MS')
        v = pd.DataFrame({'date': months, 'realtime_start': months + pd.offsets.MonthBegin(1), 'value': 4.})
        v.loc[v.index[-1], 'value'] = 8.
        for module in [laaMA4, laaMA_sandbox]:
            with patch.object(module, 'load_unemployment_vintages', return_value=v):
                flags = module._compute_regime_flags(prices)
            self.assertFalse(flags.loc['2021-06-30', 'recession'])
            self.assertTrue(flags.loc['2021-07-30', 'recession'])
            self.assertTrue(flags.loc['2021-08-02', 'recession'])

    def test_bond_carry_at_constant_yield_includes_weekend(self):
        idx = pd.to_datetime(['2024-01-05', '2024-01-08'])
        model = treasury_total_return(pd.Series(4., index=idx), 8.5)
        self.assertAlmostEqual(model.iloc[-1], 1.02 ** (6/365.25), places=12)

    def test_bond_zero_yield_and_rate_shock(self):
        idx = pd.bdate_range('2024-01-02', periods=2)
        model = treasury_total_return(pd.Series([4., 5.], index=idx), 8.5)
        self.assertTrue(.90 < model.iloc[-1] < 1.)
        flat = treasury_total_return(pd.Series([0., 0.], index=idx), 8.5)
        self.assertEqual(flat.iloc[-1], 1.)

    def test_bond_no_future_yield_leakage(self):
        idx = pd.bdate_range('2024-01-02', periods=4)
        y = pd.Series([4., 4.1, 4.2, 9.], index=idx)
        pd.testing.assert_series_equal(treasury_total_return(y, 8.5).iloc[:3],
                                       treasury_total_return(y.iloc[:3], 8.5))

    def simulate(self, p, w, **kwargs):
        return run_backtest(p, w, initial_capital=100., calculate_real_cagr=False, **kwargs)

    def test_next_close_execution_excludes_pre_execution_return(self):
        idx = pd.bdate_range('2024-01-02', periods=4)
        p = pd.DataFrame({'A': [100., 200., 220., 110.]}, index=idx)
        w = pd.DataFrame({'A': [1., 1., 0., 0.]}, index=idx)
        result = self.simulate(p, w, rebalance='changes')
        np.testing.assert_allclose(result.equity_curve, [100., 100., 110., 55.])
        self.assertEqual(list(result.trade_log.index), [idx[1], idx[3]])

    def test_holdings_drift_without_hidden_daily_rebalance(self):
        idx = pd.bdate_range('2024-01-02', periods=4)
        p = pd.DataFrame({'A': [100., 100., 200., 400.], 'B': 100.}, index=idx)
        w = pd.DataFrame({'A': .5, 'B': .5}, index=idx)
        result = self.simulate(p, w)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 250.)
        self.assertEqual(len(result.trade_log), 2)

    def test_monthly_rebalance_logs_actual_drift_trades(self):
        idx = pd.to_datetime(['2024-01-29', '2024-01-30', '2024-01-31', '2024-02-01'])
        p = pd.DataFrame({'A': [100., 100., 200., 200.], 'B': 100.}, index=idx)
        w = pd.DataFrame({'A': .5, 'B': .5}, index=idx)
        result = self.simulate(p, w)
        trades = result.trade_log.loc['2024-02-01'].set_index('ticker')
        self.assertAlmostEqual(trades.loc['A', 'old_w'], 2/3)
        self.assertAlmostEqual(trades.loc['A', 'amount'], -25.)
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 150.)

    def test_missing_trade_price_is_not_free_exposure(self):
        idx = pd.bdate_range('2024-01-02', periods=3)
        p = pd.DataFrame({'A': [np.nan, np.nan, 100.]}, index=idx)
        w = pd.DataFrame({'A': 1.}, index=idx)
        with self.assertRaisesRegex(ValueError, 'Cannot execute'):
            self.simulate(p, w)

    def test_future_targets_do_not_change_past_equity(self):
        idx = pd.bdate_range('2024-01-02', periods=5)
        p = pd.DataFrame({'A': [100., 110., 120., 130., 140.]}, index=idx)
        w = pd.DataFrame({'A': [1., 1., 1., 0., 0.]}, index=idx)
        full = self.simulate(p, w).equity_curve
        prefix = self.simulate(p.iloc[:3], w.iloc[:3]).equity_curve
        pd.testing.assert_series_equal(full.iloc[:3], prefix)

    def test_strategy_to_engine_integration(self):
        import pandas_market_calendars as mcal
        idx = mcal.get_calendar('NYSE').schedule('2019-01-01', '2021-08-05').index
        p = pd.DataFrame({t: np.linspace(100., 120., len(idx))
                          for t in ['QQQ', 'IWD', 'IAU', 'IEF', 'SGOV', 'SPY']}, index=idx)
        months = pd.date_range('2017-01-01', '2021-07-01', freq='MS')
        v = pd.DataFrame({'date': months, 'realtime_start': months + pd.offsets.MonthBegin(1), 'value': 4.})
        for module in [laaMA4, laaMA_sandbox]:
            with patch.object(module, 'load_unemployment_vintages', return_value=v):
                with patch.object(module, 'load_close_for_ma', return_value=p.QQQ):
                    w = module.get_weights(p)
            np.testing.assert_allclose(w.sum(axis=1), 1.)
            result = self.simulate(p, w)
            self.assertTrue(np.isfinite(result.equity_curve).all())
            self.assertGreater(result.equity_curve.iloc[-1], 100.)


if __name__ == '__main__':
    unittest.main()
