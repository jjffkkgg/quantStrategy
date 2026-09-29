"""Offline regression coverage for every CLI strategy and sparse targets."""
import importlib
import io
import unittest
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import runBacktest as runner
import runStrategy as signal_runner
from utils.backtest import defer_pre_history_allocations, run_backtest


class StrategyCompatibilityTests(unittest.TestCase):
    SIGNAL_NAMES = {'LAA', 'DM', 'LAA_DM', 'LAA_MA', 'LAA_MA2', 'MA2',
                    'LAA_MA2_F', 'LAA_MA3', 'LAA_MA4',
                    'DM_RP', 'HAA', 'LAA_SANDBOX'}
    @classmethod
    def setUpClass(cls):
        cls.index = mcal.get_calendar('NYSE').schedule('2019-01-01', '2021-08-05').index
        cls.names = ['LAA', 'LAA2', 'SP500_MA', 'SP500', 'SP500MA', 'DM',
                     'LAA_DM', 'LAA_MA', 'MA2', 'LAA_MA2', 'LAA_MA2F',
                     'LAA_MA3', 'LAA_MA4', 'DM_RP', 'HAA', 'LAA_SANDBOX']
        tickers = sorted({t for name in cls.names for t in runner.get_tickers_for_strategy(name)})
        cls.prices = pd.DataFrame({t: np.linspace(100., 150. + i, len(cls.index))
                                   for i, t in enumerate(tickers)}, index=cls.index)
        months = pd.date_range('2017-01-01', '2021-07-01', freq='MS')
        cls.unrate = pd.Series(4., index=months)
        cls.vintages = pd.DataFrame({'date': months,
                                    'realtime_start': months + pd.offsets.MonthBegin(1),
                                    'value': 4.})

    def mock_sources(self, stack, prices):
        for name in ['laa', 'laa2', 'laaMA', 'ma2', 'laaMA2', 'laaMA2F',
                     'laaMA3', 'laaMA4', 'laaMA_sandbox']:
            module = importlib.import_module('strategies.' + name)
            if hasattr(module, 'load_unemployment_rate'):
                stack.enter_context(patch.object(module, 'load_unemployment_rate', return_value=self.unrate))
            if hasattr(module, 'load_unemployment_vintages'):
                stack.enter_context(patch.object(module, 'load_unemployment_vintages', return_value=self.vintages))
            if hasattr(module, 'load_close_for_ma'):
                stack.enter_context(patch.object(module, 'load_close_for_ma', side_effect=lambda ticker, **kw: prices[ticker]))
        stack.enter_context(patch.object(runner, 'load_close_for_ma', side_effect=lambda ticker, **kw: prices[ticker]))
        stack.enter_context(patch('strategies.haa._load_tips_yield_data', return_value=pd.Series(2., index=self.index)))

    def test_every_cli_strategy_with_pre_history_and_missing_execution(self):
        prices = self.prices.copy()
        prices.loc[self.index[:300], 'IAU'] = np.nan
        prices.loc[self.index[:40], 'QQQ'] = np.nan
        prices.loc['2020-02-03', :] = np.nan  # missing first execution after January month-end
        prices['SGOV'] = 100.
        for name in self.names:
            with self.subTest(strategy=name), ExitStack() as stack, redirect_stdout(io.StringIO()):
                self.mock_sources(stack, prices)
                stack.enter_context(patch.object(runner, 'load_prices', return_value=prices.copy()))
                stack.enter_context(patch('sys.argv', ['runBacktest.py', name]))
                stack.enter_context(patch.object(runner.os, 'makedirs'))
                stack.enter_context(patch.object(pd.DataFrame, 'to_csv'))
                stack.enter_context(patch.object(pd.Series, 'to_csv'))
                if name == 'LAA_SANDBOX':
                    from unittest.mock import mock_open
                    stack.enter_context(patch('builtins.open', mock_open()))
                stack.enter_context(patch('utils.backtest._calc_real_cagr', return_value=np.nan))
                results = []

                def simulate(**kwargs):
                    result = run_backtest(**kwargs)
                    results.append(result)
                    return result

                stack.enter_context(patch.object(runner, 'run_backtest', side_effect=simulate))
                runner.main()
                result, = results
                self.assertTrue(np.isfinite(result.equity_curve).all())
                self.assertGreater(result.equity_curve.iloc[-1], result.equity_curve.iloc[0])
                self.assertFalse(result.trade_log.empty)
                self.assertNotIn(pd.Timestamp('2020-02-03'), result.trade_log.index)
                self.assertEqual('sleeve' in result.trade_log, name in ('LAA_MA4', 'LAA_SANDBOX'))

    def test_signal_cli_runs_all_existing_strategies(self):
        output = io.StringIO()
        with ExitStack() as stack, redirect_stdout(output):
            self.mock_sources(stack, self.prices)
            stack.enter_context(patch.object(signal_runner, 'load_prices', return_value=self.prices.copy()))
            displayed = stack.enter_context(patch.object(signal_runner, 'print_weight_result'))
            signal_runner.main()
        results = {call.args[0]: call.args[1] for call in displayed.call_args_list}
        self.assertEqual(set(results), self.SIGNAL_NAMES)
        self.assertEqual(len(displayed.call_args_list), len(self.SIGNAL_NAMES))
        for name, result in results.items():
            with self.subTest(strategy=name):
                self.assertIsInstance(result, dict, msg=str(result))
                values = np.array(list(result.values()), dtype=float)
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((values >= 0).all())
                self.assertAlmostEqual(values.sum(), 1.)
        self.assertIn('S&P MA    ->', output.getvalue())
        self.assertNotIn('Error:', output.getvalue())

    def test_new_signal_failure_does_not_block_legacy_output(self):
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            self.mock_sources(stack, self.prices)
            stack.enter_context(patch.object(signal_runner, 'load_prices', return_value=self.prices.copy()))
            stack.enter_context(patch.object(signal_runner, 'laa_sandbox_signal',
                                             side_effect=RuntimeError('test-only variant failure')))
            displayed = stack.enter_context(patch.object(signal_runner, 'print_weight_result'))
            signal_runner.main()
        results = {call.args[0]: call.args[1] for call in displayed.call_args_list}
        self.assertEqual(set(results), self.SIGNAL_NAMES)
        self.assertIn('test-only variant failure', results['LAA_SANDBOX'])
        for name in self.SIGNAL_NAMES - {'LAA_SANDBOX'}:
            self.assertIsInstance(results[name], dict, msg=f'{name}: {results[name]}')

    def test_sparse_weekend_targets_enter_after_first_observed_price(self):
        idx = pd.to_datetime(['2021-01-29', '2021-02-01', '2021-02-02', '2021-02-03', '2021-02-04'])
        prices = pd.DataFrame({'A': [np.nan, np.nan, 100., 110., 121.], 'SGOV': 100.}, index=idx)
        weights = pd.DataFrame({'A': [1.], 'SGOV': [0.]}, index=pd.to_datetime(['2021-01-31']))
        original = weights.copy()
        adjusted = defer_pre_history_allocations(prices, weights)
        pd.testing.assert_frame_equal(weights, original)
        np.testing.assert_allclose(adjusted.A, [0., 0., 1., 1., 1.])
        np.testing.assert_allclose(adjusted.SGOV, [0., 1., 0., 0., 0.])
        result = run_backtest(prices, adjusted, initial_capital=100., calculate_real_cagr=False)
        buys = result.trade_log.query("ticker == 'A'")
        self.assertEqual(list(buys.index), [pd.Timestamp('2021-02-03')])
        self.assertAlmostEqual(result.equity_curve.iloc[-1], 110.)

    def test_monthly_strategies_include_weekend_and_holiday_month_ends(self):
        for name in ['LAA', 'LAA2', 'LAA_DM', 'DM', 'DM_RP', 'HAA']:
            with self.subTest(strategy=name), ExitStack() as stack:
                self.mock_sources(stack, self.prices)
                prices = self.prices[runner.get_tickers_for_strategy(name)]
                weights = runner.get_strategy_weights(name, prices)
                self.assertIn(pd.Timestamp('2021-07-30'), weights.index)
                self.assertIn(pd.Timestamp('2021-05-28'), weights.index)
                self.assertLessEqual(weights.index.max(), prices.index.max())

    def test_legacy_regime_keeps_weekend_month_end_signal(self):
        prices = self.prices.copy()
        prices['SPY'] = np.linspace(200., 100., len(prices))
        unrate = self.unrate.copy()
        unrate.iloc[-1] = 8.
        for name in ['laaMA2', 'laaMA2F', 'laaMA3']:
            module = importlib.import_module('strategies.' + name)
            with self.subTest(strategy=name), patch.object(module, 'load_unemployment_rate', return_value=unrate):
                flags = module._compute_regime_flags(prices)
                self.assertFalse(flags.loc['2021-06-30', 'recession'])
                self.assertTrue(flags.loc['2021-07-30', 'recession'])
                self.assertTrue(flags.loc['2021-08-02', 'recession'])
