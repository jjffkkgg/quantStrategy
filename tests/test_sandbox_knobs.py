import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from strategies import laaMA_sandbox as sandbox, laaMA4 as baseline


class SandboxKnobTests(unittest.TestCase):
    def setUp(self):
        self.idx = mcal.get_calendar('NYSE').schedule('2019-01-01', '2021-08-05').index
        self.p = pd.DataFrame(100., index=self.idx, columns=['QQQ', 'IWD', 'IAU', 'IEF', 'SGOV', 'SPY'])

    def sources(self, stack, rec=False, up=True):
        flags = pd.DataFrame({'recession': rec, 'uptrend': up}, index=self.idx)
        stack.enter_context(patch.object(sandbox, '_compute_regime_flags', return_value=flags))
        stack.enter_context(patch.object(sandbox, 'load_close_for_ma', return_value=self.p.QQQ))

    def test_value_switch_and_fraction(self):
        for enabled in [False, True]:
            for rec, up in [(False, True), (True, True), (False, False), (True, False)]:
                with self.subTest(enabled=enabled, rec=rec, up=up), ExitStack() as stack:
                    self.sources(stack, rec, up)
                    stack.enter_context(patch.object(sandbox, 'VALUE_STAGED_DEFENSE', enabled))
                    stack.enter_context(patch.object(sandbox, 'VALUE_CAUTION_FRACTION', .4))
                    w = sandbox.get_weights(self.p)
                    fraction = 1. if up else (0. if rec else (.4 if enabled else 1.))
                    np.testing.assert_allclose(w.IWD, sandbox.W_VALUE * fraction)
                    np.testing.assert_allclose(w.sum(axis=1), 1.)

    def test_gold_modes_all_condition_combinations(self):
        for mode in ['AND', 'GOLD_ONLY', 'SPLIT', 'HOLD', 'BOTH_NEGATIVE']:
            for gold, bond in [(False, False), (True, False), (False, True), (True, True)]:
                p = self.p.copy()
                p.IAU = np.linspace(100., 150. if gold else 50., len(p))
                p.IEF = np.linspace(100., 150. if bond else 50., len(p))
                with self.subTest(mode=mode, gold=gold, bond=bond), ExitStack() as stack:
                    self.sources(stack)
                    stack.enter_context(patch.object(sandbox, 'GOLD_STRATEGY', mode))
                    stack.enter_context(patch.object(sandbox, 'GOLD_PRICE_SHARE', .3))
                    w = sandbox.get_weights(p)
                fraction = {'AND': float(gold and bond), 'GOLD_ONLY': float(gold),
                            'SPLIT': .3 * gold + .7 * bond, 'HOLD': 1.,
                            'BOTH_NEGATIVE': float(gold or bond)}[mode]
                self.assertAlmostEqual(w.IAU.iloc[-1], sandbox.W_GOLD * fraction)
                np.testing.assert_allclose(w.sum(axis=1), 1.)
                self.assertTrue(w.ge(-1e-12).all().all())

    def test_disabled_experiments_match_ma4(self):
        with ExitStack() as stack:
            self.sources(stack)
            stack.enter_context(patch.object(baseline, '_compute_regime_flags', return_value=pd.DataFrame(
                {'recession': False, 'uptrend': True}, index=self.idx)))
            stack.enter_context(patch.object(baseline, 'load_close_for_ma', return_value=self.p.QQQ))
            for name, value in dict(W_AGGRESSIVE=.25, W_GOLD=.25, AGGRESSIVE_COOLDOWN_DAYS=30,
                                    VALUE_STAGED_DEFENSE=False, GOLD_STRATEGY='AND').items():
                stack.enter_context(patch.object(sandbox, name, value))
            pd.testing.assert_frame_equal(sandbox.get_weights(self.p), baseline.get_weights(self.p))

    def test_gold_monthly_only_zero_missing_and_prefix(self):
        p = self.p.copy()
        p.IAU = np.linspace(100., 200., len(p))
        p.loc['2021-08-02':, 'IAU'] = 1.
        with ExitStack() as stack:
            self.sources(stack)
            stack.enter_context(patch.object(sandbox, 'VALUE_STRATEGY', 'HOLD'))
            stack.enter_context(patch.object(sandbox, 'GOLD_STRATEGY', 'SPLIT'))
            stack.enter_context(patch.object(sandbox, 'GOLD_PRICE_SHARE', .5))
            full = sandbox.get_weights(p)
            prefix = sandbox.get_weights(p.loc[:'2021-07-30'])
            self.assertAlmostEqual(full.IAU.iloc[-1], sandbox.W_GOLD * .5)
            pd.testing.assert_frame_equal(full.loc[:'2021-07-30'], prefix)
            self.assertTrue(full.IAU.iloc[:252].eq(0.).all())
            p.IAU = np.nan
            self.assertTrue(sandbox.get_weights(p).IAU.eq(0.).all())

    def test_invalid_knobs_fail_explicitly(self):
        for name, value in [('GOLD_STRATEGY', 'typo'), ('GOLD_PRICE_SHARE', 1.1),
                            ('VALUE_CAUTION_FRACTION', -.1)]:
            with patch.object(sandbox, name, value), self.assertRaises(ValueError):
                sandbox.get_weights(self.p)

    def test_both_negative_zero_boundary_and_missing_history(self):
        for gold_end, bond_end, expected in [(100., 50., 1.), (50., 100., 1.),
                                             (100., 100., 1.), (50., 50., 0.),
                                             (np.nan, 150., 0.), (150., np.nan, 0.)]:
            p = self.p.copy()
            p.IAU = np.linspace(100., gold_end, len(p))
            p.IEF = np.linspace(100., bond_end, len(p))
            with self.subTest(gold=gold_end, bond=bond_end), ExitStack() as stack:
                self.sources(stack)
                stack.enter_context(patch.object(sandbox, 'GOLD_STRATEGY', 'BOTH_NEGATIVE'))
                w = sandbox.get_weights(p)
            self.assertAlmostEqual(w.IAU.iloc[-1], sandbox.W_GOLD * expected)
            self.assertTrue(w.IAU.iloc[:252].eq(0.).all())

    def test_both_negative_waits_for_month_end(self):
        p = self.p.copy()
        p.IAU = np.linspace(100., 150., len(p))
        p.IEF = np.linspace(100., 50., len(p))
        p.loc['2021-08-02':, 'IAU'] = 1.
        with ExitStack() as stack:
            self.sources(stack)
            stack.enter_context(patch.object(sandbox, 'VALUE_STRATEGY', 'HOLD'))
            stack.enter_context(patch.object(sandbox, 'GOLD_STRATEGY', 'BOTH_NEGATIVE'))
            full = sandbox.get_weights(p)
            prefix = sandbox.get_weights(p.loc[:'2021-07-30'])
        self.assertEqual(full.IAU.iloc[-1], sandbox.W_GOLD)
        pd.testing.assert_frame_equal(full.loc[:'2021-07-30'], prefix)
