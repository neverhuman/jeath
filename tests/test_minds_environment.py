"""Environmental source accounting, chronology semantics and human-only replay routing."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import openpyxl  # noqa: F401
    import matplotlib  # noqa: F401
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Environmental human-study fixtures require NumPy, openpyxl, and Matplotlib") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from brain import minds_environment as environment
from brain.minds_paper import bibliography_entries, bibliography
import run as entrypoint


class EnvironmentTests(unittest.TestCase):
    def test_exact_boundaries_and_visible_missing_bins(self):
        def row(age, volume): return {'age_ma': age, 'cranial_volume_cm3': volume}
        records = [row(0, 100), row(np.nextafter(.1, 0), 400), row(.1, 900),
                   row(np.nextafter(.1, 1), 1600), row(.4, 2500), row(.3, 3600)]
        bins = environment.bins(records)
        self.assertEqual([r['n'] for r in bins], [2, 2, 1, 0, 1])
        self.assertAlmostEqual(bins[0]['geometric_mean_cm3'], 200)
        self.assertIsNone(bins[3]['geometric_mean_cm3'])
        # Deliberately retained archived floating-point convention: .3/.1 < 3.
        self.assertEqual(environment.bin_index(.3), 2)
        self.assertEqual(sum(r['n'] for r in bins), len(records))
        for age, volume in [(np.nan, 100), (-1, 100), (1, 0), (1, np.inf)]:
            with self.assertRaises(ValueError): environment.bins([row(age, volume)])

    def test_missing_cells_do_not_create_specimens_or_dating_errors(self):
        rows = [('title', None, None, None), ('valid', 2, -1.1, 850),
                ('missing age', 1, None, 900), ('missing volume', 1, 1.2, None),
                ('nan', 1, 'NaN', 1000)]
        # No temporary writes in third-party archive folders.
        with patch.object(environment, 'workbook_rows', side_effect=[rows, [], []]):
            fossils = environment.load_fossils()
        self.assertEqual(len(fossils['gingerich']), 1)
        self.assertEqual(fossils['gingerich'][0]['age_ma'], 1.1)
        self.assertIsNone(fossils['gingerich'][0]['dating_bounds_ma'])
        self.assertIsNone(environment.number('NA'))
        self.assertIsNone(environment.number(np.inf))

    def test_climate_chronology_rejects_duplicates_reversal_nan_and_wrong_range(self):
        header = b'citation\nsource\nversion\n\ntime value se\n'
        good = header + b'0 3.23 .03\n1000 4.1 .02\n5320 2.91 .09\n'
        rows = environment.load_climate(good)
        self.assertEqual(rows[1]['age_ma'], 1)
        self.assertEqual(rows[-1]['age_ma'], 5.32)
        for bad in [good.replace(b'1000', b'0'), good.replace(b'1000', b'6000'),
                    good.replace(b'4.1', b'nan'), good.replace(b'5320', b'5315'),
                    good.replace(b'.02', b'-.02')]:
            with self.assertRaises(ValueError): environment.load_climate(bad)

    def test_events_require_scope_status_citation_and_dispute_label(self):
        events = json.loads((ROOT / environment.REGISTER).read_text())['events']
        environment.check_events(events)
        for field, value in [('label', 'Ancestral bottleneck'), ('evidence_status', 'established'),
                             ('geography', ''), ('citations', []), ('older_ma', .5)]:
            altered = copy.deepcopy(events)
            event = next(e for e in altered if e['id'] == 'bottleneck')
            event[field] = value
            with self.assertRaises(ValueError): environment.check_events(altered)
        for e in events:
            if e['kind'] == 'interval':
                self.assertIsNone(e['reported_uncertainty'])
                self.assertIn('not', e['date_meaning'])
        self.assertEqual(next(e for e in events if e['id']=='impact')['kind'], 'point')

    def test_archived_counts_and_geometric_means_reconcile(self):
        fossils = environment.load_fossils()
        self.assertEqual({k:len(v) for k,v in fossils.items()}, {'gingerich':233,'du':94,'will':166})
        archived = json.loads((ROOT / environment.ARCHIVE / 'results/e13_fossils.json').read_text())
        bins = [r for r in environment.bins(fossils['gingerich']) if r['n']]
        self.assertEqual([r['n'] for r in bins], archived['bins']['n'])
        np.testing.assert_allclose([r['geometric_mean_cm3'] for r in bins], archived['bins']['mean'], rtol=1e-12)
        self.assertEqual(sum(.5 <= r['age_ma'] <= 1.7 for r in fossils['gingerich']),31)
        self.assertTrue(any(r['dating_bounds_ma'] for r in fossils['du']))
        self.assertTrue(all(r['dating_bounds_ma'] is None for r in fossils['will']))

    def test_bibliography_conflicts_and_all_event_citations_resolve(self):
        def item(body): return '\\bibitem{key}'+body+'\\end{thebibliography}'
        first = item('Author. Title. 2026. \\doi{10.1/a}.')
        self.assertEqual(len(bibliography_entries([first,first])),1)
        for second in [item('Author. Other title. 2026. \\doi{10.1/a}.'), item('Other. \\doi{10.2/b}.')]:
            with self.assertRaisesRegex(ValueError, 'Conflicting bibliography key'):
                bibliography_entries([first, second])
        rendered = bibliography()
        for event in json.loads((ROOT / environment.REGISTER).read_text())['events']:
            for citation in event['citations']:
                self.assertIn('\\bibitem{'+citation+'}', rendered)

    def test_missing_manuscript_citation_is_rejected(self):
        read_text = Path.read_text
        def source(path, *args, **kwargs):
            value = read_text(path, *args, **kwargs)
            if path.name == 'minds-environment.tex':
                value += r"\citep{missing_primary_source_fixture}"
            return value
        with patch.object(Path, 'read_text', source):
            with self.assertRaisesRegex(ValueError, 'Missing primary bibliography'):
                bibliography()

    def test_cached_climate_corruption_is_rejected_before_analysis(self):
        import tempfile
        directory = ROOT / 'target/minds-we-choose'
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as name:
            folder = Path(name) / 'environment'
            folder.mkdir()
            (folder / 'LR04stack.txt').write_bytes(b'corrupted fixture')
            with self.assertRaisesRegex(ValueError, 'Cached LR04 hash'):
                environment.analyze(Path(name))

    def test_replay_flags_are_human_only_and_forwarded(self):
        for study in [[], ['--study','performance-before-outcomes']]:
            result = subprocess.run([sys.executable,str(ROOT/'scripts/run.py'),'target/should-not-exist',
                                     '--remote-snapshot','missing.json',*study],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('require --study minds-we-choose',result.stderr)
        with patch.object(sys, 'argv', ['run.py','target/replay','--study','minds-we-choose',
                                      '--remote-snapshot','snapshot.json','--demographic-frozen','baseline.json']), \
             patch('brain.minds_study.run') as runner:
            entrypoint.main()
        args=runner.call_args.args
        self.assertEqual(args[0],Path('target/replay'))
        self.assertEqual(args[2:],(Path('snapshot.json'),Path('baseline.json')))


if __name__ == '__main__': unittest.main()
