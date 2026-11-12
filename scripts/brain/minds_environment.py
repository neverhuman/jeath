"""Public fossil and climate context; chronological alignment supplies no causal estimate."""
import hashlib
from io import StringIO
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('project/brain/tips/Performance_Before_Outcomes_v9')
REGISTER = Path('evidence/brain/environment-events.json')
WORKBOOKS = {
    'gingerich': 'gingerich2022/41598_2022_15481_MOESM1_ESM.xlsx',
    'du': 'du2018/du2018_ecv.xlsx',
    'will': 'will2021/41467_2021_24290_MOESM5_ESM.xlsx',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def workbook_rows(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return list(book.worksheets[0].values)
    finally:
        book.close()


def load_fossils(root=ROOT):
    records = {}
    for name, relative in WORKBOOKS.items():
        rows = workbook_rows(root / ARCHIVE / 'data' / relative)
        result = []
        for index, row in enumerate(rows, 1):
            if not row or not row[0]:
                continue
            if name == 'gingerich':
                age, volume = number(row[2]), number(row[3])
                extra = {'case_studies': number(row[1]), 'dating_bounds_ma': None}
            elif name == 'will':
                age, volume = number(row[1]), number(row[5])
                extra = {'taxon': str(row[4]), 'dating_bounds_ma': None,
                         'dating_method': str(row[8]), 'dating_source_verbatim': str(row[9])}
            else:
                data = dict(zip(rows[0], row))
                values = [number(data[f'ecv{i}']) for i in range(1, 7)]
                values = [x for x in values if x is not None]
                lo, hi, age = [number(data[k]) for k in ('min.date', 'max.date', 'mean.date')]
                if age is None and lo is not None and hi is not None:
                    age = (lo + hi) / 2
                volume = float(np.mean(values)) if values else None
                extra = {'taxon': str(data['lump.taxon']), 'region': str(data['region']),
                         'dating_bounds_ma': [lo, hi] if lo is not None and hi is not None else None,
                         'dating_sd_ma': number(data['sd.date']), 'date_distribution': str(data['date.dist']),
                         'dating_source_verbatim': str(data['date.ref'])}
            if age is None or volume is None:
                continue
            age = abs(age) if name == 'gingerich' else age
            if age < 0 or volume <= 0:
                raise ValueError(f'Invalid fossil age/volume at {name} row {index}')
            result.append({'specimen': str(row[0]).strip(), 'source_row': index,
                           'age_ma': age, 'cranial_volume_cm3': volume, **extra})
        records[name] = result
    return records


def bin_index(age):
    # Preserve the archived exp_rates.e13 binary floating-point convention exactly.
    return int(np.floor(age / 0.1))


def bins(records, maximum_index=None):
    groups = {}
    for row in records:
        age, volume = row['age_ma'], row['cranial_volume_cm3']
        if not np.isfinite(age) or age < 0 or not np.isfinite(volume) or volume <= 0:
            raise ValueError('Bins require finite nonnegative ages and positive volumes')
        groups.setdefault(bin_index(age), []).append(volume)
    stop = max(groups, default=-1) if maximum_index is None else maximum_index
    if groups and stop < max(groups):
        raise ValueError('Bin range drops observations')
    return [{'index': i, 'younger_ma': i * .1, 'older_ma': (i + 1) * .1,
             'midpoint_ma': (i + .5) * .1, 'n': len(groups.get(i, [])),
             'geometric_mean_cm3': float(np.exp(np.log(groups[i]).mean())) if i in groups else None}
            for i in range(stop + 1)]


def load_climate(raw):
    data = np.loadtxt(StringIO(raw.decode('ascii')), skiprows=5)
    if data.ndim != 2 or data.shape[1] != 3 or not np.isfinite(data).all():
        raise ValueError('Malformed LR04 data')
    if not (np.diff(data[:, 0]) > 0).all() or (data[:, 2] < 0).any():
        raise ValueError('LR04 timestamps must be unique and ordered; errors nonnegative')
    if data[0, 0] != 0 or data[-1, 0] != 5320:
        raise ValueError('Unexpected LR04 documented range')
    return [{'age_ka': float(t), 'age_ma': float(t / 1000), 'benthic_d18o_per_mil': float(v),
             'standard_error_per_mil': float(e)} for t, v, e in data]


def check_events(events):
    if len({e['id'] for e in events}) != len(events):
        raise ValueError('Duplicate environmental event')
    for e in events:
        if not all(e[k] for k in ('geography', 'evidence_status', 'date_meaning', 'citations', 'primary_dois', 'interpretation_limit')):
            raise ValueError('Event lacks scope, evidence status, dates or primary citation')
        if len(e['citations']) != len(e['primary_dois']):
            raise ValueError('Every event citation requires a DOI')
        dates = [e['age_ma']] if e['kind'] == 'point' else [e['younger_ma'], e['older_ma']]
        if not all(np.isfinite(d) and 0 <= d <= 5.32 for d in dates):
            raise ValueError('Invalid event chronology')
        if e['kind'] not in ('point', 'interval') or (e['kind'] == 'interval' and dates[0] >= dates[1]):
            raise ValueError('Malformed event duration')
        if e['id'] == 'bottleneck' and not all('disputed' in e[k].lower() for k in ('label', 'evidence_status')):
            raise ValueError('Bottleneck must carry its disputed label')


def analyze(inputs, root=ROOT):
    register = json.loads((root / REGISTER).read_text())
    check_events(register['events'])
    source = register['climate_source']
    inputs = Path(inputs) / 'environment'
    inputs.mkdir(parents=True, exist_ok=True)
    path = inputs / source['file']
    if not path.exists():
        with urlopen(source['url'], timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('Downloaded LR04 hash differs from pinned version')
        path.write_bytes(raw)
    if digest(path) != source['sha256']:
        raise ValueError('Cached LR04 hash differs from pinned version')
    fossils = load_fossils(root)
    fossil_bins = {k: bins(v) for k, v in fossils.items()}
    archived_path = root / ARCHIVE / 'results/e13_fossils.json'
    archived = json.loads(archived_path.read_text())
    occupied = [r for r in fossil_bins['gingerich'] if r['n']]
    if len(fossils['gingerich']) != archived['n'] or archived['n'] != 233:
        raise ValueError('Consensus specimen accounting differs from archived 233')
    for key, calculated in [('n', [r['n'] for r in occupied]), ('t', [r['midpoint_ma'] for r in occupied]),
                            ('mean', [r['geometric_mean_cm3'] for r in occupied])]:
        if not np.allclose(calculated, archived['bins'][key], rtol=1e-12, atol=1e-12):
            raise ValueError('Archived bin mismatch: ' + key)
    sources = {str(ARCHIVE / 'data' / p): digest(root / ARCHIVE / 'data' / p) for p in WORKBOOKS.values()}
    for p in ['results/e13_fossils.json', 'code/load_fossils.py', 'code/exp_rates.py', 'code/common.py']:
        sources[str(ARCHIVE / p)] = digest(root / ARCHIVE / p)
    data = {'schema_version': 1, 'time_unit': 'Ma before present', 'display_range_ma': [1.7, .5],
            'fossils': fossils, 'bins': fossil_bins, 'counts': {k: len(v) for k, v in fossils.items()},
            'displayed_consensus_n': sum(.5 <= r['age_ma'] <= 1.7 for r in fossils['gingerich']),
            'bin_definition': 'floor(age_ma / 0.1), identical binary floating-point operation to archived exp_rates.e13; empty bins retained as null',
            'climate': load_climate(path.read_bytes()), 'climate_source': source,
            'published_stasis': register['published_stasis'], 'events': register['events'],
            'source_sha256': sources, 'register_sha256': digest(root / REGISTER),
            'archived_models': {'models': archived['models'], 'source_result_sha256': digest(archived_path),
                'specification': 'Archived exp_rates.e13: OLS log ECV on -age; free 1/2/3-breakpoint continuous models on 0.05-Myr grid; stasis_step fixes slopes in phases I and III to zero; published_phases fixes boundaries at 2.0,1.5,0.7 Ma. Parameter penalties retain original code. No event-aligned fits or new causal tests.',
                'seed': archived['seed'], 'bootstrap_replicates': archived['nboot']},
            'limits': ['Fossil compilations overlap and cannot be pooled as independent specimens.',
                'Consensus ages have no supplied specimen dating errors; none are invented.',
                'Du date bounds and distributions are source fields, not uniform confidence intervals; Will dating notes remain verbatim.',
                'Chronological overlap is not a causal estimate; disruptions postdate the published phase onset.']}
    check(data)
    return data


def check(data):
    assert data['schema_version'] == 1 and data['counts']['gingerich'] == 233
    assert data['climate_source']['records_averaged'] == 57
    assert data['climate_source']['sha256'] == '7d1a6ca7ffc033e45e20fb76a7f39c8d99df87baa1094260f9dbdc09f0bb8945'
    for name, records in data['fossils'].items():
        assert len(records) == data['counts'][name]
        assert len({r['source_row'] for r in records}) == len(records)
        assert bins(records) == data['bins'][name]
        assert sum(r['n'] for r in data['bins'][name]) == len(records)
    assert all(r['dating_bounds_ma'] is None for r in data['fossils']['gingerich'])
    assert data['displayed_consensus_n'] == sum(.5 <= r['age_ma'] <= 1.7 for r in data['fossils']['gingerich'])
    c = data['climate']
    assert c[0]['age_ka'] == 0 and c[-1]['age_ka'] == 5320
    assert all(a['age_ka'] < b['age_ka'] for a, b in zip(c, c[1:]))
    assert all(np.isfinite(list(r.values())).all() and r['standard_error_per_mil'] >= 0 and r['age_ma'] == r['age_ka']/1000 for r in c)
    check_events(data['events'])
    return True
