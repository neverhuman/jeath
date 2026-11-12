"""Fetch or reuse an aggregate-only, metadata-only frozen H-benchmark baseline."""
import hashlib
import json
from pathlib import Path
import subprocess


def check(data):
    assert data['split_counts']=={'train':8000,'validation':1000,'test':1000}
    assert data['manifest_stat_unchanged'] and data['eligibility']['duplicate_groups_disjoint_across_splits']
    for row in data['matching_frozen_wave_metrics'].values():
        assert row['test_n_match'] and row['target_variance_match'] and row['sampled_metadata_matches']
        assert row['sampled_rows_checked']>0 and not row['full_prediction_pairing_available']
    for model in data['models'].values():
        assert model['test_metrics']['n']==1000
        assert model['test_metrics']['r2']<=1  # Predictive R2 has no finite lower bound.
    return True


def analyze(root=None,frozen_path=None):
    """Return aggregates; pass frozen_path for local replay without any remote call."""
    if frozen_path is not None:
        data=json.loads(Path(frozen_path).read_text());check(data);return data
    path=Path(__file__).with_name('minds_demographic_remote.py');program=path.read_bytes()
    command=['ssh','-o','BatchMode=yes','xbabe3','cd /home/ubuntu/HON_INDEX && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .review-venv/bin/python -']
    result=subprocess.run(command,input=program,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-4000:])
    data=json.loads(result.stdout);data['remote_script_sha256']=hashlib.sha256(program).hexdigest();check(data);return data
