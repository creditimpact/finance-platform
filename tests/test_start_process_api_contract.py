import io
import json
import re
from pathlib import Path

from jsonschema import Draft7Validator

from backend.api.app import create_app
from backend.api import app as app_module
import backend.core.orchestrators as orch
import backend.config as config

SCHEMA = Draft7Validator(json.loads(Path('backend/schemas/problem_account.json').read_text()))


def _setup_app(monkeypatch, accounts, meta_map):
    class DummyResult:
        def get(self, timeout=None):
            return {}

    class DummyTask:
        def delay(self, *a, **k):
            return DummyResult()

    monkeypatch.setattr(app_module, 'extract_problematic_accounts', DummyTask())
    monkeypatch.setattr(app_module, 'run_credit_repair_process', lambda *a, **k: None)
    monkeypatch.setattr(app_module, 'set_session', lambda *a, **k: None)
    monkeypatch.setattr(app_module.cs_api, 'load_session_case', lambda sid: None)
    monkeypatch.setattr(orch, 'collect_stageA_logical_accounts', lambda sid: accounts)
    monkeypatch.setattr(orch, 'get_stageA_decision_meta', lambda sid, aid: meta_map.get(aid))
    return create_app()


def _post_start_process(app):
    client = app.test_client()
    data = { 'email': 'user@example.com', 'file': (io.BytesIO(b'%PDF-1.4'), 'report.pdf') }
    return client.post('/api/start-process', data=data, content_type='multipart/form-data')


def _validate_accounts(records):
    for acc in records:
        base = {k: v for k, v in acc.items() if k != 'decision_meta'}
        SCHEMA.validate(base)


def test_decision_meta_included(monkeypatch):
    monkeypatch.setattr(config, 'API_INCLUDE_DECISION_META', True)
    monkeypatch.setattr(config, 'API_DECISION_META_MAX_FIELDS_USED', 6)

    ai_acc = {
        'account_id': 'a1',
        'bureau': 'Equifax',
        'primary_issue': 'late_payment',
        'tier': 'Tier1',
        'problem_reasons': ['reason'],
        'confidence': 0.9,
        'decision_source': 'ai',
    }
    rules_acc = {
        'account_id': 'r1',
        'bureau': 'TransUnion',
        'primary_issue': 'unknown',
        'tier': 'none',
        'problem_reasons': ['past_due_amount: 125.00'],
        'confidence': 0.0,
        'decision_source': 'rules',
    }
    meta_map = {
        'a1': {
            'decision_source': 'ai',
            'confidence': 0.9,
            'tier': 'Tier1',
            'fields_used': ['f1','f2','f3','f4','f5','f6','f7'],
        },
        'r1': {
            'decision_source': 'rules',
            'confidence': 0.0,
            'tier': 'none',
            'fields_used': ['payment_status', 'balance']
        },
    }

    app = _setup_app(monkeypatch, [ai_acc, rules_acc], meta_map)
    resp = _post_start_process(app)
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    records = payload['accounts']['problem_accounts']
    assert {rec['account_id'] for rec in records} == {'a1','r1'}
    _validate_accounts(records)

    for rec in records:
        meta = rec.get('decision_meta')
        assert meta is not None
        assert {'decision_source','confidence','tier'} <= set(meta)
        if 'fields_used' in meta:
            assert len(meta['fields_used']) <= config.API_DECISION_META_MAX_FIELDS_USED

    ai_meta = next(r['decision_meta'] for r in records if r['account_id']=='a1')
    assert ai_meta['decision_source'] == 'ai'
    assert ai_meta['confidence'] > 0
    assert ai_meta['tier'] in {'Tier1','Tier2','Tier3'}
    assert ai_meta['fields_used'] == meta_map['a1']['fields_used'][:6]

    rules_meta = next(r['decision_meta'] for r in records if r['account_id']=='r1')
    assert rules_meta['decision_source'] == 'rules'
    assert rules_meta['confidence'] == 0.0
    assert rules_meta['tier'] == 'none'

    text = json.dumps(payload)
    assert not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", text)
    assert not re.search(r"\b\d{3}-\d{3}-\d{4}\b", text)
    assert not re.search(r"\b\d{9,}\b", text)


def test_decision_meta_excluded_when_flag_off(monkeypatch):
    monkeypatch.setattr(config, 'API_INCLUDE_DECISION_META', False)
    ai_acc = {
        'account_id': 'a1',
        'bureau': 'Equifax',
        'primary_issue': 'late_payment',
        'tier': 'Tier1',
        'problem_reasons': ['reason'],
        'confidence': 0.9,
        'decision_source': 'ai',
    }
    rules_acc = {
        'account_id': 'r1',
        'bureau': 'TransUnion',
        'primary_issue': 'unknown',
        'tier': 'none',
        'problem_reasons': ['past_due_amount: 125.00'],
        'confidence': 0.0,
        'decision_source': 'rules',
    }
    meta_map = {
        'a1': {'decision_source': 'ai', 'confidence': 0.9, 'tier': 'Tier1'},
        'r1': {'decision_source': 'rules', 'confidence': 0.0, 'tier': 'none'},
    }
    app = _setup_app(monkeypatch, [ai_acc, rules_acc], meta_map)
    resp = _post_start_process(app)
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    records = payload['accounts']['problem_accounts']
    assert all('decision_meta' not in rec for rec in records)
    _validate_accounts(records)


def test_decision_meta_omits_fields_used(monkeypatch):
    monkeypatch.setattr(config, 'API_INCLUDE_DECISION_META', True)
    monkeypatch.setattr(config, 'API_DECISION_META_MAX_FIELDS_USED', 6)
    ai_acc = {
        'account_id': 'a1',
        'bureau': 'Equifax',
        'primary_issue': 'late_payment',
        'tier': 'Tier2',
        'problem_reasons': ['reason'],
        'confidence': 0.8,
        'decision_source': 'ai',
    }
    meta_map = {'a1': {'decision_source': 'ai', 'confidence': 0.8, 'tier': 'Tier2'}}
    app = _setup_app(monkeypatch, [ai_acc], meta_map)
    resp = _post_start_process(app)
    assert resp.status_code == 200
    payload = json.loads(resp.data)
    records = payload['accounts']['problem_accounts']
    _validate_accounts(records)
    meta = records[0]['decision_meta']
    assert 'fields_used' not in meta
