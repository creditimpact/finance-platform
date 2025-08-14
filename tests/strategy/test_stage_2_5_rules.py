import pytest
from backend.policy.policy_loader import load_rulebook
from backend.core.logic.strategy.normalizer_2_5 import normalize_and_tag


@pytest.fixture
def rulebook():
    return load_rulebook()

def test_identity_theft_without_affidavit(rulebook):
    facts = {'identity_theft': True, 'has_id_theft_affidavit': False}
    result = normalize_and_tag({}, facts, rulebook)
    assert result['rule_hits'] == ['E_IDENTITY', 'E_IDENTITY_NEEDS_AFFIDAVIT']
    assert result['needs_evidence'] == ['identity_theft_affidavit']
    assert result['suggested_dispute_frame'] == 'fraud'

def test_collection_with_admission(rulebook):
    account_cls = {'user_statement_raw': 'my fault'}
    facts = {'type': 'collection', 'is_open_revolving': True, 'utilization': 0.95}
    result = normalize_and_tag(account_cls, facts, rulebook)
    assert result['red_flags'] == ['admission_of_fault']
    assert result['suggested_dispute_frame'] == 'verification'

def test_high_utilization_paydown_first(rulebook):
    facts = {'is_open_revolving': True, 'utilization': 0.95}
    result = normalize_and_tag({}, facts, rulebook)
    assert result['rule_hits'] == ['K_UTILIZATION_PAYDOWN']
    assert result['suggested_dispute_frame'] == 'verification'

def test_medical_small_balance_policy(rulebook):
    facts = {'type': 'medical', 'amount': 400, 'status': 'unpaid'}
    result = normalize_and_tag({}, facts, rulebook)
    assert result['rule_hits'] == ['J_MEDICAL']
    assert result['suggested_dispute_frame'] == 'verification'

def test_unauthorized_inquiry_old(rulebook):
    facts = {'type': 'inquiry', 'inquiry_age_months': 26}
    result = normalize_and_tag({}, facts, rulebook)
    assert result['rule_hits'] == ['M_UNAUTHORIZED_INQUIRY']
    assert result['suggested_dispute_frame'] == 'inquiry_dispute'

def test_duplicate_tradeline(rulebook):
    facts = {'is_duplicate': True}
    result = normalize_and_tag({}, facts, rulebook)
    assert result['rule_hits'] == ['L_DUPLICATE_TRADELINE']
    assert result['suggested_dispute_frame'] == 'bureau_dispute'
