from pathlib import Path

from logic import generate_goodwill_letters
from tests.helpers.fake_ai_client import FakeAIClient


def test_orchestrator_invokes_compliance(monkeypatch, tmp_path):
    ai = FakeAIClient()

    monkeypatch.setattr(
        generate_goodwill_letters.goodwill_preparation,
        'select_goodwill_candidates',
        lambda client_info, bureau_data: {'Bank': [{'name': 'Bank', 'account_number': '1', 'status': 'Open'}]},
    )
    monkeypatch.setattr(
        generate_goodwill_letters.goodwill_preparation,
        'prepare_account_summaries',
        lambda accounts, structured, state, audit=None, ai_client=None: accounts,
    )

    def fake_prompt(**kwargs):
        return ({
            'intro_paragraph': 'hi',
            'hardship_paragraph': 'hard',
            'recovery_paragraph': 'rec',
            'closing_paragraph': 'bye',
            'accounts': [{'name': 'Bank', 'account_number': '1', 'status': 'Open', 'paragraph': 'p'}]
        }, [])

    monkeypatch.setattr(generate_goodwill_letters, 'call_gpt_for_goodwill_letter', lambda *a, **k: fake_prompt()[0])

    calls = []
    def fake_compliance(html, state, session_id, doc_type, ai_client=None):
        calls.append(html)
        return html

    monkeypatch.setattr(generate_goodwill_letters, 'run_compliance_pipeline', fake_compliance)
    monkeypatch.setattr(generate_goodwill_letters, 'render_html_to_pdf', lambda html, path: None)
    monkeypatch.setattr(generate_goodwill_letters.goodwill_rendering, 'load_creditor_address_map', lambda: {'bank': 'addr'})
    monkeypatch.setattr(generate_goodwill_letters, 'gather_supporting_docs', lambda session_id: ("", [], {}))
    monkeypatch.setattr(generate_goodwill_letters, 'get_session', lambda sid: {})

    client_info = {'legal_name': 'John Doe', 'session_id': 's1', 'state': 'CA'}
    bureau_data = {}
    generate_goodwill_letters.generate_goodwill_letters(client_info, bureau_data, tmp_path, audit=None, ai_client=ai)
    assert calls  # compliance pipeline was invoked
