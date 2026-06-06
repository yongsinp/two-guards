from datetime import date

import pytest

from two_guards.preprocessing.download_files import load_dotenv_file, resolve_token, validate_date_span


def test_resolve_token_prefers_cli_token(monkeypatch):
    monkeypatch.setenv("COURTLISTENER_API_TOKEN", "env-token")
    assert resolve_token("cli-token") == "cli-token"


def test_resolve_token_reads_primary_env_var(monkeypatch):
    monkeypatch.setenv("COURTLISTENER_API_KEY", "key-token")
    assert resolve_token(None) == "key-token"


def test_resolve_token_raises_when_missing(monkeypatch):
    monkeypatch.delenv("COURTLISTENER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        resolve_token(None)


def test_validate_date_span_allows_equal_dates():
    validate_date_span(date(2025, 1, 1), date(2025, 1, 1))


def test_validate_date_span_raises_on_invalid_order():
    with pytest.raises(SystemExit):
        validate_date_span(date(2025, 2, 1), date(2025, 1, 1))


def test_load_dotenv_file_sets_missing_env_vars(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("COURTLISTENER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("COURTLISTENER_API_KEY", raising=False)

    load_dotenv_file(str(env_file))

    assert resolve_token(None) == "from-file"


def test_load_dotenv_file_does_not_override_existing_env_vars(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("COURTLISTENER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("COURTLISTENER_API_KEY", "existing")

    load_dotenv_file(str(env_file))

    assert resolve_token(None) == "existing"
