from conftest import FAKE_KEY

from redaction import find_secrets, is_reference, looks_like_placeholder, redact


def test_redacts_known_token_formats():
    gh = "ghp_" + "a" * 4 + "Bc9" * 12
    text = f"key = '{FAKE_KEY}'\ntoken: {gh}"
    out = redact(text)
    assert FAKE_KEY not in out and gh not in out
    assert "[REDACTED:aws_access_key_id]" in out


def test_redacts_url_password_and_literal_assignment():
    out = redact("db = 'postgres://user:hunter2pass@host/db'\npassword = \"Sup3rS3cret!\"")
    assert "hunter2pass" not in out and "Sup3rS3cret!" not in out


def test_redacts_env_style_values():
    out = redact("SECRET_KEY=abcdef1234567890\nDEBUG=1")
    assert "abcdef1234567890" not in out and "DEBUG=1" in out


def test_references_are_not_secrets():
    assert is_reference("${{ secrets.TOKEN }}")
    assert is_reference("$TOKEN")
    assert redact('token: "${{ secrets.NPM_TOKEN }}"') == 'token: "${{ secrets.NPM_TOKEN }}"'


def test_placeholders_detected():
    assert looks_like_placeholder("your-api-key-here")
    assert looks_like_placeholder("xxxxxxxx")
    assert not looks_like_placeholder("q8Z2mL0pR4")


def test_find_secrets_reports_lines_without_values():
    hits = find_secrets(f"a = 1\nAWS = '{FAKE_KEY}'\n")
    assert [h.line for h in hits] == [2]
    assert all(FAKE_KEY not in h.redacted_line for h in hits)


def test_find_secrets_ignores_placeholders_flagged():
    hits = find_secrets("API_KEY=changeme\n", env_style=True)
    assert hits and all(h.likely_placeholder for h in hits)


def test_redacts_escaped_quotes_inside_strings():
    out = redact('x = "password = \\"Sup3rS3cret!\\""')
    assert "Sup3rS3cret!" not in out
