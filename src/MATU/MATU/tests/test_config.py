from matu.config import choose, section


def test_choose_prefers_cli_value():
    assert choose("cli", {"value": "config"}, "value", "default") == "cli"


def test_choose_uses_config_before_default():
    assert choose(None, {"value": "config"}, "value", "default") == "config"


def test_section_missing_is_empty():
    assert section({}, "missing") == {}
