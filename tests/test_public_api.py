"""Basic package metadata and API smoke tests."""

import phbin


def test_version_is_available():
    assert isinstance(phbin.__version__, str)
    assert phbin.__version__


def test_core_helpers_are_exported():
    assert callable(phbin.make_input_imgobj)
    assert callable(phbin.make_pd)
    assert callable(phbin.make_pi)
    assert callable(phbin.draw_pi)
