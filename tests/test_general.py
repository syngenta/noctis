import noctis


def test_version():
    version = noctis.__version__

    assert version is not None
