from importlib.metadata import version

import lazystats


def test_version_matches_installed_distribution() -> None:
    assert lazystats.__version__ == version("lazystats")
