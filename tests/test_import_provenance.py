from pathlib import Path

import lazystats


def test_pytest_imports_lazystats_from_this_checkout() -> None:
    repository = Path(__file__).resolve().parents[1]
    imported = Path(lazystats.__file__).resolve()
    assert imported.is_relative_to(repository / "src"), (
        f"pytest imported lazystats from another checkout: {imported}"
    )
