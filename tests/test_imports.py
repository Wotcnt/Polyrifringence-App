def test_import_poly_engine():
    import poly_engine  # noqa

from pathlib import Path
import py_compile

def test_app_py_compiles():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    assert app_path.exists(), "app.py missing at repo root"
    py_compile.compile(str(app_path), doraise=True)
