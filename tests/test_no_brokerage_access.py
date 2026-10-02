"""
Safety tests: the project must not be able to reach a brokerage.

These tests read the project's own source code (they do not run it) and fail
if anything could connect to the internet or a broker:

1. Only a short, approved list of imports is allowed. Adding something like
   `requests`, `socket`, or `robin_stocks` makes this test fail until a
   human deliberately reviews it and updates ALLOWED_IMPORTS.
2. No credentials-style names (API keys, passwords, tokens) may be used as
   variables, settings, functions, or arguments. Comments are ignored, so a
   note like "there are no API keys here" is fine.
3. The paper-trading safety switch must be on.
"""

import ast
import pathlib
import unittest

from config import settings

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE_FOLDERS = ["src", "config"]

# Every module the project code is allowed to import. None of these can
# open a network connection. (sqlite3 only reads and writes a local file.)
# `random` is deliberately NOT allowed: the agent must never make up prices.
ALLOWED_IMPORTS = {"math", "dataclasses", "datetime", "csv", "os", "abc",
                   "enum", "sqlite3", "contextlib", "warnings", "json",
                   "hashlib", "argparse", "config", "src"}

FORBIDDEN_WORDS = ["api_key", "apikey", "secret", "password", "token",
                   "credential", "oauth", "login", "robin_stocks"]


def project_python_files():
    for folder in CODE_FOLDERS:
        yield from sorted((PROJECT_ROOT / folder).rglob("*.py"))


def imported_modules(path):
    """Top-level names of every module imported by one file."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            yield (node.module or "").split(".")[0]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "__import__"):
            yield "__import__"   # sneaky dynamic import: never allowed


def names_used(path):
    """Every variable, setting, function, attribute and argument name."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg


class NoBrokerageAccessTests(unittest.TestCase):
    def test_code_files_were_found(self):
        names = {p.name for p in project_python_files()}
        self.assertIn("paper_trader.py", names)
        self.assertIn("risk_manager.py", names)

    def test_only_approved_imports(self):
        for path in project_python_files():
            for module in imported_modules(path):
                with self.subTest(file=path.name, module=module):
                    self.assertIn(module, ALLOWED_IMPORTS,
                                  f"{path.name} imports '{module}', which is not "
                                  "on the approved (offline-only) list.")

    def test_no_credentials_in_code(self):
        for path in project_python_files():
            for name in names_used(path):
                for word in FORBIDDEN_WORDS:
                    with self.subTest(file=path.name, name=name):
                        self.assertNotIn(word, name.lower())

    def test_checker_catches_a_planted_credential(self):
        """Make sure the scanner really works, using a throwaway snippet."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bad = pathlib.Path(tmp) / "bad.py"
            bad.write_text("import requests\nROBINHOOD_PASSWORD = 'x'\n")
            self.assertIn("requests", set(imported_modules(bad)))
            self.assertIn("ROBINHOOD_PASSWORD", set(names_used(bad)))

    def test_paper_trading_switch_is_on(self):
        self.assertIs(settings.PAPER_TRADING, True)


if __name__ == "__main__":
    unittest.main()
