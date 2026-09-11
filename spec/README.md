# Cross-language fixtures

`fixtures/*.json` is the contract between this package and its Dart port:
both test suites read the same files and must produce the same outputs.

This package is the reference implementation, so the expectations are
generated from it:

```
python spec/generate_fixtures.py
```

Run that after **deliberately** changing one of the covered functions, then
copy `spec/fixtures/` into the Dart repository. Never regenerate to make an
unexplained diff go away — a changed fixture is a changed contract.

Each of the divergences these fixtures cover (two different content hashes,
two glob dialects, two truncation rules, two JSON separator styles) reached
production unnoticed because nothing compared the two implementations.
