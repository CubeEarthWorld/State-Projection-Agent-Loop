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

`projection.json` is one whole turn exactly as the model receives it
(messages and native tool schemas): no change to either package may alter a
byte of it unnoticed. `checklists_v1.json` is the checklist wire format, the
one fixture here that is authored by hand rather than generated.

## Bundled tool definitions

The definitions of the capabilities this package bundles (`meta.*`,
`state.*`, `planning.checklist.manage`, `meta.agent.spawn`) live as JSON
package data under `src/state_projection_loop/builtin/defs/`. Handlers stay
in code — they are the part that genuinely differs per language.

This is their canonical home. The Dart port keeps a copy under
`spec/tools/`, compiled into a string constant because Dart cannot portably
read its own package's data files at runtime. After changing a definition
here, copy the directory across, run `dart run tool/generate_defs.dart`
there, and run both test suites.
