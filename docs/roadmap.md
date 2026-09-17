# Roadmap: specified, not built

Everything listed in the README is implemented and test-pinned in both ports.
This file holds the one contract that is specified but deliberately not built
yet; it claims cross-language parity only once its JSON fixtures exist in both
repositories.

Design constraints every pack must respect: policy is the only veto point;
the ledger is the only truth; a pack is one JSON definition file plus one
handler map per language, installed through `install_builtins` /
`installBuiltins` like the base packs.

## memory — cross-session notes `[spec only]`

`memory.note.save(text, tags)` / `memory.note.search(query, k)` over a
`MemoryStore` protocol (default: JSONL beside the ledger directory), keyed
outside any run namespace so notes outlive runs and sessions. Only search
hits enter the context, as observations; nothing is injected automatically,
so a stale note can never masquerade as an instruction. The store interface
is two methods (`save`, `search`) so a database or vector index can replace
the default without touching the pack.
