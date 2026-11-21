# Blast Radius selector

For each patch to sqlglot (pinned at `eb11aec`), decide which tests can fail and
pack them onto at most 8 workers.

## How it works

```
patch ──► static impact analysis ──► candidate tests ──► replay cheap candidates ──► failing ∪ un-replayed ──► worker packing
          (baseline maps, AST diff)                     (patched overlay, unittest)                       (LPT + local search)
```

### 1. Baseline maps (built offline from the unchanged pinned commit)

`tools/` builds `maps/index.pkl.gz` by running sqlglot's own test suite on the
pinned commit with a tracer (`tools/brmap_plugin.py`, `sys.monitoring`):

| Map | What it records |
| --- | --- |
| line → tests | every executed line of `sqlglot/` and `tests/`, per test; class-level setup is attributed to every test in the class |
| class → tests | receiver classes (`type(self)` / `cls`, whole MRO) of every sqlglot call a test makes |
| file → readers | repository files each test opens, and directories it lists |
| identifier → tests | tests executing a line that mentions an identifier |
| import-time lines | lines executed while importing sqlglot, all dialects and the test modules |
| collection imports | modules each test file imports when collected |

Process pools used by the tests are traced in-process with the same pickle
round-trip a real pool performs (sqlglot expressions pickle via `serde`). The
suite is traced under several hash seeds and the union is kept. Nothing in the
maps comes from any patch, answer or mutation outcome.

### 2. Static impact analysis (`selector/analyze.py`)

Each patch is applied in memory to the baseline and the Python files are
compared as ASTs (`selector/astdiff.py`). Each change site is located in the
baseline tree and classified:

* **Runtime code** (inside a function or lambda body): tests that executed the
  innermost enclosing statement (whole statement span, so operator-precedence
  changes are covered). Insertions use the neighbouring statements.
* **Code that runs at import** (metaclass hooks, factories called from class
  bodies): everything that imports the module.
* **Definition-time code** (class attributes, decorators, defaults, module
  constants, imports): users of the class and its subclasses, readers of the
  identifier, and any import-time code that copies it.
* **AST-identical edits** (comments, formatting): only tests that read the file
  as text. **Docstrings**: readers of `__doc__`.
* **Data files** (SQL fixtures etc.): tests that opened them; added/removed
  files also hit directory listers.
* **Files that no longer compile / modules that fail to import**: every test
  that imports them.
* pytest configuration changes, binary patches and anything unparseable fall
  back to all tests.

### 3. Replay (`selector/replay.py`)

Candidates are confirmed by actually running them. The main process preloads
heavy third-party modules once and forks one child per commit. The child builds
a symlink overlay of the checkout in `/tmp` with only the patched files written,
first imports every changed module (an import failure fails every importer),
then runs the candidate `unittest` cases. A test is dropped only if it ran to
completion with no failure, error, failing subtest, unexpected success or
timeout; anything else, including a crashed or killed child, keeps it.

Replay costs CPU, which is charged, so it is planned per commit: candidates are
replayed cheapest first up to the threshold that maximises
`cost(all candidates) − cost(un-replayed) − replay CPU`. Expensive tests whose
replay would cost about what including them costs are included without
replay, and tests that fit in the schedule's slack are included for free.

### 4. Packing (`selector/schedule.py`)

For every worker count 1..8: longest-processing-time first, then bounded
improving moves/swaps. The cheapest `max load + 2 s × workers` wins; an empty
selection costs nothing.

## Build and run

```sh
docker build -t blast-radius .
docker run --rm --network none --read-only --tmpfs /tmp:size=64m --user 65534:65534 \
  -v $PWD/sample/commits:/data/commits:ro -v $PWD/sample/tests.json:/data/tests.json:ro blast-radius
```

The image installs the starter kit's `requirements.lock` from PyPI, downloads
sqlglot at `eb11aec` from its GitHub repository, byte-compiles it and verifies
that the committed index was built from exactly these sources.

## Rebuilding the maps

```sh
# in a checkout of sqlglot at eb11aec with requirements.lock installed
for s in 0 1 2; do
  PYTHONHASHSEED=$s BRMAP_OUT=maps_s$s BRMAP_ROOT=$PWD SKIP_INTEGRATION=0 \
    PYTHONPATH=$PWD:/path/to/tools python -m pytest -p brmap_plugin -n auto --dist loadscope tests
done
python tools/import_pass.py $PWD import_pass.pkl
python tools/build_index.py $PWD tests.json maps/index.pkl.gz import_pass.pkl maps_s0 maps_s1 maps_s2
```
