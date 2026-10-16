# Composing usage policies for models trained on many sources

Code and data for the paper of the same name.

When a model is trained on data held by several organisations, each owner
attaches its own conditions of use, and nothing in the existing standards says
what the group may collectively do with the result. This repository implements a
composition algebra for a fragment of [ODRL](https://www.w3.org/TR/odrl-model/)
and proves that capability is **anti-monotone** in the contributor set: admitting
a contributor can only narrow what the group may lawfully do. On top of the
algebra it provides a measure of how similar a group's conditions are, a
selection procedure that uses anti-monotonicity to prune a search over subsets,
and a model bundle that carries its composed agreement. Everything is
demonstrated on GA4GH [DUO](https://github.com/EBISPOT/DUO) conditions as
published by the European Genome-phenome Archive.

## Layout

| Path | Contents |
|---|---|
| `odrl_fl/odrl/` | The policy model, the JSON-LD parser, the composition algebra and the subsumption taxonomy |
| `odrl_fl/` | Corpus generation, similarity, selection, bundles and the training paths |
| `experiments/` | One script per result; each writes a CSV to `results/` |
| `tests/` | 544 tests, including property-based checks of anti-monotonicity |
| `vocab/` | The DUO classification, the code frequencies and the EGA corpus |
| `results/` | Committed outputs; every number in the paper comes from here |
| `policies/` | Hand-written example policies |

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

Requires Python 3.11 or later. The test suite runs without network access and
without the TCGA cohort, which is fetched on demand rather than vendored.

## Reproducing the results

The figures regenerate from the committed CSVs, so this needs no experiment to
be re-run:

```bash
python experiments/make_figures.py
```

To recompute the results themselves:

```bash
python experiments/corpus_demo.py      # DUO classification and adoption rate
python experiments/run_measured.py     # similarity, and capability loss
python experiments/run_selection.py    # selection and the policy-blind baseline
```

`run_measured.py` reads the EGA corpus from `vocab/ega_corpus.json`, which is
committed. To refetch it from the archive, run `python -m odrl_fl.ega --fetch`.

## Where the paper's numbers come from

| Paper | Source |
|---|---|
| Table 2, DUO classification | `vocab/duo.json` |
| Table 3, measured similarity | `vocab/ega_corpus.json`, via `odrl_fl/ega.py` |
| Table 4, capability loss | `results/measured.csv` |
| Table 5, selection | `results/selection.csv` |
| Adoption rate, 21.4% | `vocab/ega_corpus.json` provenance block |

Each experiment script prints the numbers it writes, so a run can be compared
against the committed CSV directly.

## Scope

The algebra and its results are combinatorial: they follow from composing
declared terms and never touch a model, a metric or a dataset. The repository
therefore makes no claim about which permitted group produces the better model,
and the selection procedure identifies the groups that are *permitted* rather
than the group that is best.

Two limitations are worth stating here as well as in the paper. Bundle manifests
are not signed, so a forger who rewrites the declarations and the agreement
consistently produces a bundle that verifies. And the method observes conditions
rather than data, so a contributor combining permissive terms with poor data is
invisible to it.

## Licence

MIT. See `LICENSE`.
