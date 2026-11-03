# The revised manuscript

This folder holds the revised version of

> M. Usman and Y. Lee, **"Does Trust Decohere? A Pre-Registered Test for Quantum
> Hardware Reliability,"** submitted to *IEEE Transactions on Reliability*.

The folder is self-contained: everything needed to rebuild the PDF is here.

| File | What it is |
|---|---|
| `itn_revise.pdf` | The revised manuscript (15 pages). |
| `itn_revise.tex` | Its LaTeX source. The references are written out in the file, so no BibTeX run is needed. |
| `IEEEtran.cls` | The IEEE journal class the manuscript is set in. |
| `figures/` | The nine figure files the manuscript includes (mapped to figure numbers below). |
| `tables/` | `T5_simval.tex` (Table I) and `T6_reliability.tex` (Tables VIII–X). All other tables are typeset inside `itn_revise.tex`. |

## Figure files

The file names follow the generator's numbering, not the paper's.

| File | In the paper |
|---|---|
| `figures/F1_concept.pdf` | Fig. 1 |
| `figures/F2_architecture.pdf` | Fig. 2 |
| `figures/simval/sim_vs_real_marginals.pdf`, `figures/simval/sim_vs_real_dynamics.pdf` | Fig. 3 (top, bottom) |
| `figures/F3_crps.pdf` | Fig. 4 |
| `figures/F4_ablations.pdf` | Fig. 5 |
| `figures/F5_transfer.pdf` | Fig. 6 |
| `figures/F6_regret.pdf` | Fig. 7 |
| `figures/F7_event.pdf` | Fig. 8 |

## Building the PDF

Run pdfLaTeX three times in this folder, so that the cross-references settle:

```bash
pdflatex itn_revise.tex
pdflatex itn_revise.tex
pdflatex itn_revise.tex
```

The manuscript uses the `newtx` fonts, which a full TeX Live installation includes.
`itn_revise.pdf` was built with pdfTeX 1.40.28 (TeX Live 2025) and newtx 1.756. Line
breaking depends on these versions, so another TeX installation can set the same source on
a different number of pages.

## Where the figures and tables come from

Every figure and table file in this folder is produced by code in this repository. The
top-level [README](../README.md#where-each-result-comes-from) maps each one to its generator
and to the records or data it reads. The manuscript's Data Availability statement cites
commit `4a87c7f`, the state of the code and records it was built from; this folder was added
in a later commit.
