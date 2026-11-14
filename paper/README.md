# Research manuscript

`manuscript/paper.pdf` is the paper (13 pages, IEEE conference format).

Nothing in this folder modifies the project code or the experiment outputs in
`outputs/`; the analyses here only read them.

## Layout

```
manuscript/
    paper.pdf            the compiled manuscript
    main.tex             IEEEtran source; includes the sec_*.tex files and the bibliography
    sec_*.tex            abstract, introduction, related work, data, method, setup, results, discussion
    figures/             figures as PDF (used by LaTeX), SVG and PNG
    build.py             fallback renderer for machines without TeX (see below)
analysis/
    tx_analysis.py           transactions: retraining over time, random vs temporal split,
                             drift diagnosis, Isolation Forest by period, paired bootstrap, homophily
    wallet_analysis.py       wallets: feature-group ablation, retraining over time, look-ahead subsets
    wallet_split_protocol.py wallets: random vs temporal split at two row granularities
    wallet_causal.py         wallets: neighbourhoods restricted to already-seen wallets; graph facts
    score_analysis.py        all layers: threshold transfer, calibration, triage, review budgets
    make_figures.py          all figures of the paper
    verify_refs.py           bibliographic look-ups used to check the reference list
    results/                 every number quoted in the paper that is not in outputs/
        network_full/        network layer re-run with all background windows (Table III)
```

## Rebuild

```
# analyses (from paper/analysis; each reads ../../outputs and the datasets)
python tx_analysis.py
python wallet_analysis.py
python wallet_split_protocol.py
python wallet_causal.py
python score_analysis.py
python make_figures.py

# network run reported in the paper (from the project root; about one hour)
python network_anomaly_detection.py --input Dataset_UGR --background-rate 1.0 --out-dir paper/analysis/results/network_full

# manuscript (from paper/manuscript)
tectonic -X compile main.tex --outdir build/tectonic --keep-logs
copy build\tectonic\main.pdf paper.pdf
```

`main.tex` also compiles with pdflatex or on Overleaf (upload the `manuscript` folder).
Without any TeX engine, `python build.py` renders an approximation through Microsoft
Edge to `build/paper_browser.pdf`.

## Before submitting

- Fill in the author block at the top of `main.tex` (names, affiliation, e-mail).
- The Reproducibility section says the code accompanies the paper. Add the repository
  address there once the code is published, or reword it.
- Check the venue's rules on disclosing AI assistance in writing and analysis.
