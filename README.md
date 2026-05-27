# Neurochemical Fingerprint of Stroke

**Code repository for:** *Neurotransmitter disconnection unmasks inflammatory vulnerability and shapes functional recovery after ischemic stroke* (CNSR-III; N = 3,582).

Analysis pipeline that integrates individual lesion masks with 17 PET-derived neurotransmitter atlases to construct patient-specific neurochemical fingerprints, performs structural orthogonalization (Koch residual), and runs ordinal regression / additive interaction / mediation / spatial-null / dynamic-trajectory modules used in the manuscript.

---

## Repository layout

```
.
├── preprocessing/         FSL / FastSurfer / DeepISLES + atlas registration
│   ├── batch_run.sh                       per-subject driver
│   ├── lesion_normalize_*.sh              FLIRT lesion→MNI152 with cost-function masking
│   ├── atlas_to_1mm.{py,sh}               resample NT atlases to 1 mm³ MNI space
│   ├── extract_weighted_load_*.sh         Koch-style weighted NT load extraction
│   ├── extract_cst_load.{py,sh}           corticospinal-tract physical damage load
│   └── merge_cst_to_master.py             merge CST load into master clinical CSV
│
├── analysis/              Main statistical pipeline
│   ├── Master_NT_Analysis_v4.py           ⭐ orchestrator: residualization → ordinal models → interactions → atlas
│   ├── residual_analysis.py               TLV orthogonalization (Koch residual)
│   ├── ordinal_logistic_mRS.py            proportional-odds ordinal regression (Models A–D)
│   ├── cst_analysis.py                    CST physical-damage robustness check (Model D)
│   ├── outcome_analysis.py                deep-phenotyping domain-selective effects
│   ├── interaction_deep_dive.py           NT × inflammation interaction analysis
│   ├── double_dissociation.py             pre- vs. post-synaptic dissociation
│   ├── mice_reliability.py                MICE multiple-imputation sensitivity
│   ├── dynamic_inflammation_trajectory.py BSL/M03/M12 hsCRP trajectory (Supplementary Fig. S7)
│   ├── koch_analysis_pipeline.R           Koch-residual reference implementation (R)
│   └── verify_manuscript_numbers.py       reproduces every in-text statistic
│
├── figure/                Publication figure generation
│   ├── select_figure1_case.py             criteria for picking the Figure-1 example patient
│   ├── generate_figure1_panels.py, figure1_combine.py
│   ├── plot_figure2.py … plot_figure6_smalllesion.py
│   └── plot_figure_S1_consort_flow.py … plot_figure_S6_validation.py
│
├── manuscript_tools/      Publication & figure/table assembly
│   ├── generate_publication_ready.py      assembles Tables 2–4 + Supp Tables S1–S5 CSVs
│   ├── tables_to_images.py                CSV → publication-quality PNG/PDF tables
│   ├── fix_table_naming.py                renumbers raw CSV outputs to manuscript order
│   ├── combine_figure_panels.py           combines Fig 3/4/5 sub-panels into A/B/C/D plates
│   ├── render_all_tables_on_server.sh     one-click HPC driver for the above
│   └── _merge_md_to_full.py               assembles Methods/Results/Discussion → Full_Manuscript.md
│
├── Methods_Section.md     authoritative source for Methods text
├── Results_Section.md     authoritative source for Results text
├── Discussion_Section.md  authoritative source for Discussion text
├── Introduction_Section.md
└── Figure_Legends.md      authoritative source for figure legends
```

## Environment

- **Python 3.10**: statsmodels 0.14 · scikit-learn 1.3 · SciPy 1.11 · pandas · numpy · matplotlib · nibabel · python-docx · lifelines
- **R 4.x** (only for `analysis/koch_analysis_pipeline.R`)
- **FSL 6.0**, **FastSurfer 2.4.2** (FreeSurfer v7.3.2 license), **DeepISLES** (SEALS ensemble)
- Deep-learning steps were run in Docker containers on an HPC with NVIDIA A100 GPUs.

```bash
python -m venv .venv && source .venv/bin/activate
pip install statsmodels==0.14 scikit-learn==1.3 scipy==1.11 \
            pandas numpy matplotlib nibabel python-docx lifelines
```

## Reproducing the analysis

```bash
# 1. preprocessing (per subject; runs FSL + FastSurfer + DeepISLES via Docker)
bash preprocessing/batch_run.sh <subject_id>

# 2. core neurotransmitter + outcome analysis (Tables 2-4, Figures 1-5)
python analysis/Master_NT_Analysis_v4.py

# 3. dynamic inflammation trajectory (Supplementary Figure S7)
python analysis/dynamic_inflammation_trajectory.py                  # BSL+M03, N = 1,334
python analysis/dynamic_inflammation_trajectory.py --triplet-only   # BSL+M03+M12, N = 820

# 4. reproduce every in-text number from the manuscript
python analysis/verify_manuscript_numbers.py

# 5. assemble publication-ready tables (CSVs -> PNG/PDF + combined PDF)
python manuscript_tools/generate_publication_ready.py
python manuscript_tools/fix_table_naming.py /path/to/publication_ready
python manuscript_tools/tables_to_images.py /path/to/publication_ready
```

## Data

The CNSR-III clinical data are available from the corresponding author on reasonable request, subject to approval by the CNSR-III steering committee and the relevant institutional review boards. The normative PET atlases are publicly available from the JuSpace toolbox (<https://github.com/juspace/JuSpace>) and the Hansen normative atlas. The JHU ICBM-DTI-81 white-matter atlas is distributed with FSL.

## Funding

This study is supported by grants from the **Beijing Hospitals Authority Clinical Medicine Development of Special Funding Support (ZLRK202312)**, the **Beijing Municipal Science & Technology Commission (No. Z241100009024046)**, and the **Chinese Academy of Medical Sciences Innovation Fund for Medical Sciences (2019-I2M-5-029)**.

## Citation

If you use this code, please cite:
> Liu Z, Li Z, Wang Y, on behalf of the CNSR-III investigators. *Neurotransmitter disconnection unmasks inflammatory vulnerability and shapes functional recovery after ischemic stroke.* (submitted, 2026)

## License

MIT License — see [`LICENSE`](LICENSE).
