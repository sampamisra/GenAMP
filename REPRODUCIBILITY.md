# Reproducibility Package

This document accompanies the GenAMP manuscript and specifies exactly what is
needed to reproduce the reported tables and figures, in response to reviewer
requests for a fully specified reproducibility package.

## 1. Source code version

- Repository: https://github.com/sampamisra/GenAMP
- Version used for the reported results: the commit tagged `v1.0.0` on the
  `main` branch, commit `8b1590534abe32deb2fa85aeb51329d91f191992` (also
  re-verified under tag `v1.0.2`, identical code).
- Archived, citable snapshot (Zenodo): **https://doi.org/10.5281/zenodo.21696805**
  — this is the version-specific DOI for the archived v1.0.0 release. The
  concept DOI `10.5281/zenodo.21696804` always resolves to whichever version
  is archived most recently and should not be used to cite this specific
  result set.

## 2. Environment and dependencies

- Python: 3.9 (tested); should also run on 3.8–3.10
- Recommended setup:
  ```
  conda create -n genamp python=3.9
  conda activate genamp
  pip install -r requirements.txt
  ```
- Core dependencies (see `requirements.txt` for the pinned versions used to
  produce the reported results):
  - `torch` (CUDA build recommended; GPU used for all reported runs)
  - `transformers` (HuggingFace) — provides `BertModel`/`BertTokenizer` and
    `EsmModel`/`AutoTokenizer`
  - `scikit-learn`, `pandas`, `numpy`, `tqdm`, `matplotlib`
- Pretrained language-model backbones (downloaded automatically by
  `transformers` on first run, and cached locally):
  - `Rostlab/prot_bert`
  - `facebook/esm2_t12_35M_UR50D`
- Hardware used for reported runs: `<FILL IN — GPU model, VRAM, approx. wall-clock time per stage>`

`requirements.txt` should be produced from the exact environment used for the
paper's runs, e.g. `pip freeze > requirements.txt` inside the `genamp` conda
env, then hand-trimmed to remove unrelated system packages.

## 3. Random seeds

All stochastic components are seeded for determinism (modulo known
non-determinism in some CUDA/cuDNN kernels):

| Script | Seed | Where set |
|---|---|---|
| `AmpGEncode.py` (Stage 1: VAE + RL sequence generation) | `SEED = 7` | line 20–21, applied to `random`, `numpy`, `torch` (CPU + CUDA) |
| `TrainAMP.py` (Stage 2: AMP classifier training) | `seed=36` via `set_seed(36)` | line 22–30 |
| `TrainMIC.py` (Stage 2: MIC classifier training + CV) | `SEED = 36` | line 17–38; also used as `random_state=SEED` for the train/test split and the 10-fold `StratifiedKFold` (line 169–295) |

Reviewers/readers rerunning training should expect results within normal
run-to-run variance of these seeds; exact bit-for-bit reproduction is not
guaranteed across different GPU architectures/CUDA versions.

## 4. Trained-model availability

Trained checkpoints (Stage 1 VAE/RL generator and Stage 2 AMP/MIC ensemble
heads) are hosted externally due to file size and are not committed to the
repository:

- https://drive.google.com/drive/folders/1hdwsKo8oxES_GdoURBQh6_0IVAsN0s9I

Expected checkpoint filenames and where each script looks for them:

- Stage 1 (`AmpGEncode.py`, `MODEL_DIR = Model/Model_1ststage/`):
  - `VAE_ECD_RL_best.ckpt`, `VAE_ECD_RL_encoder.pt`, `VAE_ECD_RL_decoder.pt`
  - `amp_classifier.pt`, `mic_classifier_best.pt`
- Stage 2 ensemble (`amp_to_22func_pipeline.py`, `Model/Model_2ndstage/`):
  - `best_Bert_cnn_bilstm_attn_fold6.pth`
  - `best_esm2_cnn_bilstm_attn_fold6.pth`
  - `cv_protbert_cnn_bilstm_attn_FT_fold6.pth`
  - `cv_esm2_cnn_bilstm_attn_FT_HP_fold6.pth`

`<FILL IN — confirm license/terms under which these checkpoints are shared,
and whether the Drive folder will be mirrored to the archived Zenodo/GitHub
Release for long-term availability, since Drive links are not guaranteed
stable for citation purposes.>`

## 5. Datasets

Located under `Dataset/{1ststage,2ndstage,3rdstage}/` (paths as referenced in
code) / `data/` (feature tables: `AAindex.txt`, `PAAC.txt`, `blosum62.txt`,
`6-pc`):

- Stage 1: `unlabelled_positive.csv`, `unlabelled_negative.csv`,
  `mic_data.csv`, `Uniprot_0_25_train.csv`, `Uniprot_0_25_val.csv`
- Stage 2: `Train_data.csv`, `Test_data.csv`
- Stage 3: `multilabel_Traindata.csv`

`<FILL IN — dataset license/redistribution terms, and original source
accessions if derived from UniProt/DBAASP/etc., so readers can independently
verify provenance.>`

## 6. Exact steps to reproduce the reported tables

1. **Stage 1 — sequence generation**
   ```
   python AmpGEncode.py
   ```
   Outputs to `Result_1ststage/`: `generated_denovo_ecd_ar.csv`,
   `generated_analogs_ecd_ar.csv` (+ FASTA), `rl_training_stability_log.csv`,
   `rl_training_stability_plot.{png,svg}`. Uses `SEED = 7`.

2. **Stage 2 + Stage 3 — AMP screening ensemble and 22-function tagging**
   ```
   python amp_to_22func_pipeline.py
   ```
   Loads the Stage 1 CSVs, runs the AMP ensemble (ProtBERT + ESM2 heads),
   writes tagged CSVs and pass-rate summaries to `Result_after_2ndstage/`
   (`generated_denovo_ecd_ar_tagged.csv`, `generated_analogs_ecd_ar_tagged.csv`,
   `denovo_ensemble_counts.csv`, `analogs_ensemble_counts.csv`), then runs
   Stage 3 only on `ensemble_amp == 1` rows and writes 22-function
   predictions to `Final_Result/`.

3. **Classifier training (independent of Stage 1/2 generation, used for the
   reported classifier performance tables)**
   ```
   python TrainAMP.py
   python TrainMIC.py
   ```
   `TrainMIC.py` reports 10-fold stratified CV metrics (`SEED = 36`).

4. **Supplementary reviewer table**
   ```
   python export_supplementary_generated_peptides.py
   ```
   Writes `Supplementary_Data_Generated_Peptides.csv` (sequence, generation
   mode, reward score, AMP probability, MIC-class probability, ensemble
   screening score, ML-A2P probabilities, hemolytic/cytotoxic predictions
   where available, nearest-neighbor identity to training data). This is the
   file to consult to reproduce/verify the supplementary peptide table
   reported in the manuscript.

`<FILL IN — map each specific manuscript table/figure number to the exact
output file + column(s) above, e.g. "Table 2 = Result_after_2ndstage/
denovo_ensemble_counts.csv + analogs_ensemble_counts.csv", "Figure 3 =
Result_1ststage/rl_training_stability_plot.png", so reviewers do not have to
infer the mapping.>`

## 7. Known non-determinism / caveats

- GPU non-determinism: some cuDNN kernels used by `torch` are not bit-exact
  across runs even with fixed seeds; report metric ranges over repeated runs
  where feasible.
- HuggingFace model downloads (`Rostlab/prot_bert`,
  `facebook/esm2_t12_35M_UR50D`) depend on the models remaining available at
  those hub identifiers; pin `transformers` and consider vendoring/caching
  the weights alongside the archived release.
