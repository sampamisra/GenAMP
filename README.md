GenAMP Pipeline
End-to-end antimicrobial peptide (AMP) discovery pipeline:
1. Stage 1: Generate candidate peptide sequences using a conditional VAE + reinforcement learning.
2. Stage 2: Filter sequences with an AMP ensemble (ProtBERT/ESM2 models) and label each sequence as AMP or not. 
3. Stage 3: For AMP-positive sequences, assign 22 antimicrobial / functional attributes.
4.  Environment / Requirements
- Python 3.8+
- PyTorch with GPU support recommended
- transformers (HuggingFace)
- pandas, numpy, scikit-learn, tqdm
- Trained weights:
  • Stage 1 generative VAE + RL decoder
  • amp_classifier.pt and mic_classifier_best.pt (used for RL reward shaping and scoring)  fileciteturn0file1
  • Stage 2 ensemble weights (Model/Model_2ndstage/*)  fileciteturn0file0
  • Stage 3 specialists and stacker models (Model/Model_3rdstage/*)  fileciteturn0file0
2. Directory Layout
Dataset/
  1ststage/
    unlabelled_positive.csv
    unlabelled_negative.csv
    mic_data.csv
    Uniprot_0_25_train.csv
    Uniprot_0_25_val.csv
  2ndstage\
  Train_data
  Test_data
  3rdstage\
  multilabel_Traindata
Model/
  Model_1ststage/
    VAE_ECD_RL_best.ckpt
    VAE_ECD_RL_encoder.pt
    VAE_ECD_RL_decoder.pt
    amp_classifier.pt
    mic_classifier_best.pt
  Model_2ndstage/
    best_Bert_cnn_bilstm_attn_fold6.pth
    best_esm2_cnn_bilstm_attn_fold6.pth
    cv_protbert_cnn_bilstm_attn_FT_fold6.pth
    cv_esm2_cnn_bilstm_attn_FT_HP_fold6.pth
  Model_3rdstage/
    specialists/fold_1/*.pth
    specialists/fold_1/scaler.pkl
    stacker_from_saved/fold_*/meta.pth
    labels.json (or label_cols.json)
Result_1ststage/           (Stage 1 output)
Result_after_2ndstage/     (Stage 2 output)
Final_Result/              (Stage 3 output)
3. Stage 1 — Sequence Generation (AmpGEncode.py)  fileciteturn0file1
Goal: generate antimicrobial-like peptide sequences using a VAE with encoder–conditioner–decoder (ECD) plus optional RL.
- Encoder: biGRU → latent z
- Conditioner (ECD): transforms z to encourage controllable attributes
- Decoder: autoregressive GRU+LSTM that emits amino acids until EOS
- Optional RL (REINFORCE): rewards high AMP probability and low MIC using pretrained classifiers amp_classifier.pt and mic_classifier_best.pt
Outputs (saved in Result_1ststage/):
- generated_denovo_ecd_ar.csv
  • de novo sequences sampled from the latent space
- generated_analogs_ecd_ar.csv
  • analog sequences sampled around seed peptides
Each CSV row includes:
- name
- sequence
- amp_prob (predicted AMP likelihood)
- mic_prob_low (predicted “low MIC”, i.e. potent)
- length, charge at pH 7.4, hydrophobicity, hydrophobic moment, isoelectric point
These CSVs are the inputs to Stage 2.
4. Stage 2 — AMP Screening (amp_to_22func_pipeline.py, Stage-2)  
Goal: decide which generated sequences are actually AMPs.
Process:
1. Load Result_1ststage/generated_denovo_ecd_ar.csv and generated_analogs_ecd_ar.csv
2. Run an ensemble of four heads:
   - token_protbert: ProtBERT embeddings → CNN–BiLSTM–Attention
   - token_esm2: ESM2 embeddings → CNN–BiLSTM–Attention
   - ft_protbert: finetuned ProtBERT+CNN+BiLSTM+Attention
   - ft_esm2: finetuned ESM2+CNN+BiLSTM+Attention
3. For each peptide, compute:
   - ensemble_softscore = mean AMP probability
   - ensemble_amp = 1/0 decision (default rule: soft score ≥ 0.5)
Saved outputs (in Result_after_2ndstage/):
- generated_denovo_ecd_ar_tagged.csv
- generated_analogs_ecd_ar_tagged.csv
Each tagged CSV:
- copies original sequence rows
- adds ensemble_softscore and ensemble_amp
Additionally:
- denovo_ensemble_counts.csv
- analogs_ensemble_counts.csv
These summarize how many sequences passed AMP filtering.
5. Stage 3 — 22-Function Profiling (amp_to_22func_pipeline.py, Stage-3)  fileciteturn0file0
Goal: For sequences predicted as AMPs (ensemble_amp == 1), assign 22 functional attributes (multi-label).
Process:
1. Take only rows where ensemble_amp == 1
2. Build sequence feature embeddings:
   - amino acid indices
   - BLOSUM62
   - PAAC
   - one-hot
3. Feed features into:
   - 22 specialist binary MLPs, one per functional label
   - optional meta-stacker models that learn co-occurrence structure
4. For each function:
   - <function>_prob (0–1)
   - <function> (binary call: prob ≥ 0.5)
Saved outputs (in Final_Result/):
- generated_denovo_ecd_ar_tagged_amp_22func_pred.csv
- generated_analogs_ecd_ar_tagged_amp_22func_pred.csv
These CSVs contain only AMP-positive candidates and include predicted functional fingerprints across all 22 attributes.
6. Full Workflow
Step 1: run AmpGEncode.py to generate sequences (Stage 1).
Outputs to Result_1ststage/.  fileciteturn0file1
Step 2: run amp_to_22func_pipeline.py (this runs Stage 2 and Stage 3 end-to-end).
- Loads Stage 1 CSVs
- Runs Stage 2 AMP ensemble
- Saves tagged CSVs + pass-rate summaries in Result_after_2ndstage/
- Runs Stage 3 only on sequences where ensemble_amp == 1
- Saves 22-function predictions in Final_Result/  fileciteturn0file0
