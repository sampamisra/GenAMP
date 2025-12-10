GenAMP Pipeline
End-to-end antimicrobial peptide (AMP) discovery pipeline:
1. Stage 1: Generate candidate peptide sequences using a VAE + reinforcement learning.
2. Stage 2: Filter sequences with an AMP ensemble (ProtBERT/ESM2 models) and label each sequence as AMP or not. 
3. Stage 3: For AMP-positive sequences, assign 22 antimicrobial / functional attributes.
   
Environment / Requirements
- Python 3.8+
- PyTorch with GPU support recommended
- transformers (HuggingFace)
- pandas, numpy, scikit-learn, tqdm
  
Trained weights:


  • Stage 1 generative VAE + RL decoder
   amp_classifier.pt and mic_classifier_best.pt (used for RL reward shaping and scoring)  
  • Stage 2 ensemble weights (Model/Model_2ndstage/*)  
  • Stage 3 specialists and stacker models (Model/Model_3rdstage/*)  
  

Model can be downloaed from https://drive.google.com/drive/folders/1hdwsKo8oxES_GdoURBQh6_0IVAsN0s9I?usp=sharing


	
Model



  Model_1ststage
  
  
    VAE_ECD_RL_best.ckpt
	
    VAE_ECD_RL_encoder.pt
	
    VAE_ECD_RL_decoder.pt
	
    amp_classifier.pt
	
    mic_classifier_best.pt
	
  Model_2ndstage
  
  
    best_Bert_cnn_bilstm_attn_fold6.pth
	
    best_esm2_cnn_bilstm_attn_fold6.pth
	
    cv_protbert_cnn_bilstm_attn_FT_fold6.pth
	
    cv_esm2_cnn_bilstm_attn_FT_HP_fold6.pth
	
  Model_3rdstage/

  
    specialists/fold_1/*.pth/
	
    specialists/fold_1/scaler.pkl/
	
    stacker_from_saved/fold_*/meta.pth/
	
    labels.json (or label_cols.json)/
	


Dataset



  1ststage

  
    unlabelled_positive.csv
	
    unlabelled_negative.csv
	
    mic_data.csv
	
    Uniprot_0_25_train.csv
	
    Uniprot_0_25_val.csv


 2ndstage

  
    Train_data.csv
	
    Test_data.csv
	

3rdstage

  
    multilabel_Traindata.csv
	


Full Workflow

Step 1: run AmpGEncode.py to generate sequences (Stage 1).

Outputs to Result_1ststage/


Step 2: run amp_to_22func_pipeline.py (this runs Stage 2 and Stage 3 end-to-end).


- Loads Stage 1 CSVs
- Runs Stage 2 AMP ensemble
- Saves tagged CSVs + pass-rate summaries in Result_after_2ndstage/
- Runs Stage 3 only on sequences where ensemble_amp == 1
- Saves 22-function predictions in Final_Result/  fileciteturn0file0
