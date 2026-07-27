 # **GenAMP Pipeline**

 ## **End-to-end antimicrobial peptide (AMP) discovery pipeline:**

**1. Stage 1:** Generate candidate peptide sequences using a VAE + reinforcement learning.

**2. Stage 2:** Filter sequences with an AMP ensemble (ProtBERT/ESM2 models) and label each sequence as AMP or not. 

**3. Stage 3:** For AMP-positive sequences, assign 22 antimicrobial / functional attributes.

First, clone and navigate to the repository.


	git clone https://github.com/sampamisra/GenAMP.git

	cd GenAMP

This process can take several minutes, depending on network speed.
Create and activate a virtual environment using python 3.9 with virtualenv or cond

	conda create -n genamp python=3.9

    conda activate genamp


 ## **Environment / Requirements**
- Python 3.9 (tested; 3.8–3.10 expected to work)
- PyTorch with GPU support recommended
- transformers (HuggingFace)
- pandas, numpy, scikit-learn, tqdm

Install pinned dependencies with:

    pip install -r requirements.txt

For the exact code version, seeds, trained-model checkpoints, and the
mapping from scripts/output files to reported tables, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

 ## **Model** 
 Trained checkpoints can be downloaded from https://drive.google.com/drive/folders/1hdwsKo8oxES_GdoURBQh6_0IVAsN0s9I?usp=sharing
 (expected filenames and target directories are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#4-trained-model-availability))


	



 ## **Dataset**



   ### **1st stage**

  
    unlabelled_positive.csv
	
    unlabelled_negative.csv
	
    mic_data.csv
	
    Uniprot_0_25_train.csv
	
    Uniprot_0_25_val.csv


 ### **2nd stage**

  
    Train_data.csv
	
    Test_data.csv
	

### **3rd stage**

  
    multilabel_Traindata.csv
	


 ## **Full Workflow**

 Train the AMP classification model: Runs using 

**Step 1:** run **AmpGEncode.py** to generate sequences (Stage 1).

Outputs to Result_1ststage/


**Step 2:** run **amp_to_22func_pipeline.py** (this runs Stage 2 and Stage 3 end-to-end).


- Loads Stage 1 CSVs
- Runs Stage 2 AMP ensemble
- Saves tagged CSVs + pass-rate summaries in Result_after_2ndstage/
- Runs Stage 3 only on sequences where ensemble_amp == 1
- Saves 22-function predictions in Final_Result/

**Step 3:** run **export_supplementary_generated_peptides.py** to create the reviewer-facing supplementary generated-peptide table.

    python export_supplementary_generated_peptides.py

On Windows systems without Python on PATH, the equivalent PowerShell builder can be used:

    powershell -NoProfile -ExecutionPolicy Bypass -File .\build_supplementary_generated_peptides.ps1

This writes **Supplementary_Data_Generated_Peptides.csv**, including sequence, generation mode, reward score, AMP probability, MIC-class probability, ensemble screening score, ML-A2P probabilities, hemolytic/cytotoxic predictions when available, and nearest-neighbor identity to the training data.
