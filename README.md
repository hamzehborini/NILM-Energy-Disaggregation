NILM: Non-Intrusive Load Monitoring for Household Energy Disaggregation


Disaggregating appliance-level energy consumption from a single smart meter signal — no per-device sensors required.



Overview

Non-Intrusive Load Monitoring (NILM) uses machine learning to infer which household appliances are running and how much power each consumes, using only the aggregate reading from a single smart meter. This has real applications in energy billing transparency, appliance-level insights for consumers, and demand forecasting.

This project implements a complete end-to-end NILM system covering four appliances from the REFIT Electrical Load Measurements dataset: Kettle, Fridge, Washing Machine, and Dishwasher. This was originally a team graduation project (University of Jordan, KASIT); this repo reflects my contribution and the project as a whole.

Results

Each appliance has a dedicated deep learning model, evaluated under a strict Leave-One-House-Out (LOHO) protocol — House 20 was held out completely unseen during training and tuning.

ApplianceArchitectureF1PrecisionRecallKettlePure CNN (w=99)0.7840.7260.853FridgeCNN-BiLSTM (w=99)0.6880.5450.933Washing MachineCNN-BiLSTM (w=599)0.5770.5610.593Dishwasher4-ch MultiScale CNN-BiLSTM (w=599)0.7260.7060.747

Approach

Development proceeded through three phases:


Classical ML baseline — SVM and Random Forest with 16 hand-crafted features (F1 up to 0.548)
BiLSTM phase — sequence modeling using the Seq2Point windowing convention, reformulated as binary classification (F1 = 0.67 on kettle)
Final CNN-BiLSTM phase — per-appliance window size and threshold tuning, yielding the results above


Framework: PyTorch 2.6, trained on Google Colab.

Demo

The system is deployed as a 5-page Streamlit web application providing:


Live disaggregation
Usage analytics
Bill estimation (Jordan EMRC tariffs)
Energy forecasting
A savings simulator


(screenshots below)

What I'd Improve


The microwave was trained extensively but capped at F1=0.40 (DilatedCNN + Focal Loss at 8-second resolution) and was excluded from the final lineup — with more time, a higher-resolution approach or appliance-specific architecture search could push this further
Washing machine performance (F1=0.577) lags the others — likely due to its long, variable-duration cycles; a sequence-to-sequence approach rather than binary classification might help
Evaluation is currently limited to one held-out house; testing across more unseen houses would give a more robust generalization estimate


Tech Stack


Data: REFIT Electrical Load Measurements dataset (20 UK households, 8-second intervals, resampled to 1-minute)
Models: PyTorch (CNN-BiLSTM), scikit-learn (SVM/RF baseline)
App: Streamlit
Training: Google Colab + Google Drive


Repository Structure

├── code/           # Model training notebooks (kettle, fridge, washing machine, dishwasher)
├── app/            # Streamlit deployment app
├── docs/           # Full project report (PDF)
└── README.md

How to Run


Clone the repo: git clone https://github.com/hamzehborini/NILM-Energy-Disaggregation.git
Install dependencies: pip install -r requirements.txt
To explore model training: open notebooks in code/ via Jupyter or Colab
To run the app: cd app && streamlit run app.py


Acknowledgements

Originally developed as a graduation project supervised by Dr. Ruba Obiedat, King Abdullah II School of Information Technology, University of Jordan, using the REFIT dataset (Murray et al., 2017).
