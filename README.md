# DiaBeat — educational diabetes classification research

This hackathon-era project compares small machine-learning models on the public **Pima-style diabetes dataset** and includes a Flask prediction demonstration. It is a **research prototype, not a diagnostic test or a clinical risk score**.

## New reproducible evaluation

A separate evaluation pipeline now trains a majority-class baseline, logistic regression, KNN and random forest on the original raw data. It uses a stratified 75/25 holdout (seed 42), replaces zero placeholders for selected measurements with missing values, and fits imputation/scaling on training data only. The report outputs accuracy, F1, recall and ROC AUC. Results do not establish patient-level or out-of-sample clinical usefulness.

```powershell
git clone https://github.com/rainrunner20/DiaBeat.git
cd DiaBeat
py -m venv .venv
.\.venv\Scripts\python -m pip install -r backend/requirements-eval.txt
.\.venv\Scripts\python -m unittest discover -s backend/tests -v
.\.venv\Scripts\python backend/evaluate_models.py --input backend/diabetes.csv --output reports/diabeat_metrics.json
```

The existing web demo uses previously serialized model artifacts rather than models from this new evaluation pipeline. Pickle artifacts must only be loaded from trusted source files, and compatibility of their original training environment still needs testing before deploying the app.

## Safety and quality limits

- The Flask demo now rejects missing, negative and nonfinite values and displays each model output separately. It no longer calls the **largest predicted percentage** the best model.
- Predictions have not been externally validated, calibrated or assessed for clinical subgroups. Do not interpret them as a personal diagnosis.
- Do not collect or store real patient data in this demonstration. Real clinical systems require review of safety, privacy, bias, calibration and applicable requirements.
- Legacy React frontend and bundled pickle files still need installation, interoperability and deployment verification.
- GitHub Actions validates the research evaluation independently of the old pickle artifacts.

### Project history

Originally built as a hackathon learning project. Historical training notebooks and data files are retained for transparency; the new evaluation pipeline is the supported reproducible benchmark.
