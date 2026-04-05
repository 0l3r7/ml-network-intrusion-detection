# Models Folder

This folder contains the trained machine learning models and preprocessing files used in the **ML-Based Network Intrusion Detection System (NIDS) with Real-Time Alerting** project.

## Files

* `columns.pkl` → stores the feature column order used during prediction
* `iso_model.pkl` → Isolation Forest model for anomaly detection
* `iso_scaler.pkl` → scaler used for the Isolation Forest model
* `rf_model.pkl` → Random Forest model for classification
* `scaler.pkl` → scaler used for preprocessing input data

## Notes

* These files are required for running predictions in the application.
* If model files are missing, they can be recreated by running the training script:

  * `src/train.py`
* Large model files may be excluded from the repository if needed and regenerated locally.
