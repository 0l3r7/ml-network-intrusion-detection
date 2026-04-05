# Demo Notes

This folder stores documentation and screenshots for the project demonstration.

## Recommended Screenshots

Place screenshots inside `docs/screenshots/` such as:

* `dashboard.png` → main Streamlit dashboard
* `alerts.png` → live intrusion alert detection
* `model_results.png` → model prediction or performance output

## Demo Workflow

1. Run the Streamlit dashboard:

   ```cmd
   python -m streamlit run src/app.py
   ```
2. Start the live packet monitoring / detection process.
3. Simulate an Nmap scan from another machine or VM:

   ```bash
   nmap -sS <target-ip>
   ```
4. Observe real-time alerts on the dashboard.
5. Review the detection results and model behavior.

## Notes

* Make sure the required model files are available in the `models/` folder.
* Make sure the dataset is available in the `data/raw/` folder if needed.
* This file is used as a quick guide for demonstrating the project.

## to train
```cmd
python src/train.py
```