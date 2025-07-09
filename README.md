# Candidate Discovery App

This simple Streamlit demo uses mock recruiting data to show how candidates can be rediscovered based on past applications, interview feedback and resume content.

The repository now includes a small set of CSVs under the `data/` folder so the app loads with working sample data out of the box. You can open `streamlit_app.py` with Streamlit and immediately see the dashboard. The sidebar allows uploading your own CSVs to replace the defaults.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

By default the application will read the CSV files in `data/`. Feel free to modify or replace them with your own exports when ready.
