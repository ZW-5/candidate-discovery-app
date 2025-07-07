import os
import sys
import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError

# Ensure we can import matching_engine
sys.path.append(os.path.join(os.getcwd(), "app"))
from matching_engine import load_data, match_candidates

st.set_page_config(page_title="Candidate Rediscovery", layout="wide")

# — Sidebar: Upload toggle —
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = False
if st.sidebar.button("Upload Custom Data"):
    st.session_state.upload_mode = True
st.sidebar.markdown("---")

# — File uploaders (shown only after you click) —
if st.session_state.upload_mode:
    st.sidebar.subheader("Upload Your CSVs")
    uploaded_apps     = st.sidebar.file_uploader("Applications CSV",       type="csv")
    uploaded_feedback = st.sidebar.file_uploader("Interview Feedback CSV", type="csv")
    uploaded_reqs     = st.sidebar.file_uploader("Requisitions CSV",       type="csv")
    uploaded_resumes  = st.sidebar.file_uploader("Resumes CSV",            type="csv")
    uploaded_jds      = st.sidebar.file_uploader("Job Descriptions CSV",   type="csv")
else:
    # on first load or until button clicked
    uploaded_apps = uploaded_feedback = uploaded_reqs = uploaded_resumes = uploaded_jds = None

def load_df(uploaded, path):
    if uploaded is not None:
        return pd.read_csv(uploaded, engine='python')
    return pd.read_csv(path, engine='python')

# — Data paths —
data_dir      = "data"
apps_path     = os.path.join(data_dir, "applications.csv")
feedback_path = os.path.join(data_dir, "interview_feedback.csv")
reqs_path     = os.path.join(data_dir, "requisitions.csv")
resumes_path  = os.path.join(data_dir, "resumes.csv")
jds_path      = os.path.join(data_dir, "job_descriptions.csv")

# — Load or generate data —
if st.session_state.upload_mode and all([uploaded_apps, uploaded_feedback, uploaded_reqs, uploaded_resumes, uploaded_jds]):
    apps     = load_df(uploaded_apps,     apps_path)
    feedback = load_df(uploaded_feedback, feedback_path)
    reqs     = load_df(uploaded_reqs,     reqs_path)
    resumes  = load_df(uploaded_resumes,  resumes_path)
    jds      = load_df(uploaded_jds,      jds_path)
else:
    apps, feedback, reqs, resumes, jds = load_data(data_dir=data_dir)

# — Run the matching engine every time —
os.makedirs("outputs", exist_ok=True)
match_file = "outputs/candidate_matches.csv"
match_candidates(apps, feedback, reqs, resumes, jds)

# — Load matches safely —
try:
    matches = pd.read_csv(match_file, engine='python')
    # If no rows, pandas will give empty DF but not error.
except EmptyDataError:
    st.warning("No matches could be generated. Check your data or upload valid CSVs.")
    matches = pd.DataFrame(columns=[
        'applicant_id','job_req_id','job_family','location',
        'feedback_text','feedback_sentiment','matched_feedback',
        'feedback_similarity','resume_text','resume_jd_similarity',
        'rediscovery_index_score'
    ])

# — Merge in highlights only if we have matches —
if not matches.empty:
    highlights = pd.read_csv(
        os.path.join(data_dir, "resume_jd_highlights.csv"),
        engine='python'
    )[['applicant_id','job_req_id','resume_highlights','jd_match_points']]
    df = matches.merge(highlights, on=['applicant_id','job_req_id'], how='left')
else:
    df = matches  # empty

# — Sidebar filters (always visible) —
st.sidebar.subheader("Filters")
job_families = st.sidebar.multiselect(
    "Job Family", options=df.get('job_family', []),
    default=list(df['job_family'].unique()) if not df.empty else []
)
locations = st.sidebar.multiselect(
    "Location", options=df.get('location', []),
    default=list(df['location'].unique()) if not df.empty else []
)
if not df.empty:
    score_min, score_max = st.sidebar.slider(
        "Rediscovery Index Score",
        float(df['rediscovery_index_score'].min()),
        float(df['rediscovery_index_score'].max()),
        (0.7, float(df['rediscovery_index_score'].max()))
    )
else:
    score_min, score_max = 0.0, 1.0

# — Apply filters —
filtered = df
if not df.empty:
    filtered = df[
        df['job_family'].isin(job_families) &
        df['location'].isin(locations) &
        df['rediscovery_index_score'].between(score_min, score_max)
    ]

# — Download button —
csv = filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "Download Filtered CSV", data=csv,
    file_name="candidate_matches_filtered.csv", mime="text/csv"
)

# — Main display —
st.title("🔁 Candidate Rediscovery Dashboard")
st.markdown("Browse rediscovered candidates or upload your own data.")

if filtered.empty:
    st.info("No results to display. Adjust filters or upload valid data.")
else:
    st.dataframe(
        filtered[[
            'applicant_id','job_req_id','job_family','location',
            'feedback_text','feedback_sentiment','matched_feedback',
            'feedback_similarity','resume_text','resume_jd_similarity',
            'rediscovery_index_score','resume_highlights','jd_match_points'
        ]],
        use_container_width=True
    )
