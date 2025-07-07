import os
import sys
import pandas as pd
import streamlit as st
from io import StringIO

# Ensure we can import matching_engine
sys.path.append(os.path.join(os.getcwd(), "app"))
from matching_engine import load_data, match_candidates

st.set_page_config(page_title="Candidate Rediscovery", layout="wide")

st.sidebar.header("Data Upload (optional)")
# 1) File uploaders for each dataset
uploaded_apps     = st.sidebar.file_uploader("Applications CSV",         type="csv")
uploaded_feedback = st.sidebar.file_uploader("Interview Feedback CSV",   type="csv")
uploaded_reqs     = st.sidebar.file_uploader("Requisitions CSV",         type="csv")
uploaded_resumes  = st.sidebar.file_uploader("Resumes CSV",              type="csv")
uploaded_jds      = st.sidebar.file_uploader("Job Descriptions CSV",     type="csv")

# 2) Helper to read uploaded or default
def load_df(uploaded, path):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    else:
        return pd.read_csv(path, engine="python")

# 3) Prepare the data directory if needed
os.makedirs("outputs", exist_ok=True)
data_dir = "data"
apps_path     = os.path.join(data_dir, "applications.csv")
feedback_path = os.path.join(data_dir, "interview_feedback.csv")
reqs_path     = os.path.join(data_dir, "requisitions.csv")
resumes_path  = os.path.join(data_dir, "resumes.csv")
jds_path      = os.path.join(data_dir, "job_descriptions.csv")

# 4) Load or generate mock data
if any(u is None for u in [uploaded_apps, uploaded_feedback, uploaded_reqs, uploaded_resumes, uploaded_jds]):
    # If any uploads are missing, let the matching engine generate defaults
    apps, feedback, reqs, resumes, jds = load_data(data_dir=data_dir)
else:
    # All files provided: load from uploads
    apps     = load_df(uploaded_apps,     apps_path)
    feedback = load_df(uploaded_feedback, feedback_path)
    reqs     = load_df(uploaded_reqs,     reqs_path)
    resumes  = load_df(uploaded_resumes,  resumes_path)
    jds      = load_df(uploaded_jds,      jds_path)

# 5) Run matching
match_file = "outputs/candidate_matches.csv"
# Always overwrite with new run
match_candidates(apps, feedback, reqs, resumes, jds)

# 6) Read in highlights and results
matches    = pd.read_csv(match_file,    engine="python")
highlights = pd.read_csv(
    os.path.join(data_dir, "resume_jd_highlights.csv"),
    engine="python",
    usecols=["applicant_id","job_req_id","resume_highlights","jd_match_points"],
)

df = matches.merge(highlights, on=["applicant_id","job_req_id"], how="left")

# ----- Sidebar Filters -----
st.sidebar.header("Filters")
job_families = st.sidebar.multiselect(
    "Job Family", options=df['job_family'].unique(), default=df['job_family'].unique()
)
locations = st.sidebar.multiselect(
    "Location",   options=df['location'].unique(),   default=df['location'].unique()
)
score_min, score_max = st.sidebar.slider(
    "Rediscovery Index Score",
    float(df['rediscovery_index_score'].min()),
    float(df['rediscovery_index_score'].max()),
    (0.7, float(df['rediscovery_index_score'].max()))
)

# ----- Apply Filters -----
filtered = df[
    df['job_family'].isin(job_families) &
    df['location'].isin(locations) &
    df['rediscovery_index_score'].between(score_min, score_max)
]

# ----- Download Button -----
csv = filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "Download Filtered CSV", data=csv,
    file_name="candidate_matches_filtered.csv", mime="text/csv"
)

# ----- Main Display -----
st.title("🔁 Candidate Rediscovery Dashboard")
st.markdown("Browse rediscovered candidates or upload your own data to run the matching algorithm.")

st.dataframe(
    filtered[[
        'applicant_id','job_req_id','job_family','location',
        'feedback_text','feedback_sentiment','matched_feedback',
        'feedback_similarity','resume_text','resume_jd_similarity',
        'rediscovery_index_score','resume_highlights','jd_match_points'
    ]],
    use_container_width=True
)
