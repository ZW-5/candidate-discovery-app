import os
import sys
import pandas as pd
import streamlit as st

# Ensure we can import matching_engine
sys.path.append(os.path.join(os.getcwd(), "app"))
from matching_engine import load_data, match_candidates

st.set_page_config(page_title="Candidate Rediscovery", layout="wide")

# Initialize session state for upload toggle
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = False

# Sidebar: Upload Data Button
if st.sidebar.button("Upload Custom Data"):
    st.session_state.upload_mode = True

st.sidebar.markdown("---")

# File uploaders under upload mode
if st.session_state.upload_mode:
    st.sidebar.subheader("Upload Your CSVs")
    uploaded_apps     = st.sidebar.file_uploader("Applications CSV",       type="csv")
    uploaded_feedback = st.sidebar.file_uploader("Interview Feedback CSV", type="csv")
    uploaded_reqs     = st.sidebar.file_uploader("Requisitions CSV",       type="csv")
    uploaded_resumes  = st.sidebar.file_uploader("Resumes CSV",            type="csv")
    uploaded_jds      = st.sidebar.file_uploader("Job Descriptions CSV",   type="csv")
else:
    uploaded_apps = uploaded_feedback = uploaded_reqs = uploaded_resumes = uploaded_jds = None

# Helper to read uploaded or default
def load_df(uploaded, path):
    if uploaded is not None:
        return pd.read_csv(uploaded, engine='python')
    else:
        return pd.read_csv(path, engine='python')

# Paths
data_dir = "data"
apps_path     = os.path.join(data_dir, "applications.csv")
feedback_path = os.path.join(data_dir, "interview_feedback.csv")
reqs_path     = os.path.join(data_dir, "requisitions.csv")
resumes_path  = os.path.join(data_dir, "resumes.csv")
jds_path      = os.path.join(data_dir, "job_descriptions.csv")

# Load or generate data
if st.session_state.upload_mode and all([uploaded_apps, uploaded_feedback, uploaded_reqs, uploaded_resumes, uploaded_jds]):
    apps     = load_df(uploaded_apps,     apps_path)
    feedback = load_df(uploaded_feedback, feedback_path)
    reqs     = load_df(uploaded_reqs,     reqs_path)
    resumes  = load_df(uploaded_resumes,  resumes_path)
    jds      = load_df(uploaded_jds,      jds_path)
else:
    apps, feedback, reqs, resumes, jds = load_data(data_dir=data_dir)

# Run matching
os.makedirs("outputs", exist_ok=True)
match_file = "outputs/candidate_matches.csv"
match_candidates(apps, feedback, reqs, resumes, jds)

# Load results + highlights
matches    = pd.read_csv(match_file, engine='python')
highlights = pd.read_csv(os.path.join(data_dir, "resume_jd_highlights.csv"), engine='python')
highlights = highlights[['applicant_id','job_req_id','resume_highlights','jd_match_points']]

df = matches.merge(highlights, on=['applicant_id','job_req_id'], how='left')

# Sidebar filters
st.sidebar.subheader("Filters")
job_families = st.sidebar.multiselect(
    "Job Family", options=df['job_family'].unique(),
    default=list(df['job_family'].unique())
)
locations = st.sidebar.multiselect(
    "Location", options=df['location'].unique(),
    default=list(df['location'].unique())
)
score_min, score_max = st.sidebar.slider(
    "Rediscovery Index Score",
    float(df['rediscovery_index_score'].min()),
    float(df['rediscovery_index_score'].max()),
    (0.7, float(df['rediscovery_index_score'].max()))
)

# Apply filters
filtered = df[
    df['job_family'].isin(job_families) &
    df['location'].isin(locations) &
    df['rediscovery_index_score'].between(score_min, score_max)
]

# Download button
csv = filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "Download Filtered CSV", data=csv,
    file_name="candidate_matches_filtered.csv", mime="text/csv"
)

# Main display
st.title("🔁 Candidate Rediscovery Dashboard")
st.markdown("Browse rediscovered candidates or upload custom data to run matching.")

st.dataframe(
    filtered[[
        'applicant_id','job_req_id','job_family','location',
        'feedback_text','feedback_sentiment','matched_feedback',
        'feedback_similarity','resume_text','resume_jd_similarity',
        'rediscovery_index_score','resume_highlights','jd_match_points'
    ]],
    use_container_width=True
)
