import os
import sys
import pandas as pd
import streamlit as st

# Ensure we can import matching_engine from the app folder
sys.path.append(os.path.join(os.getcwd(), "app"))
from matching_engine import load_data, match_candidates

st.set_page_config(page_title="Candidate Rediscovery", layout="wide")

# 1) Generate the match file if it doesn't exist
os.makedirs("outputs", exist_ok=True)
match_file = "outputs/candidate_matches.csv"
if not os.path.exists(match_file):
    apps, feedback, reqs, resumes, jds = load_data(data_dir="data")
    match_candidates(apps, feedback, reqs, resumes, jds)

# 2) Load matches and highlights
matches = pd.read_csv(match_file)
highlights = pd.read_csv("data/resume_jd_highlights.csv")

# Merge to include resume highlights and JD match points
df = matches.merge(
    highlights[['applicant_id','job_req_id','resume_highlights','jd_match_points']],
    on=['applicant_id','job_req_id'],
    how='left'
)

# Sidebar filters
st.sidebar.header("Filters")
job_families = st.sidebar.multiselect(
    "Job Family",
    options=df['job_family'].unique(),
    default=list(df['job_family'].unique())
)
locations = st.sidebar.multiselect(
    "Location",
    options=df['location'].unique(),
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
    "Download CSV",
    data=csv,
    file_name="candidate_matches_filtered.csv",
    mime="text/csv"
)

# Main display
st.title("🔁 Candidate Rediscovery Dashboard")
st.markdown("Browse high-quality rediscovered candidates.")

st.dataframe(
    filtered[[
        'applicant_id', 'job_req_id', 'job_family', 'location', 'interview_score',
        'feedback_text', 'matched_to_hired_id', 'matched_feedback',
        'feedback_similarity', 'resume_jd_similarity', 'rediscovery_index_score',
        'resume_highlights', 'jd_match_points'
    ]],
    use_container_width=True
)
