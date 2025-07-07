import pandas as pd
import streamlit as st
import os

st.set_page_config(page_title="Candidate Rediscovery", layout="wide")

# 1) Load the core matches and the highlights
matches = pd.read_csv("outputs/candidate_matches.csv")
highlights = pd.read_csv("data/resume_jd_highlights.csv")

# 2) Merge them so you have both highlight columns available
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

# Main area
st.title("🔁 Candidate Rediscovery Dashboard")
st.markdown("Browse high-quality rediscovered candidates.")

st.dataframe(
    filtered[[
        'applicant_id', 'job_family', 'location', 'interview_score',
        'feedback_text', 'matched_to_hired_id', 'matched_feedback',
        'feedback_similarity', 'resume_jd_similarity', 'rediscovery_index_score',
        'resume_highlights', 'jd_match_points'
    ]],
    use_container_width=True
)
