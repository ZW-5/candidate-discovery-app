# app/matching_engine.py

import os
import re
import pandas as pd
import random
from faker import Faker
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer, util

# Initialize Faker and seeds
faker = Faker()
random.seed(42)
Faker.seed(42)

def sanitize_column(df, col):
    """Remove newlines, carriage returns, and stray quotes from a text column."""
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r'[\r\n"]', ' ', regex=True)
        .str.strip()
    )
    return df

def generate_mock_data(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)

    # 1) Requisitions
    job_families = ['Engineering', 'Marketing', 'Sales', 'HR']
    locations    = ['New York', 'San Francisco', 'London', 'Chicago']
    levels       = ['L3', 'L4', 'L5']
    reqs = []
    for i in range(1, 21):
        reqs.append({
            "job_req_id": f"JR{i:03d}",
            "job_family": random.choice(job_families),
            "location":   random.choice(locations),
            "level":      random.choice(levels),
            "status":     random.choice(["Open", "Filled"])
        })
    pd.DataFrame(reqs).to_csv(f"{data_dir}/requisitions.csv", index=False)

    # 2) Applications
    apps = []
    for i in range(200):
        req = random.choice(reqs)
        apps.append({
            "applicant_id":      f"A{i+1000}",
            "job_req_id":        req['job_req_id'],
            "application_date":  faker.date_between(start_date='-180d', end_date='today'),
            "location":          req['location'],
            "job_family":        req['job_family'],
            "level":             req['level'],
            "hired":             random.random() < 0.2
        })
    pd.DataFrame(apps).to_csv(f"{data_dir}/applications.csv", index=False)

    # 3) Interview Feedback
    strong_feedback = [
        "Demonstrated exceptional leadership and technical depth.",
        "Excellent communication skills; very confident.",
        "Highly analytical and collaborative.",
        "Strong problem-solving mindset and culture fit.",
        "Impressive domain knowledge and adaptability."
    ]
    weak_feedback = [
        "Struggled to articulate ideas clearly.",
        "Lacked depth in technical discussion.",
        "Unconvincing examples; low clarity.",
        "Seemed unsure about role expectations.",
        "Minimal engagement during interview."
    ]
    feedback = []
    for app in apps:
        text  = random.choice(strong_feedback if random.random()<0.3 else weak_feedback)
        score = random.choices([2,3,4,5], weights=[1,2,3,2])[0]
        feedback.append({
            "applicant_id":     app['applicant_id'],
            "job_req_id":       app['job_req_id'],
            "interviewer":      faker.user_name(),
            "feedback_text":    text,
            "interview_score":  score
        })
    fb_df = pd.DataFrame(feedback)
    fb_df = sanitize_column(fb_df, 'feedback_text')
    fb_df.to_csv(f"{data_dir}/interview_feedback.csv", index=False)

    # 4) Resumes
    resume_samples = [
        "Experienced software engineer with 8+ years in full-stack development.",
        "Marketing strategist with proven demand-gen expertise.",
        "Enterprise account executive with 10+ years in consultative sales.",
        "HR business partner focused on DEI strategy and workforce planning.",
        "Product manager skilled in agile development and roadmap execution."
    ]
    resumes = []
    for app in apps:
        text = random.choice(resume_samples)
        resumes.append({
            "applicant_id": app['applicant_id'],
            "resume_text":  text
        })
    res_df = pd.DataFrame(resumes)
    res_df = sanitize_column(res_df, 'resume_text')
    res_df.to_csv(f"{data_dir}/resumes.csv", index=False)

    # 5) Job Descriptions
    jds = []
    for req in reqs:
        desc = faker.paragraph(nb_sentences=6)
        jds.append({
            "job_req_id":       req['job_req_id'],
            "job_title":        f"{req['job_family']} Specialist",
            "job_description":  desc
        })
    jds_df = pd.DataFrame(jds)
    jds_df = sanitize_column(jds_df, 'job_description')
    jds_df.to_csv(f"{data_dir}/job_descriptions.csv", index=False)

def load_data(data_dir='data'):
    """Generate mock data if missing, then load all datasets with Python CSV engine."""
    if not os.path.exists(os.path.join(data_dir, 'applications.csv')):
        generate_mock_data(data_dir)

    apps     = pd.read_csv(f"{data_dir}/applications.csv",        engine='python')
    feedback = pd.read_csv(f"{data_dir}/interview_feedback.csv",  engine='python')
    reqs     = pd.read_csv(f"{data_dir}/requisitions.csv",        engine='python')
    resumes  = pd.read_csv(f"{data_dir}/resumes.csv",             engine='python')
    jds      = pd.read_csv(f"{data_dir}/job_descriptions.csv",    engine='python')
    return apps, feedback, reqs, resumes, jds

def match_candidates(apps, feedback, reqs, resumes, jds):
    """Run the Rediscovery v2 matching: feedback + resume→JD similarity, then write output."""
    # Merge all sources
    df = pd.merge(apps, feedback, on=["applicant_id","job_req_id"])
    df = pd.merge(df, reqs,     on="job_req_id", suffixes=('','_req'))
    df = pd.merge(df, resumes,  on="applicant_id")
    df = pd.merge(df, jds,      on="job_req_id")

    # Filter to high-quality feedback
    df = df[df['interview_score'] >= 4]
    hired     = df[df['hired'] == True].reset_index(drop=True)
    not_hired = df[df['hired'] == False].reset_index(drop=True)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = []

    for _, row in not_hired.iterrows():
        # Peer feedback match
        peers = hired[
            (hired['job_family']==row['job_family']) &
            (hired['location']==row['location'])
        ]
        if peers.empty:
            continue

        fb_embs  = model.encode(peers['feedback_text'].tolist(),      convert_to_tensor=True)
        vec_fb   = model.encode(row['feedback_text'],                 convert_to_tensor=True)
        sims_fb  = util.cos_sim(vec_fb, fb_embs)[0]
        idx_fb   = torch.argmax(sims_fb).item()
        score_fb = float(sims_fb[idx_fb])

        # Resume→JD match
        vec_res  = model.encode(row['resume_text'],                  convert_to_tensor=True)
        vec_jd   = model.encode(row['job_description'],              convert_to_tensor=True)
        score_jd = float(util.cos_sim(vec_res, vec_jd)[0])

        # Include if strong in either dimension
        if score_fb>=0.7 or score_jd>=0.7:
            idx_score = round((score_fb + score_jd)/2, 3)
            peer      = peers.iloc[idx_fb]
            results.append({
                "applicant_id":            row["applicant_id"],
                "job_req_id":              row["job_req_id"],
                "job_family":              row["job_family"],
                "location":                row["location"],
                "interview_score":         row["interview_score"],
                "feedback_text":           row["feedback_text"],
                "resume_text":             row["resume_text"],
                "matched_to_hired_id":     peer["applicant_id"],
                "matched_feedback":        peer["feedback_text"],
                "feedback_similarity":     score_fb,
                "resume_jd_similarity":    score_jd,
                "rediscovery_index_score": idx_score
            })

    # Write output
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(results).to_csv("outputs/candidate_matches.csv", index=False)
    print(f"✅ {len(results)} matches written to outputs/candidate_matches.csv")

if __name__ == "__main__":
    apps, feedback, reqs, resumes, jds = load_data()
    match_candidates(apps, feedback, reqs, resumes, jds)
