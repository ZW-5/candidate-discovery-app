
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

def load_data(data_dir='data'):
    apps = pd.read_csv(f"{data_dir}/applications.csv")
    feedback = pd.read_csv(f"{data_dir}/interview_feedback.csv")
    reqs = pd.read_csv(f"{data_dir}/requisitions.csv")
    resumes = pd.read_csv(f"{data_dir}/resumes.csv")
    jds = pd.read_csv(f"{data_dir}/job_descriptions.csv")
    return apps, feedback, reqs, resumes, jds

def match_candidates(apps, feedback, reqs, resumes, jds):
    df = pd.merge(apps, feedback, on=["applicant_id", "job_req_id"])
    df = pd.merge(df, reqs, on="job_req_id", suffixes=('', '_req'))
    df = pd.merge(df, resumes, on="applicant_id")
    df = pd.merge(df, jds, on="job_req_id")

    # Filter to high quality feedback only
    df = df[df['interview_score'] >= 4]
    hired = df[df['hired'] == True].reset_index(drop=True)
    not_hired = df[df['hired'] == False].reset_index(drop=True)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    results = []

    for _, row in not_hired.iterrows():
        # Match to strong hired peers (feedback similarity)
        hired_subset = hired[
            (hired['job_family'] == row['job_family']) &
            (hired['location'] == row['location'])
        ]
        if hired_subset.empty:
            continue

        hired_embeddings = model.encode(hired_subset['feedback_text'].tolist(), convert_to_tensor=True)
        vec = model.encode(row['feedback_text'], convert_to_tensor=True)
        sims = util.cos_sim(vec, hired_embeddings)[0]
        best_idx = torch.argmax(sims).item()
        feedback_score = float(sims[best_idx])

        # Resume ↔ JD similarity score
        resume_vec = model.encode(row['resume_text'], convert_to_tensor=True)
        jd_vec = model.encode(row['job_description'], convert_to_tensor=True)
        jd_score = float(util.cos_sim(resume_vec, jd_vec)[0])

        if feedback_score >= 0.7 or jd_score >= 0.7:
            index_score = round((feedback_score + jd_score) / 2, 3)
            best_match = hired_subset.iloc[best_idx]
            results.append({
                "applicant_id": row["applicant_id"],
                "job_req_id":      row["job_req_id"],   
                "job_family": row["job_family"],
                "location": row["location"],
                "interview_score": row["interview_score"],
                "feedback_text": row["feedback_text"],
                "resume_text": row["resume_text"],
                "matched_to_hired_id": best_match["applicant_id"],
                "matched_feedback": best_match["feedback_text"],
                "feedback_similarity": feedback_score,
                "resume_jd_similarity": jd_score,
                "rediscovery_index_score": index_score
            })

    out = pd.DataFrame(results)
    out.to_csv("outputs/candidate_matches.csv", index=False)
    print(f"✅ {len(results)} rediscovery matches saved to outputs/candidate_matches.csv")

if __name__ == "__main__":
    apps, feedback, reqs, resumes, jds = load_data()
    match_candidates(apps, feedback, reqs, resumes, jds)
