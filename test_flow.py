import requests, time, json
BASE = "http://127.0.0.1:8765"

# Step 1: Upload john_doe PDF
print("=== STEP 1: Upload ===")
with open(r"C:\Users\Dell\Downloads\john_doe_synthetic_medical_report_table_image_different.pdf", "rb") as f:
    r = requests.post(BASE+"/api/upload", data=f.read(), headers={"X-Filename":"john_doe_synthetic_medical_report_table_image_different.pdf","Content-Type":"application/octet-stream"})
job = r.json()
job_id = job.get("job_id","")
print("Upload started, job_id:", job_id)

# Poll until done
for i in range(120):
    time.sleep(3)
    s = requests.get(BASE+"/api/upload/status?job_id="+job_id).json()
    st = s.get("status")
    print("  [%d] %s %s%%" % (i, s.get("stage","?"), s.get("progress_percent","?")))
    if st in ("completed","error"):
        print("\nStatus:", st)
        print("Elapsed:", s.get("elapsed_seconds"), "seconds")
        print("OCR Accuracy:", s.get("ocr_accuracy"))
        print("OCR Detail:", json.dumps(s.get("ocr_detail")))
        break

# Step 2: Ask a question
print("\n=== STEP 2: Ask question ===")
r2 = requests.post(BASE+"/api/ask", json={
    "actor_name": "test_user",
    "role": "doctor",
    "question": "What medication is prescribed to John Doe?",
    "candidate_k": 10,
    "top_k": 7,
})
ans = r2.json()
print("Answer:", ans.get("answer","")[:200])
print("Retrieval Accuracy:", ans.get("retrieval_accuracy"))
print("Answer Evaluation:", json.dumps(ans.get("answer_evaluation"), indent=2))
