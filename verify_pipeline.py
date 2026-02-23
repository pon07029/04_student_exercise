import os
import pandas as pd
import json
import traceback
import sys
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from openai import OpenAI

# Fix encoding for Windows Console
sys.stdout.reconfigure(encoding='utf-8')

# 0. Setup
os.environ["OPENAI_API_KEY"] = "sk-proj-YOUR_API_KEY_HERE"

OUTPUT_DIR = "./data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("--- Step 1: Data Preprocessing ---")
try:
    INPUT_CSV_PATH = "./data/sample_data.csv"
    df_raw = pd.read_csv(INPUT_CSV_PATH)
    df = df_raw.head(5).copy()
    
    def format_input_from_row(row):
        symptom = str(row.get('symptom', ''))
        pet_info = str(row.get('pet_information', ''))
        issue_details = str(row.get('issue_details', ''))
        more_details = str(row.get('more_details', ''))
        full_description = f"""
증상 (Symptom):
{symptom}

반려동물 정보 (Pet Information):
{pet_info}

상세 문제 (Issue Details):
{issue_details}

보호자 추가 설명 (More Details from Guardian):
{more_details}
"""
        return full_description.strip()

    df['formatted_prompt'] = df.apply(format_input_from_row, axis=1)
    df_processed = df[['formatted_prompt', 'answer']]
    output_path = os.path.join(OUTPUT_DIR, "processed_sample.csv")
    df_processed.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df_processed)} processed cases to {output_path}")

except Exception as e:
    print(f"FAILED Step 1: {e}")
    traceback.print_exc()

print("\n--- Step 2: Single Agent ---")
try:
    # Use standard GPT-4o
    my_llm = LLM(
        model="gpt-4o"
    )
    
    pet_consultant = Agent(
        role="종합 반려동물 건강 컨설턴트",
        goal="긴급도를 분류하고 답변은 반드시 한국어로 작성하며 적절한 조언을 제공합니다.",
        backstory="수년의 임상 경험을 갖춘 전문 수의사입니다.",
        llm=my_llm,
        verbose=True
    )
    
    consultation_task = Task(
        description="다음 케이스를 분석하세요: {case_input}\n반드시 모든 내용을 한국어로 작성하세요.",
        expected_output="모든 내용이 한국어로 작성된 종합적인 건강 상담 리포트",
        agent=pet_consultant
    )
    
    single_agent_crew = Crew(
        agents=[pet_consultant],
        tasks=[consultation_task],
        verbose=True
    )
    
    df_processed = pd.read_csv(output_path)
    
    for i, row in df_processed.iterrows():
        print(f"Running Single Agent Case #{i+1}...")
        result_single = single_agent_crew.kickoff(inputs={"case_input": row['formatted_prompt']})
        with open(os.path.join(OUTPUT_DIR, f"single_result_{i}.md"), "w", encoding="utf-8") as f:
            f.write(result_single.raw)
            
    print("Step 2 Run Successful")

except Exception as e:
    print(f"FAILED Step 2: {e}")
    traceback.print_exc()

print("\n--- Step 3: Multi Agent ---")
try:
    triage_agent = Agent(
        role="응급 분류 간호사 (Triage)", goal="환자의 긴급도를 가장 먼저 분류합니다", backstory="경험 많은 동물병원 응급실 간호사입니다", llm=my_llm
    )
    vet_agent = Agent(
        role="수의사 (Vet)", goal="증상을 파악하고 질병을 진단합니다", backstory="정확하고 논리적인 전문 수의사입니다", llm=my_llm
    )
    advisor_agent = Agent(
        role="상담가 (Advisor)", goal="보호자에게 결과를 알기 쉽고 친절하게 한국어로 설명합니다", backstory="보호자와 친절하게 소통하는 커뮤니케이션 전문가입니다", llm=my_llm
    )
    
    triage_task = Task(
        description="다음 케이스의 긴급도를 분류하고 한국어로 짧게 요약하세요: {case_input}", expected_output="한국어로 작성된 긴급도 판단 결과 (예: 낮음/보통/높음/응급)", agent=triage_agent
    )
    diagnosis_task = Task(
        description="응급 분류 결과를 바탕으로 예상되는 질병과 원인을 진단하세요. 결과는 한국어로 작성하세요.", expected_output="한국어로 작성된 예상 질병 목록과 진단 사유", agent=vet_agent, context=[triage_task]
    )
    advice_task = Task(
        description="앞선 모든 결과를 종합하여 보호자를 위한 최종 조언과 가이드를 한국어로 작성하세요.", expected_output="한국어로 작성된 보호자를 위한 최종 상담 리포트", agent=advisor_agent, context=[triage_task, diagnosis_task]
    )
    
    multi_crew = Crew(
        agents=[triage_agent, vet_agent, advisor_agent],
        tasks=[triage_task, diagnosis_task, advice_task],
        verbose=True
    )
    
    for i, row in df_processed.iterrows():
        print(f"Running Multi Agent Case #{i+1}...")
        result_multi = multi_crew.kickoff(inputs={"case_input": row['formatted_prompt']})
        with open(os.path.join(OUTPUT_DIR, f"multi_result_{i}.md"), "w", encoding="utf-8") as f:
            f.write(result_multi.raw)
            
    print("Step 3 Run Successful")

except Exception as e:
    print(f"FAILED Step 3: {e}")
    traceback.print_exc()

print("\n--- Step 4: Evaluation ---")
try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    def evaluate(answer, truth):
        prompt = f"""실제 수의사의 정답(Ground Truth)을 기준으로 학생(AI)이 생성한 답변을 평가하세요.
        실제 수의사 정답: {truth}
        학생(AI) 답변: {answer}
        Accuracy(정확도) 항목에 대해 0에서 10점 사이의 점수를 매겨 JSON 형태로만 출력하세요."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    results = {"Single": [], "Multi": [], "Vet": []}
    
    for i in range(len(df_processed)):
        print(f"Evaluating Case #{i+1}...")
        ground_truth = df_processed.iloc[i]['answer']
        
        try:
            with open(os.path.join(OUTPUT_DIR, f"single_result_{i}.md"), "r", encoding="utf-8") as f:
                s_res = f.read()
        except FileNotFoundError:
            s_res = "File not found."
            
        try:
            with open(os.path.join(OUTPUT_DIR, f"multi_result_{i}.md"), "r", encoding="utf-8") as f:
                m_res = f.read()
        except FileNotFoundError:
            m_res = "File not found."
            
        score1 = evaluate(s_res, ground_truth)
        score2 = evaluate(m_res, ground_truth)
        score3 = evaluate(ground_truth, ground_truth)
        
        results["Single"].append(score1)
        results["Multi"].append(score2)
        results["Vet"].append(score3)
        print(f"Case {i+1} Scores: Single={score1}, Multi={score2}, Vet={score3}")
        
    print("Generating visualization...")
    import matplotlib.pyplot as plt
    x = range(1, len(df_processed) + 1)
    s_scores = [s.get("Accuracy", 0) for s in results["Single"]]
    m_scores = [s.get("Accuracy", 0) for s in results["Multi"]]
    v_scores = [s.get("Accuracy", 0) for s in results["Vet"]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, s_scores, marker='o', label='Single Agent')
    plt.plot(x, m_scores, marker='s', label='Multi Agent')
    plt.plot(x, v_scores, marker='^', label='Vet (Ground Truth)', color='green')
    plt.title('AI vs Vet: Evaluation Scores (Accuracy)')
    plt.xlabel('Case Number')
    plt.ylabel('Score (0-10)')
    plt.xticks(x)
    plt.ylim(0, 10.5)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'evaluation_plot.png'))
    print(f"Plot saved to {os.path.join(OUTPUT_DIR, 'evaluation_plot.png')}")
    
    print("Step 4 Run Successful")

except Exception as e:
    print(f"FAILED Step 4: {e}")
    traceback.print_exc()
