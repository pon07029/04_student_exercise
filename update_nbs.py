import json

new_source = [
    "# ==========================================\n",
    "# 다양한 방식의 시각화 (Visualization)\n",
    "# ==========================================\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "\n",
    "if avg_single and avg_multi and avg_vet:\n",
    "    df_scores = pd.DataFrame({\n",
    "        'Metric': list(avg_single.keys()),\n",
    "        'Single Agent': list(avg_single.values()),\n",
    "        'Multi Agent': list(avg_multi.values()),\n",
    "        'Vet (Ground Truth)': list(avg_vet.values())\n",
    "    }).set_index('Metric')\n",
    "\n",
    "    # 스타일과 폰트 설정\n",
    "    sns.set_theme(style=\"whitegrid\")\n",
    "    plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 깨짐 방지 (윈도우)\n",
    "    plt.rcParams['axes.unicode_minus'] = False\n",
    "    \n",
    "    # 그래프 사이즈 설정\n",
    "    plt.figure(figsize=(18, 5))\n",
    "\n",
    "    # 1. 막대 그래프 (Bar Chart)\n",
    "    plt.subplot(1, 3, 1)\n",
    "    df_scores.plot(kind='bar', ax=plt.gca(), rot=0)\n",
    "    plt.title('Performance Comparison (Bar Chart)')\n",
    "    plt.ylabel('Score (0-10)')\n",
    "    plt.ylim(0, 10)\n",
    "    plt.legend(loc='lower right')\n",
    "\n",
    "    # 2. 선 그래프 (Line Chart)\n",
    "    plt.subplot(1, 3, 2)\n",
    "    df_scores.plot(kind='line', marker='o', ax=plt.gca())\n",
    "    plt.title('Performance Trends (Line Chart)')\n",
    "    plt.ylabel('Score (0-10)')\n",
    "    plt.ylim(0, 10)\n",
    "    plt.grid(True)\n",
    "    plt.legend(loc='lower right')\n",
    "\n",
    "    # 3. 레이더 차트 (Radar Chart)를 위한 데이터 준비\n",
    "    ax = plt.subplot(1, 3, 3, polar=True)\n",
    "    categories = df_scores.index.tolist()\n",
    "    N = len(categories)\n",
    "    \n",
    "    angles = [n / float(N) * 2 * np.pi for n in range(N)]\n",
    "    angles += angles[:1]\n",
    "    \n",
    "    for column in df_scores.columns:\n",
    "        values = df_scores[column].tolist()\n",
    "        values += values[:1]\n",
    "        ax.plot(angles, values, linewidth=2, linestyle='solid', label=column)\n",
    "        ax.fill(angles, values, alpha=0.1)\n",
    "\n",
    "    plt.xticks(angles[:-1], categories)\n",
    "    ax.set_rlabel_position(0)\n",
    "    plt.yticks([2, 4, 6, 8, 10], [\"2\", \"4\", \"6\", \"8\", \"10\"], color=\"grey\", size=8)\n",
    "    plt.ylim(0, 10)\n",
    "    plt.title('Performance Radar Chart')\n",
    "    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n"
]

files_to_update = [
    'c:/Users/SNUVBMI5/Desktop/JHJ/AIChatVet/04_student_exercise/00_AIChatVet_Full_Course.ipynb',
    'c:/Users/SNUVBMI5/Desktop/JHJ/AIChatVet/04_student_exercise/4_Evaluation.ipynb'
]

for file_path in files_to_update:
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 마지막 코드 셀을 찾아서 시각화 부분 교체
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = ''.join(cell['source'])
            if 'import matplotlib.pyplot' in source_str or '시각화' in source_str:
                cell['source'] = new_source
                print(f"Updated {file_path}")
                break
                
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
