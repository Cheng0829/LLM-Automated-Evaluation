# LLM自动化评估

使用[速度评估.py](./速度评估.py)进行测速，使用[准确率评测.py](./自动化评测.py)测速模型准确率。评估结果保存在[assests](/assests)文件夹里面

**数据集来源**：[ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)

**模型来源**：[ollama](https://ollama.com)

**Device**: AMD 7940HX + 4060(8GB) + 64GB RAM

> 所有q4、q5量化均为对应_K_M量化，所有模型均为指令微调后的instruct/chat模型，非base预训练模型。

| Model                    | Size (GB) | Speed (token/s) | Accuracy (%) |
|--------------------------|-----------|-----------------|--------------|
| qwen2.5-0.5b-q4          |    0.4    |      346.98     |    32.94     |
| qwen2.5-0.5b-fp16        |    1      |      89.08      |    38.98     |
| qwen2.5-1.5b-q4          |    1      |      50.84      |    59.65     |
| qwen2.5-1.5b-fp16        |    3.1    |      32.19      |    61.83     |
| qwen2.5-3b-q4            |    1.9    |      71.48      |    65.01     |
| qwen2.5-3b-q8            |    3.3    |      42.15      |    66.38     |
| qwen2.5-7b-q4-abliterate |    4.7    |      29.46      |    71.79     |
| qwen2.5-7b-q4            |    4.7    |      41.15      |    76.77     |
| qwen2.5-7b-q5            |    5.4    |      26.28      |    77.71     |
| qwen2.5-7b-fp16*         |    15     |        /        |    78.79     |
| qwen-max-0125*           |     /     |        /        |    86.85     |
| qwen-turbo-1101*         |     /     |        /        |    82.75     |
| gemma3-4b-q4             |    3.3    |      56.04      |    42.47     |
| gemma3-4b-q8             |    5.0    |      38.72      |    44.33     |
| gemma3-12b-q4            |    8.1    |      10.87      |    56.16     |
| phi4-14b-q4              |    9.1    |      7.37       |    57.53     |
| deepseek-r1-1.5b-q4      |    1.1    |      130.0      |    36.11     |
| deepseek-r1-1.5b-fp16    |    3.6    |      71.63      |    37.73     |
| deepseek-r1-7b-q4        |    4.7    |      47.01      |    50.44     |
| deepseek-v3-671b-fp16*   |    1300   |        /        |    82.07     |
| doubao-1.5pro-32k*       |     /     |        /        |    90.10     |

各类别准确率评估(%)：

| Model                    | Average | STEM  | Social Science | Humanity | Other |
|--------------------------|---------|-------|----------------|----------|-------|
| qwen2.5-0.5b-q4          |  32.94  | 26.98 |     43.38      |  32.05   | 33.03 |
| qwen2.5-0.5b-fp16        |  38.98  | 32.26 |     52.00      |  40.71   | 36.22 |
| qwen2.5-1.5b-q4          |  59.65  | 54.53 |     67.38      |  63.78   | 57.18 |
| qwen2.5-1.5b-fp16        |  61.83  | 55.47 |     69.23      |  66.67   | 60.59 |
| qwen2.5-3b-q4            |  65.01  | 58.68 |     76.62      |  69.55   | 60.82 |
| qwen2.5-3b-q8            |  66.38  | 59.81 |     79.08      |  73.08   | 60.14 |
| qwen2.5-7b-q4-abliterate |  71.79  | 65.09 |     81.23      |  73.40   | 71.75 |
| qwen2.5-7b-q4            |  76.77  | 68.30 |     85.54      |  80.13   | 78.13 |
| qwen2.5-7b-q5            |  77.71  | 70.75 |     85.23      |  80.45   | 78.59 |
| qwen2.5-7b-fp16*         |  78.79  | 72.08 |     86.46      |  81.61   | 79.22 |
| qwen-max-0125*           |  86.85  | 83.40 |     91.38      |  88.39   | 86.56 |
| qwen-turbo-1101*         |  82.75  | 79.06 |     89.54      |  84.52   | 81.32 |
| gemma3-4b-q4             |  42.47  | 39.81 |     49.54      |  42.95   | 40.09 |
| gemma3-4b-q8             |  44.33  | 41.89 |     50.46      |  41.99   | 44.42 |
| gemma3-12b-q4            |  56.16  | 53.77 |     63.69      |  52.88   | 55.81 |
| phi4-14b-q4              |  57.53  | 59.25 |     67.69      |  53.53   | 50.80 |
| deepseek-r1-1.5b-q4      |  36.11  | 35.47 |     44.62      |  32.69   | 33.03 |
| deepseek-r1-1.5b-fp16    |  37.73  | 40.75 |     45.23      |  29.17   | 40.75 |
| deepseek-r1-7b-q4        |  50.44  | 50.57 |     59.69      |  46.15   | 46.47 |
| deepseek-v3-671b*        |  82.07  | 80.19 |     88.00      |  83.97   | 78.59 |
| doubao-1.5pro-32k*       |  90.10  | 86.23 |     92.31      |  93.27   | 90.89 |

数据集分类：

```python
# ceval分为4类，共52个数据集，共1606道题，总token数82070
# STEM有20个数据集530道题，Social_Science有10个数据集325道题，Humanity有11个数据集312道题，Other有11个数据集439道题
task_map = {
    # STEM数据量: 530, token数: 25308
    "STEM": {
        "Electrical_Engineer": "注册电气工程师", # 数据量: 42, token数: 2059
        "Metrology_Engineer": "注册计量师", # 数据量: 29, token数: 1064
        "College_Programming": "大学编程", # 数据量: 42, token数: 1773
        "Computer_Architecture": "计算机组成", # 数据量: 26, token数: 815
        "Operating_System": "操作系统", # 数据量: 24, token数: 489
        "Computer_Network": "计算机网络", # 数据量: 24, token数: 889
        "Discrete_Mathematics": "离散数学", # 数据量: 21, token数: 626
        "Probability_and_Statistics": "概率统计", # 数据量: 23, token数: 2196
        "Advanced_Mathematics": "高等数学", # 数据量: 24, token数: 1429
        "College_Chemistry": "大学化学", # 数据量: 29, token数: 1074
        "College_Physics": "大学物理", # 数据量: 24, token数: 1913
        "Veterinary_Medicine": "兽医学", # 数据量: 28, token数: 1021
        "High_School_Biology": "高中生物", # 数据量: 24, token数: 1371
        "High_School_Chemistry": "高中化学", # 数据量: 24, token数: 1271
        "High_School_Physics": "高中物理", # 数据量: 24, token数: 1262
        "High_School_Mathematics": "高中数学", # 数据量: 23, token数: 1724
        "Middle_School_Chemistry": "初中化学", # 数据量: 25, token数: 1031
        "Middle_School_Physics": "初中物理", # 数据量: 24, token数: 930
        "Middle_School_Biology": "初中生物", # 数据量: 26, token数: 964
        "Middle_School_Mathematics": "初中数学", # 数据量: 24, token数: 1407
    },
    # Social_Science数据量: 325, token数: 16771
    "Social_Science": {
        "Teacher_Qualification": "教师资格", # 数据量: 49, token数: 2877
        "Business_Administration": "工商管理", # 数据量: 38, token数: 1692
        "Mao_Zedong_Thought": "毛泽东思想和中国特色社会主义理论体系概论", # 数据量: 29, token数: 958
        "Marxism": "马克思主义基本原理", # 数据量: 24, token数: 542
        "College_Economics": "大学经济学", # 数据量: 60, token数: 2418
        "Education_Science": "教育学", # 数据量: 34, token数: 960
        "High_School_Geography": "高中地理", # 数据量: 24, token数: 887
        "High_School_Politics": "高中政治", # 数据量: 24, token数: 3731
        "Middle_School_Geography": "初中地理", # 数据量: 17, token数: 598
        "Middle_School_Politics": "初中政治", # 数据量: 26, token数: 2108
    },
    # Humanity数据量: 312, token数: 17102
    "Humanity": {
        "Professional_Tour_Guide": "导游资格", # 数据量: 34, token数: 1017
        "Legal_Professional": "法律职业资格", # 数据量: 28, token数: 2260
        "Art_Studies": "艺术学", # 数据量: 38, token数: 1025
        "Chinese_Language_and_Literature": "中国语言文学", # 数据量: 28, token数: 666
        "Law": "法学", # 数据量: 29, token数: 1894
        "Logic": "逻辑学", # 数据量: 27, token数: 4233
        "Ideological_and_Moral_Cultivation": "思想道德修养与法律基础", # 数据量: 24, token数: 524
        "Modern_Chinese_History": "近代史纲要", # 数据量: 28, token数: 1117
        "High_School_History": "高中历史", # 数据量: 25, token数: 1852
        "High_School_Chinese": "高中语文", # 数据量: 24, token数: 1139
        "Middle_School_History": "初中历史", # 数据量: 27, token数: 1375
    },
    # Other数据量: 439, token数: 22889
    "Other": {
        "Environmental_Impact_Assessment_Engineer": "环境影响评价工程师", # 数据量: 36, token数: 2271
        "Urban_and_Rural_Planner": "注册城乡规划师", # 数据量: 51, token数: 1727
        "Fire_Engineer": "注册消防工程师", # 数据量: 36, token数: 1917
        "Physician": "医师资格", # 数据量: 54, token数: 1552
        "Tax_Accountant": "税务师", # 数据量: 54, token数: 2880
        "Accountant": "注册会计师", # 数据量: 54, token数: 3426
        "Civil_Servant": "公务员", # 数据量: 52, token数: 6052
        "Clinical_Medicine": "临床医学", # 数据量: 27, token数: 1102
        "Basic_Medicine": "基础医学", # 数据量: 24, token数: 485
        "Plant_Protection": "植物保护", # 数据量: 27, token数: 843
        "Sports_Science": "体育学", # 数据量: 24, token数: 634
    },
}
```
