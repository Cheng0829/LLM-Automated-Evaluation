import os, time, sys, random, re
from tqdm import tqdm
import tiktoken
from datetime import datetime
from openai import OpenAI
from datasets import load_dataset, concatenate_datasets

provider_list = {
    "aliyun": {
        # "api_key" : os.getenv("BALIYUN_API_KEY"),
        "api_key" : "sk-973a8ecb703f48db906c3cf2ebe7b3b6",
        "base_url" : "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_list": ["deepseek-v3", "deepseek-r1", "qwen-max-0125", "qwen-turbo-1101", "qwen-vl-max-0125", "qwq-plus"]
    },
    "bytedance": {
        # "api_key" : os.getenv("BYTEDANCE_API_KEY"),
        "api_key" : "6c693abd-8e7c-4227-abaa-972200f7e7d5",
        "base_url" : "https://ark.cn-beijing.volces.com/api/v3",
        "model_list": ["deepseek-v3-241226", "deepseek-r1-250120", "doubao-1-5-pro-32k-250115", "doubao-1-5-pro-256k-250115"]
    },
    "baidu": {
        "api_key" : os.getenv("BAIDUYUN_API_KEY"),
        "base_url" : "https://qianfan.baidubce.com/v2",
        "model_list": ["deepseek-v3", "deepseek-r1"]
    },
    "tencent": {
        "api_key" : os.getenv("TENGXUNYUN_API_KEY"),
        "base_url" : "https://api.lkeap.cloud.tencent.com/v1",
        "model_list": ["deepseek-v3", "deepseek-r1"]
    },
    "deepseek": {
        "api_key" : os.getenv("DEEPSEEK_API_KEY"),
        "base_url" : "https://api.deepseek.com",
        "model_list": ["deepseek-chat", "deepseek-reasoner"]
    },
}

def llm(provider_name, model_name, system_prompt="", user_prompt="", stream=True, temperature=0.6, encoding=tiktoken.encoding_for_model("gpt-4"), max_tokens=None):

    old_timestamp = int(time.time()) # 发送时的时间戳

    client = OpenAI(
        api_key = provider_list[provider_name]["api_key"],
        base_url = provider_list[provider_name]["base_url"]
    )
    completion = client.chat.completions.create(

        model=model_name,

        messages = [
            {'role': 'system', 'content': "你是一个做题专家。请完成下列单项选择题。\n\n## output format\n只能输出一个选项编号字母，不要有解析等其他任何内容。"},
            {'role': 'user', 'content': user_prompt},
        ],
        stream=stream,
        temperature=temperature,
    )

    # 流式输出
    if stream:
        response = ""
        start_think_flag = 1
        end_think_flag = 1
        start_think_string = "<think>\n"
        end_think_string = "</think>\n"
        for msg in completion:
            delta = msg.choices[0].delta
            if 'reasoning_content' in delta.model_extra and delta.model_extra['reasoning_content'] is not None:
                text_delta = delta.model_extra['reasoning_content']
                text_delta = start_think_string * start_think_flag + text_delta # 添加"<think>"
                start_think_flag = 0
                response += text_delta
                # print(text_delta, end='')
            if delta.content:
                text_delta = delta.content
                if start_think_flag == 0:
                    text_delta = end_think_string * end_think_flag + text_delta # 添加"</think>"
                    end_think_flag = 0
                # print(text_delta, end='')
                response += text_delta
            if max_tokens != None and len(encoding.encode(response)) > max_tokens:
                break            
    # 非流式输出
    else:
        response = completion.choices[0].message.content
        if 'reasoning_content' in response.choices[0].message.model_extra:
            response = f"<think>\n{response.choices[0].message.model_extra['reasoning_content']}\n</think>\n\n" + response
    tokens_num = len(encoding.encode(response)) + len(encoding.encode(system_prompt)) + len(encoding.encode(user_prompt))
    # 去除答案中的think部分
    if '<think>' in response and '</think>' in response:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    # print(response)
    return response, tokens_num
        

dataset_list = []

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

# for category in task_map:
#     task_list = list(task_map[category].keys())
#     for i in range(0, len(task_list)):
#         task_list[i] = task_list[i].lower()
#         dataset_tmp = load_dataset(r"ceval/ceval-exam", name=task_list[i])
#         dataset = concatenate_datasets([dataset_tmp["dev"], dataset_tmp["val"]])
#         dataset_list.append(dataset)
#         print(len(dataset_list), end=' ')

def test_split():
    encoding = tiktoken.encoding_for_model("gpt-4")
    # 加载数据集
    sum_correct = 0
    sum_total = 0
    order = 0
    for category in task_map:
        category_tokens_sum = 0
        task_list = list(task_map[category].keys())        
        for i in range(0, len(task_list)):
            # input("waiting: ")
            task_chinese_name = task_map[category][task_list[i]]
            task_list[i] = task_list[i].lower()
            # try:
            dataset_tmp = load_dataset(r"ceval/ceval-exam", name=task_list[i])
            dataset = concatenate_datasets([dataset_tmp["dev"], dataset_tmp["val"]])
            dataset_tokens_sum = 0
            for item in dataset:
                question = item['question']
                token_num = len(encoding.encode(question))
                dataset_tokens_sum += token_num
            category_tokens_sum += dataset_tokens_sum
            print(f"{category}-dataset-token_num={dataset_tokens_sum}")
        
        print(f"\ncategory_sum_tokens={category_tokens_sum}")
test_split()