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
task_map = {
    "STEM": {
        "Electrical_Engineer": "注册电气工程师", # 42
        "Metrology_Engineer": "注册计量师", # 29
        "College_Programming": "大学编程", # 42
        "Computer_Architecture": "计算机组成", # 26
        "Operating_System": "操作系统", # 24
        "Computer_Network": "计算机网络", # 24
        "Discrete_Mathematics": "离散数学", # 21
        "Probability_and_Statistics": "概率统计", # 23
        "Advanced_Mathematics": "高等数学", # 24
        "College_Chemistry": "大学化学", # 29
        "College_Physics": "大学物理", # 24
        "Veterinary_Medicine": "兽医学", # 28
        "High_School_Biology": "高中生物", # 24
        "High_School_Chemistry": "高中化学", # 24
        "High_School_Physics": "高中物理", # 24
        "High_School_Mathematics": "高中数学", # 23
        "Middle_School_Chemistry": "初中化学", # 25
        "Middle_School_Physics": "初中物理", # 24
        "Middle_School_Biology": "初中生物", # 26
        "Middle_School_Mathematics": "初中数学", # 24
    },
    "Social_Science": {
        "Teacher_Qualification": "教师资格", # 49
        "Business_Administration": "工商管理", # 38
        "Mao_Zedong_Thought": "毛泽东思想和中国特色社会主义理论体系概论", # 29
        "Marxism": "马克思主义基本原理", # 24
        "College_Economics": "大学经济学", # 60
        "Education_Science": "教育学", # 34
        "High_School_Geography": "高中地理", # 24
        "High_School_Politics": "高中政治", # 24
        "Middle_School_Geography": "初中地理", # 17
        "Middle_School_Politics": "初中政治", # 26
    },
    "Humanity": {
        "Professional_Tour_Guide": "导游资格", # 34
        "Legal_Professional": "法律职业资格", # 28
        "Art_Studies": "艺术学", # 38
        "Chinese_Language_and_Literature": "中国语言文学", # 28
        "Law": "法学", # 29
        "Logic": "逻辑学", # 27
        "Ideological_and_Moral_Cultivation": "思想道德修养与法律基础", # 24
        "Modern_Chinese_History": "近代史纲要", # 28
        "High_School_History": "高中历史", # 25
        "High_School_Chinese": "高中语文", # 24
        "Middle_School_History": "初中历史", # 27
    },
    "Other": {
        "Environmental_Impact_Assessment_Engineer": "环境影响评价工程师", # 36
        "Urban_and_Rural_Planner": "注册城乡规划师", # 51
        "Fire_Engineer": "注册消防工程师", # 36
        "Physician": "医师资格", # 54
        "Tax_Accountant": "税务师", # 54
        "Accountant": "注册会计师", # 54
        "Civil_Servant": "公务员", # 52
        "Clinical_Medicine": "临床医学", # 27
        "Basic_Medicine": "基础医学", # 24
        "Plant_Protection": "植物保护", # 27
        "Sports_Science": "体育学", # 24
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