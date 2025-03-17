import requests
from datasets import load_dataset, concatenate_datasets
import re, random, json
from tqdm import tqdm
import re, time, tiktoken, ollama
from ollama import ChatResponse
from ollama import Options


def llm(model, query, temperature=0.6, stream=True, encoding=tiktoken.encoding_for_model("gpt-4"), max_tokens=None):
    # return "A"
    options = Options(
        temperature=temperature,
        # num_gpu=0, # num_gpu=0即使用CPU计算
        # num_thread=32,
        # num_ctx=4096, # 上下文窗口大小
    )
    # 流式输出
    response = ollama.chat(
        model=model,
        messages=[
            # system_prompt的token数：44，1606道题，1606*44=70664
            {
                "role": "system",
                "content": "你是一个做题专家。请完成下列单项选择题。",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        options=options,
        stream=stream,
        format={
            "type": "object",
            "properties": {
                "choice": {
                    "type": "string"
                },
            },
            "required": [
                "choice"
            ]
        },
        keep_alive=5
    )
    if stream:
        chunks = ""
        # 逐块打印响应内容
        for chunk in response:
            chunks += chunk["message"]["content"]
            # print(chunk["message"]["content"], end="", flush=True)
            if max_tokens != None and len(encoding.encode(chunks)) > max_tokens:
                break
        response = chunks
    else:
        # print(response["message"]["content"])
        response = response["message"]["content"]
    # print()
    # print(response)
    # stream=True时无效并卡死程序
    # with open("tmp.txt", "a", encoding="utf-8") as f:
    #     f.write(response + "\n"+ 100*'*' + '\n')
    if '<think>' in response and '</think>' in response:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    '{"choice":"D"}'

    response = json.loads(response)
    response = response['choice'] if 'choice' in response else response

    return response.strip()

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

def test_split(model_name, is_inference=False):
    encoding = tiktoken.encoding_for_model("gpt-4")
    model_name_write = model_name.replace(":", "_").replace("/", "_")
    # with open(f"assests/{model_name_write}.txt", "w", encoding="utf-8") as f:
    #     f.write(f"")
    # 加载数据集
    sum_correct = 0
    sum_total = 0
    order = 0
            
    for category in task_map:
        category_correct = 0
        category_total = 0
        task_list = list(task_map[category].keys())        
        for i in range(0, len(task_list)):
            task_chinese_name = task_map[category][task_list[i]]
            task_list[i] = task_list[i].lower()
            # try:
            dataset_tmp = load_dataset(r"ceval/ceval-exam", name=task_list[i])
            dataset = concatenate_datasets([dataset_tmp["dev"], dataset_tmp["val"]])
            # dataset = dataset_list[order]
            print(f"\nNo.{order}: {category}-{task_list[i]}({task_chinese_name})数据集加载完成, len(dataset)={len(dataset)}")
            # except:
                # print(f"\nNo.{order}: {category}-{task_list[i]}({task_chinese_name})数据集加载失败")
                # continue
            if is_inference:
                # dataset只选取其中最多2条数据
                random_indices = random.sample(range(len(dataset)), 2)
                dataset = dataset.select(random_indices)

            # 初始化统计变量
            correct = 0
            total = len(dataset)
            for item in tqdm(dataset, desc=f"Processing"):
                # try:
                # 构造完整问题
                user_prompt = f"{item['question']}\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n答案："

                # 调用Ollama API
                model_answer = llm(model_name, user_prompt, stream=True, encoding=encoding, max_tokens=4096)
                # 提取并验证答案
                """从模型输出中提取答案选项（A/B/C/D）"""
                match = re.search(r"[A-D]", model_answer.upper())
                extracted = match.group(0) if match else None
                if extracted and extracted == item["answer"]:
                    correct += 1
                # except:
                #     print("\nerror.")
            # 输出结果
            sum_correct += correct
            sum_total += total
            category_correct += correct
            category_total += total
            print(f"No.{order}: {category}-{task_list[i]}({task_chinese_name})数据集准确率: {correct}/{total} = {correct/total:.2%}")
            with open(f"assests/{model_name_write}.txt", "a", encoding="utf-8") as f:
                f.write(f"No.{order}: {category}-{task_list[i]}({task_chinese_name})数据集准确率: {correct}/{total} = {correct/total:.2%}\n\n")
            order += 1

        print(f"类别{category}平均准确率: {category_correct}/{category_total} = {category_correct/category_total:.2%}")
        with open(f"assests/{model_name_write}.txt", "a", encoding="utf-8") as f:
            f.write(f"类别{category}平均准确率: {category_correct}/{category_total} = {category_correct/category_total:.2%}\n\n")

    with open(f"assests/{model_name_write}.txt", "a", encoding="utf-8") as f:
        f.write(f"总准确率: {sum_correct}/{sum_total} = {sum_correct/sum_total:.2%}\n\n")
    print(f"总准确率: {sum_correct}/{sum_total} = {sum_correct/sum_total:.2%}")

test_split(model_name="qwen2.5:0.5b")
# test_split(model_name="qwen2.5:0.5b-instruct-fp16")
# test_split(model_name="qwen2.5:1.5b")
# test_split(model_name="qwen2.5:1.5b-instruct-fp16")
# test_split(model_name="qwen2.5:3b")
# test_split(model_name="qwen2.5:3b-instruct-q8_0")
# test_split(model_name="huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M")
# test_split(model_name="qwen2.5:7b")
# test_split(model_name="qwen2.5:7b-instruct-q5_K_M")
# test_split(model_name="gemma3:4b")
# test_split(model_name="gemma3:4b-it-q8_0")
# test_split(model_name="gemma3:12b")
# test_split(model_name="phi4")
# test_split(model_name="deepseek-r1:1.5b")
# test_split(model_name="deepseek-r1:1.5b-qwen-distill-fp16")
# test_split(model_name="deepseek-r1-7b")
