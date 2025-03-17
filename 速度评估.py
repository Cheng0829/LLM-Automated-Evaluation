import re, time, tiktoken, ollama
from ollama import ChatResponse
from ollama import Options

def llm(model, query, temperature=0.6, stream=True):
    options = Options(
            temperature=temperature,
            # num_gpu=0, # num_gpu=0即为使用CPU计算
            # num_thread=32,
            # num_ctx=4096, # 上下文窗口大小
        )
    # 流式输出
    response = ollama.chat(
        model=model,
        messages=[
            # {
            #     "role": "system",
            #     "content": "你是一个做题专家。请完成下列单项选择题。",
            # },
            {
                "role": "user",
                "content": query,
            },
        ],
        options=options,
        # options={"temperature": temperature},
        stream=stream,
        keep_alive=5
    )
    if stream:
        chunks = ""
        # 逐块打印响应内容
        for chunk in response:
            chunks += chunk["message"]["content"]
            # print(chunk["message"]["content"], end="", flush=True)
        return chunks
    else:
        # print(response["message"]["content"])
        return response["message"]["content"]

def main(model, user_query="你是谁", stream=False):
    # "请用我给你的4个数字，通过加、减、乘、除、括号，组成一个运算，使得结果为24。注意：数字需要全部使用我提供的数字：4 4 6 8。"
    encoding = tiktoken.encoding_for_model("gpt-4")
    old_time = time.time()
    # print("LLM: ", end='')
    # deepseek-r1-7b
    # qwen2.5:3b-instruct-q8_0
    # qwen2.5:7b-instruct-q5_K_M
    # qwq:32b-q4_K_M
    llm_response = llm(model, user_query, stream=stream)
    input_tokens = len(encoding.encode(user_query))
    output_tokens = len(encoding.encode(llm_response))
    # print("\n\n输入token数：{}，输出token数：{}".format(input_tokens, output_tokens))
    # print("time：{}s.".format(round(time.time() - old_time, 2)))
    # print("speed: {} token/s.".format(round(output_tokens / (time.time() - old_time), 2)))
    return output_tokens / (time.time() - old_time)

if __name__ == "__main__":
    model_list = [
        # "qwen2.5:0.5b",
        # "qwen2.5:0.5b-instruct-fp16",
        # "qwen2.5:1.5b",
        # "qwen2.5:1.5b-instruct-fp16",
        # "qwen2.5:3b",
        # "qwen2.5:3b-instruct-q8_0",
        # "huihui_ai/qwen2.5-abliterate:7b-instruct-q4_K_M",
        # "qwen2.5:7b",
        # "qwen2.5:7b-instruct-q5_K_M",
        # "gemma3:4b",
        # "gemma3:4b-it-q8_0",
        "gemma3:12b",
        "phi4",
        # "deepseek-r1:1.5b",
        # "deepseek-r1:1.5b-qwen-distill-fp16",
        # "deepseek-r1-7b",
    ]
    user_query_list = [
        "下列选项中，符合原癌基因概念的是____ A.正常基因, B.细胞癌基因, C.肿瘤抑制基因, D.凋亡调节基因",
        # "请用我给你的4个数字，通过加、减、乘、除、括号，组成一个运算，使得结果为24。注意：数字需要全部使用我提供的数字：4 4 6 8。",
        "既然监狱里全是罪犯，警察为什么不去监狱里抓人？",
        "美国首都在哪里？",
        "翻译成英文：当时的他是最好的他，可是很久很久以后的我才是最好的我。最好的我们之间，隔了一整个青春，怎么奔跑也跨不过的青春，就只能挥出手道别。"
    ]
    n = 3
    for model in model_list:
        print(f"Testing: {model}...")
        speed = 0
        for user_query in user_query_list:
            for i in range(n):
                # print('\n' + 100*'*' + '\n')
                speed += main(model=model, user_query=user_query, stream=False)
        speed = round(speed / (n * len(user_query_list)), 2)
        print("model: {}, speed: {} token/s.".format(model, speed))
        with open(r"assests/record所有ollama模型测速.txt", "a") as f:
            f.write("model: {}, speed: {} token/s.\n".format(model, speed))