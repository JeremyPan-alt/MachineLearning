from openai import OpenAI

client = OpenAI(api_key="sk-ed1378c13b8f4a6ea6401c04b56362af", base_url="https://api.deepseek.com")

messages = [{"role": "system", "content": "You are a helpful assistant"}]

while True:
    # 获取用户输入
    user_input = input("你: ")
    if user_input.lower() == '退出':
        break

    # 将用户输入添加到消息列表
    messages.append({"role": "user", "content": user_input})

    # 以流式方式创建聊天完成
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True
    )

    # 处理流式响应
    full_response = ""
    print("助手: ", end="", flush=True)
    for chunk in response:
        # 检查是否有内容
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content') and delta.content:
            chunk_content = delta.content
            full_response += chunk_content
            print(chunk_content, end="", flush=True)
    print()

    # 将助手的回复添加到消息列表
    messages.append({"role": "assistant", "content": full_response})
