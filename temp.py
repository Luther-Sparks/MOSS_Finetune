tmp = [{"instruction":"ins1", "input":"inp1","output":"out1"}]

train_dataset = [
    {
        "input": f"""<s>下面是描述任务的指令，并与提供进一步符合上下文的输入。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 输入：
{sample["input"]}

### 响应：
""" if len(sample["input"].strip()) != 0 else f"""<s>下面是描述任务的指令。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 响应：
""",
        "output": f"{sample['output']}</s>"
    } for sample in tmp
]

print(train_dataset)

[
    {
        'input': '<s>下面是描述任务的指令，并与提供进一步符合上下文的输入。请编写适当完成请求的响应。\n\n### 指令：\nins1\n\n### 输入：\ninp1\n\n### 响应：\n', 
        'output': 'out1</s>'
    }
]