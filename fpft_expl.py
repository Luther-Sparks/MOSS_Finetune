import sys
sys.path.append("../..")
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, GenerationConfig
from collie import Trainer, EvaluatorForPerplexity, CollieConfig, PPLMetric, \
    DecodeMetric, CollieDatasetForTraining, CollieDatasetForGeneration, \
    LossMonitor, TGSMonitor, MemoryMonitor, EvalMonitor, GradioProvider, \
        EvaluatorForGeneration, LRMonitor, BleuMetric, LlamaForCausalLM

config = CollieConfig.from_pretrained("/remote-home/share/MOSS_7B_Base")
config.dp_size = 4
config.train_micro_batch_size = 4
config.eval_batch_size = 10
config.eval_per_n_epochs = 1
config.eval_per_n_steps = 20
config.train_epochs = 10
config.ds_config = {
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        }
    },
    "zero_force_ds_cpu_optimizer": False
}
config.seed = 1024
# Prepare training dataset
train_dataset = [
    {
        "input": f"""<s>下面是描述任务的指令，并与提供进一步符合上下文的输入。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 输入：
{sample["input"]}

### 响应：
""" if not sample["input"] else f"""<s>下面是描述任务的指令。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 响应：
""",
        "output": f"{sample['output']}</s>"
    } for sample in load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0", split="train[:-500]")
]
# Prepare perplexity evaluation dataset
ratio = 0.1
eval_dataset_ppl, train_dataset = train_dataset[:int(
    len(train_dataset) * ratio)], train_dataset[int(len(train_dataset) * ratio):]
# Prepare generation based evaluation dataset
eval_dataset_bleu = [
    {
        "text": f"""<s>下面是描述任务的指令，并与提供进一步符合上下文的输入。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 输入：
{sample["input"]}

### 响应：
""" if not sample["input"] else f"""<s>下面是描述任务的指令。请编写适当完成请求的响应。

### 指令：
{sample["instruction"]}

### 响应：
""",
        "target": " ".join(f"{sample['output']}</s>".split())
    } for sample in load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0", split="train[-500:]")
]
# Prepare model
model = LlamaForCausalLM.from_pretrained(
    "/remote-home/share/MOSS_7B_Base", config=config)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
tokenizer = LlamaTokenizer.from_pretrained(
    "/remote-home/share/MOSS_7B_Base")
# Convert to CoLLie Dataset
train_dataset = CollieDatasetForTraining(train_dataset,
                                          tokenizer=tokenizer)
eval_dataset_ppl = CollieDatasetForTraining(eval_dataset_ppl,
                                            tokenizer=tokenizer)
eval_dataset_bleu = CollieDatasetForGeneration(eval_dataset_bleu,
                                               tokenizer=tokenizer)
# Prepare Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model=model,
    config=config,
    dataset=eval_dataset_ppl,
    monitors=[
        EvalMonitor(config)
    ],
    metrics={
        "ppl": PPLMetric(gather_result=True)
    },
)
evaluator_bleu = EvaluatorForGeneration(
    model=model,
    config=config,
    dataset=eval_dataset_bleu,
    monitors=[
        EvalMonitor(config)
    ],
    metrics={
        "bleu": BleuMetric(gather_result=True, ngram=1),
        "decode": DecodeMetric()
    },
    generation_config=GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=100,
    )
)
# Prepare Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    config=config,
    optimizer=optimizer,
    train_dataset=train_dataset,
    monitors=[
        LossMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LRMonitor(config)
    ],
    data_provider=GradioProvider(tokenizer, port=12888, stream=True,
                                 generation_config=GenerationConfig(
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     max_new_tokens=100,
                                 )),
    evaluators=[evaluator_ppl, evaluator_bleu]
)
# Command: torchrun --standalone --nproc_per_node=4 sft.py
trainer.train()
# trainer.save_checkpoint(path="/mnt/petrelfs/zhangshuo/model/test_save_checkpoint", mode="model")