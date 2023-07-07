import os
import json
import torch
import argparse
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/remote-home/share/MOSS_7B_Base", help="Path or name to the model. (e.g. `openlmlab/further-trained-llama-7b-patch`)")
parser.add_argument("--devices", type=str, default="0", help="Devices to use. (e.g. `0,1`)")
parser.add_argument("--max_length", type=int, default=512, help="Max length to generate. (e.g. 2048)")
parser.add_argument("--do_sample", type=bool, default=True, help="Whether to enable sampling in generation. (e.g. True)")
parser.add_argument("--top_k", type=int, default=40, help="top_k in generation. (e.g. 40)")
parser.add_argument("--top_p", type=float, default=0.8, help="top_p in generation. (e.g. 0.8)")
parser.add_argument("--temperature", type=float, default=0.7, help="temperature in generation. (e.g. 0.7)")
parser.add_argument("--penalty", type=float, default=1.02, help="repetition_penalty in generation. (e.g. 1.02)")
args = parser.parse_known_args()[0]

def cli(model: str = args.model,
        devices: str = args.devices,
        max_length: int = args.max_length,
        do_sample: bool = args.do_sample,
        top_k: int = args.top_k,
        top_p: float = args.top_p,
        temperature: float = args.temperature,
        penalty: float = args.penalty):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    if not os.path.exists(model):
        model = snapshot_download(model)
    # if os.path.exists(os.path.join(model, "patch_history.json")):
    #     patch_history = json.load(open(os.path.join(model, "patch_history.json"), mode="r", encoding="utf-8"))
    #     if False in patch_history.values():
    #         from tools.patch_model import patch_model
    #         patch_model(patch_model=model)
    tokenizer = LlamaTokenizer.from_pretrained(model)
    config = LlamaConfig.from_pretrained(model)
    with init_empty_weights():
        raw_model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16)
    raw_model.tie_weights()
    model = load_checkpoint_and_dispatch(
        raw_model, model, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"], dtype=torch.float16
    )
    print("请输入提示, 输入 exit 退出")
    while True:
        prompt = input("Prompt: ")
        if prompt.strip() == "exit":
            break
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs.input_ids[:, 0] = 1
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.cuda(), 
                attention_mask=inputs.attention_mask.cuda(), 
                max_length=max_length, 
                do_sample=do_sample, 
                top_k=top_k, 
                top_p=top_p, 
                temperature=temperature,
                repetition_penalty=penalty,
                num_return_sequences=1,
                eos_token_id=tokenizer.sp_model.eos_id(),
                bos_token_id=tokenizer.sp_model.bos_id())
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(prompt + response)
    
if __name__ == "__main__":
    cli()