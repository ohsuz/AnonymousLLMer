import os
import argparse
import json
import time

from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

from prompts.mcqa import *


with open("../api_keys/aimlapi.txt", 'r') as f :
    api_key = f.readline().strip()
api = OpenAI(api_key=api_key, base_url="https://api.aimlapi.com/v1")

### ---------------------------------- Arguments ---------------------------------- ###

def get_args():
    parser = argparse.ArgumentParser()
    # General Settings
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct-Turbo")
    parser.add_argument("--data_path", type=str, default="AnonymousLLMer/finance-corpus-krx")
    parser.add_argument("--data_type", type=str, default="hf", choices=["hf", "json"])
    parser.add_argument("--prompt_type", type=str, default="lecture")
    parser.add_argument("--checkpoint_every", type=int, default=50, help="Save checkpoint every N steps")
    # Generation Parameters
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=8192)
    
    return parser.parse_args()

args = get_args()

print(f"### Check Arguments: {args}")

### ----------------------------------- Load Data ----------------------------------- ###

if args.data_type == "hf":
    dataset_save_name = args.data_path.split("/")[-1]
    output_dir = f"../data/{dataset_save_name}_{args.prompt_type}.json"
    ds = load_dataset(args.data_path)
    texts = list(ds['train']['text'])

elif args.data_type == "json":
    dataset_save_name = args.data_path.split(".json")[0]
    output_dir = f"{dataset_save_name}_{args.prompt_type}.json"
    with open(args.data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = [datum['text'] for datum in data]

print(f"### Check Save Dir: {output_dir}")

### ----------------------------------- Generation ----------------------------------- ###

ranges = [i for i in range(0, len(texts), args.checkpoint_every)]
ranges.append(len(texts))

global_idx = 0
results = []
for i in range(len(ranges)-1):
    start_idx, end_idx = ranges[i], ranges[i+1]
    cur_texts = texts[start_idx:end_idx]
    instructions, outputs = [], []
    
    for text in tqdm(cur_texts):
        match args.prompt_type:
            case "mcqa":
                user_prompt = input_finance_mcqa(text)
                
        completion = api.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        instructions.append(user_prompt)
        outputs.append(completion.choices[0].message.content.strip())

        
    for output_idx, output in enumerate(outputs):
        result = {
            "id": global_idx,
            "text": cur_texts[output_idx],
            "instruction": instructions[output_idx].strip(),
            "response": output,
            "created": int(time.time()),
            "gen_input_configs": {
                "type": args.prompt_type,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "input_generator": args.model_name,
            }
        }
        global_idx += 1
        results.append(result)
        
    with open(output_dir, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved. Total prompts: {len(results)}")


with open(output_dir, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Responses generated from {args.model_name}. Total prompts: {len(results)}")