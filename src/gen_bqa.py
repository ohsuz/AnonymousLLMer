import os
import argparse
import json
import time

from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
from huggingface_hub import InferenceClient

from prompts.bqa import *


with open("../api_keys/aimlapi.txt", 'r') as f :
    api_key = f.readline().strip()

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

### ----------------------------------- Generation ----------------------------------- ###

def load_previous_response(prompt_type, extract_column_name, start_idx, end_idx):
    """  Reloading only one column of previous inference results 

    Args:
        prompt_type (str): when load previous inference results, suffix file_name
        extract_column_name (str): in file, extract only one columns

    Returns:
        list: data list (one columns)
    """
    dataset_save_name = args.data_path.split("/")[-1]
    load_dir = f"../data/{dataset_save_name}_{prompt_type}.json"
    with open(load_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)
    column_data = [datum[extract_column_name] for datum in data]
    cur_column_data = column_data[start_idx:end_idx]
    return cur_column_data

ranges = [i for i in range(0, len(texts), args.checkpoint_every)]
ranges.append(len(texts))

global_idx = 0
results = []
for i in range(len(ranges)-1):
    start_idx, end_idx = ranges[i], ranges[i+1]
    cur_texts = texts[start_idx:end_idx]
    instructions, outputs = [], []
    
    match args.prompt_type:
        case "bqa_cq_answer": # reuse bqa_complex_question response
            questions = load_previous_response("bqa_complex_question", "response", start_idx, end_idx)
        case "bqa_q_answer": # reuse bqa_complex_question response
            questions = load_previous_response("bqa_complex_question", "response", start_idx, end_idx)
        case "bqa_blend_question_answers": # reuse above 3 response
            questions = load_previous_response("bqa_complex_question", "response", start_idx, end_idx)
            cq_answers = load_previous_response("bqa_cq_answer", "response", start_idx, end_idx)
            q_answers = load_previous_response("bqa_q_answer", "response", start_idx, end_idx)

    print(len(questions ))
    print(len(texts))
    for idx in tqdm(range(args.checkpoint_every)):
        match args.prompt_type:
            case "bqa_complex_question":
                user_prompt = input_finance_bqa_complex_question(texts[idx])
            case "bqa_cq_answer": # reuse bqa_complex_question response
                user_prompt = input_finance_bqa_cq_answer(questions[idx], texts[idx])
            case "bqa_q_answer": # reuse bqa_complex_question response
                user_prompt = input_finance_bqa_q_answer(questions[idx])
            case "bqa_blend_question_answers": # reuse above 3 response
                user_prompt = input_finance_bqa_blend_question_answers(questions[idx], cq_answers[idx], q_answers[idx])
                
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