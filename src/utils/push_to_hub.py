import json
import argparse
from datasets import Dataset

### ---------------------------------- Arguments ---------------------------------- ###

def get_args():
    parser = argparse.ArgumentParser()
    # General Settings
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct-Turbo")
    parser.add_argument("--data_path", type=str, default="AnonymousLLMer/finance-corpus-krx")
    parser.add_argument("--data_type", type=str, default="hf", choices=["hf", "json", "recursive_load"])
    parser.add_argument("--prompt_type", type=str, default="bqa_blend_question_answers")
    parser.add_argument("--checkpoint_every", type=int, default=50, help="Save checkpoint every N steps")

    return parser.parse_args()

args = get_args()
print(f"### Check Arguments: {args}")

### ----------------------------------- Load Data ----------------------------------- ###

if args.data_type == "hf":
    dataset_save_name = args.data_path.split("/")[-1]
elif args.data_type == "json":
    dataset_save_name = args.data_path.split(".json")[0].split("/")[-1]
else:
    print('Error: Not implemented for this args.data_type.')
        
def load_previous_response_dataset(prompt_type, extract_column_name):
    load_dir = f"../data/{dataset_save_name}_{prompt_type}.json"
    with open(load_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)
    column_data = [datum[extract_column_name] for datum in data]
    return column_data

match args.prompt_type:
    case "mcqa":
        instruction = load_previous_response_dataset("mcqa", "instruction")
        match = load_previous_response_dataset("mcqa", "match")
        output = load_previous_response_dataset("mcqa", "output")
        texts = load_previous_response_dataset("mcqa", "text")

        result_dict = {
            'instruction': instruction,
            'match': match,
            'output': output,
            'text': texts
        }
    case "bqa":
        texts = load_previous_response_dataset("bqa_complex_question", "text")
        questions = load_previous_response_dataset("bqa_complex_question", "response")
        cq_answers = load_previous_response_dataset("bqa_cq_answer", "response")
        q_answers = load_previous_response_dataset("bqa_q_answer", "response")
        blend_answers = load_previous_response_dataset("bqa_blend_question_answers", "response")

        result_dict = {
            'text': texts, 
            'instruction': questions, 
            'output': blend_answers, 
            'cq_answer': cq_answers,
            'q_answer': q_answers,
        }


dataset = Dataset.from_dict(result_dict)
dataset.push_to_hub(f"nayohan/bqa-{dataset_save_name}-test", private=True)