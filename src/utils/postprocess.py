import re
import json
import argparse
from datasets import load_dataset

def post_process_cqa(data, prompt_type):
    """ Extract ‘category’, ‘question’, ‘answer’ from the data inferred by the LLM model and append them to the List
    Args:
        data (pd.Dataframe): load from model inference result json
    Returns:
        output_list: overwrite model inference result into json file
    """
    for idx, output in enumerate(data):  # parsing qa
        if prompt_type=='mcqa':
            json_string_list = output['response'].split('\n\n') # 여러 출력
            for json_string in json_string_list:
                try:
                    category = re.search(r'"category":\s*"([^"]+)"', json_string)
                    question = re.search(r'"question":\s*"([^"]+)"', json_string)
                    answer = re.search(r'"answer":\s*"([^"]+)"', json_string)
                    matches = re.findall(r"정답은\s([A-Z])", answer.group(1))
                    data[idx]['category'] = category.group(1)
                    data[idx]['instruction'] = question.group(1)
                    data[idx]['output'] = answer.group(1)
                    data[idx]['match'] = matches[0]
                    break
                except:
                    data[idx]['category'] = ""
                    data[idx]['instruction'] = ""
                    data[idx]['output'] = ""
                    data[idx]['match'] = ""
        
        elif prompt_type=="bqa_complex_question":
            print(output)
            question = output['response'].split('[질문 4]\nQ. ')[-1]
            data[idx]['response'] = question
        else:
            print('Please implement the prompt_type extraction code.')
    return data

if __name__=="__main__":
    ### ----------------------------------- Load Data ----------------------------------- ###
    def get_args():
        parser = argparse.ArgumentParser()
        # General Settings
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct-Turbo")
        parser.add_argument("--data_path", type=str, default="AnonymousLLMer/finance-corpus-krx")
        parser.add_argument("--data_type", type=str, default="previous_load", choices=["hf", "json", "previous_load"])
        parser.add_argument("--prompt_type", type=str, default="lecture")
        parser.add_argument("--checkpoint_every", type=int, default=50, help="Save checkpoint every N steps")

        return parser.parse_args()

    args = get_args()

    if args.data_type == "hf":
        dataset_save_name = args.data_path.split("/")[-1]
        output_dir = f"../data/{dataset_save_name}_{args.prompt_type}.json"
        with open(output_dir, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif args.data_type == "json":
        dataset_save_name = args.data_path.split(".json")[0]
        output_dir = f"{dataset_save_name}_{args.prompt_type}.json"
        with open(output_dir, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        print('Error: Not implemented for this args.data_type.')

        
    ### ----------------------------------- Postprocess ----------------------------------- ###
    processed_data = post_process_cqa(data, args.prompt_type)
        
    with open(output_dir, "w") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"Processed dataset saved. {output_dir}")