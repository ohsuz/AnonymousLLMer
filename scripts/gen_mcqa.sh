# Huggingface Dataset
python ../src/gen_mcqa.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "mcqa" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/utils/postprocess.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "mcqa" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/utils/push_to_hub.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "mcqa"

# Local Json File
python ../src/gen_mcqa.py --data_type "json" --data_path "../data/finance-corpus-krx.json" --prompt_type "mcqa" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/utils/postprocess.py --data_type "json" --data_path "../data/finance-corpus-krx.json" --prompt_type "mcqa" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/utils/push_to_hub.py --data_type "json" --data_path "../data/finance-corpus-krx.json" --prompt_type "mcqa"
