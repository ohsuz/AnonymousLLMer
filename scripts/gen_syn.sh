# Huggingface Dataset
python ../src/gen_syn.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "finance_wikipedia" --checkpoint_every 10

# Local Json File
python ../src/gen_syn.py --data_type "json" --data_path "../data/finance-corpus-krx.json" --prompt_type "finance_wikipedia" --checkpoint_every 10