# Huggingface Dataset
python ../src/gen_bqa.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "bqa_complex_question" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/utils/postprocess.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "bqa_complex_question"  --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct

python ../src/gen_bqa.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "bqa_cq_answer" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/gen_bqa.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "bqa_q_answer" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/gen_bqa.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx" --prompt_type "bqa_blend_question_answers" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct

python ../src/utils/push_to_hub.py --data_type "hf" --data_path "AnonymousLLMer/finance-corpus-krx"



# Local Json File
python ../src/gen_bqa.py --data_type "json" --data_path "../data/finance-corpus-krx.json" --prompt_type "bqa_complex_question" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/utils/postprocess.py --data_type "json" --data_path "../data/finance-corpus-krx_bqa_complex_question.json" --prompt_type "bqa_complex_question" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct

python ../src/gen_bqa.py --data_type "json" --data_path "../data/finance-corpus-krx.json" --prompt_type "bqa_cq_answer" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/gen_bqa.py --data_type "json" --data_path "../data/finance-corpus-krx.json" --prompt_type "bqa_q_answer" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct
python ../src/gen_bqa.py --data_type "json" --data_path "../data/finance-corpus-krx.json" --prompt_type "bqa_blend_question_answers" --checkpoint_every 10 --model_name Qwen/Qwen2.5-72B-Instruct

python ../src/utils/push_to_hub.py --data_type "json" --data_path "../data/finance-corpus-krx.json"