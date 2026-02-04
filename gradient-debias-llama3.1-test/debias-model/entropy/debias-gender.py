import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from faig import  calculate_final_faig

bias_type = "gender"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
model_name = "llm/Meta-Llama-3.1-8B-Instruct"
bias_neurons_save_path = "gradient-debias-llama3.1-test/bias_neuron_index/entropy"
os.makedirs(bias_neurons_save_path, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("gradient-debias/synthetic_data/template.json", "r") as f:
    templates = json.load(f)

with open("gradient-debias/synthetic_data/sensitive_information.json", "r") as f:
    sensitive_info = json.load(f)

QUESTIONS_FILE_PATH = "gradient-debias-llama3.1-test/synthetic_data/all_questions_gender.json"

def load_questions_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {file_path}")
    return questions

def build_prompt(question, options):
    messages_faig = [
        {"role": "system", "content": """
         Based on the context, fill in the blank using the candidate options. 
         Only answer with ONE word from the options without any other words.
         """},
         {
            "role": "user", 
         "content": 
            f"""
            Context:{question}
            Options:{options}
            """
        }
    ]
    messages_seig = [
        {
            "role": "user", 
         "content": 
            f"""
            Fill in the blank in the sentence using the following candidate words.
            Sentence:{question}
            Candidate words:{options}
            Answer:
            """
        }
    ]
    
    prompt_faig = tokenizer.apply_chat_template(
        messages_faig,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_seig = tokenizer.apply_chat_template(
        messages_seig,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt_faig, prompt_seig

def apply_mask_and_generate(generated_hidden_states, mask, linear_layer, tokenizer, inputs_length):
    modified_hidden_states = generated_hidden_states.clone()
    
    expanded_mask = mask.unsqueeze(0).unsqueeze(0).expand_as(modified_hidden_states)
    modified_hidden_states = torch.where(
        expanded_mask, 
        torch.zeros_like(modified_hidden_states), 
        modified_hidden_states
    )

    with torch.no_grad():
        new_logits = linear_layer(modified_hidden_states)
        new_token_ids = torch.argmax(new_logits, dim=-1)
        
        new_response = tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
        
        return new_response

test_questions = load_questions_from_json(QUESTIONS_FILE_PATH)

print(f"Total {len(test_questions)} test questions")

print("Calculating FAIG neuron importance scores for all questions...")
all_faig_scores = []
hidden_states_list = []
original_responses = []
inputs_lengths = [] 
linear_layer = None

for idx, item in enumerate(tqdm(test_questions)):
    print(f"\nQuestion {idx+1}/{len(test_questions)} - FAIG calculation phase")
    print(f"Question: {item['question']}")
    
    prompt, _ = build_prompt(item["question"], item["options"])
    
    faig_scores, hidden_states, original_response, layer, filled_word = calculate_final_faig(
        prompt, model, tokenizer, item["options"]
    )
    
    all_faig_scores.append(faig_scores)
    hidden_states_list.append(hidden_states)
    original_responses.append(original_response)
    inputs_lengths.append(hidden_states.shape[1])
    
    if linear_layer is None:
        linear_layer = layer
        
    if filled_word:
        print(f"Final answer: {filled_word}")

print("Calculating global FAIG neuron importance mean...")
stacked_faig = torch.stack(all_faig_scores)
mean_faig_scores = torch.mean(stacked_faig, dim=0)

faig_file = os.path.join(bias_neurons_save_path, f"faig_{bias_type}.json")
with open(faig_file, 'w') as f:
    json.dump({
        "model": model_name,
        "faig_score": mean_faig_scores.tolist()
    }, f, indent=2)
print(f"Saved FAIG scores to: {faig_file}")
print("faig_score",len(mean_faig_scores.tolist()))

top_faig_indices = torch.topk(mean_faig_scores, 10).indices
print("Top 10 FAIG neuron indices:", top_faig_indices.tolist())
print("Top 10 FAIG neuron values:", mean_faig_scores[top_faig_indices].tolist())
