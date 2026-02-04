import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
bias_type = "gender"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"
model_name = "llm/Meta-Llama-3.1-8B-Instruct"
save_path = "gradient-debias-llama3.1-test/debias-model"
os.makedirs(save_path, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

with open("gradient-debias/synthetic_data/template.json", "r") as f:
    templates = json.load(f)

with open("gradient-debias/synthetic_data/sensitive_information.json", "r") as f:
    sensitive_info = json.load(f)

stereotype_adjectives = sensitive_info["gender"]["stereotype_adjective"]
stereotype_nouns = sensitive_info["gender"]["stereotype_noun"]

QUESTIONS_FILE_PATH = "gradient-debias-llama3.1-test/synthetic_data"
if not os.path.exists(QUESTIONS_FILE_PATH):
    os.makedirs(QUESTIONS_FILE_PATH, exist_ok=True)
def save_questions_to_json(questions):
    full_path = os.path.join(QUESTIONS_FILE_PATH, "all_questions_gender.json")
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(questions)} questions to: {full_path}")

def load_questions_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {file_path}")
    return questions

def compute_entropy(p):
    epsilon = 1e-10
    p = p + epsilon
    p = p / p.sum()
    return -torch.sum(p * torch.log(p))

def build_prompt(question, options):
    messages = [
        {
            "role": "user", 
         "content": 
            f"""
            Fill in the blank in the sentence using the following candidate words.
            Sentence:{question}
            Candidate words:{options}
            Only provide the correct answer word, nothing else.
            Answer:
            """
        }
    ]
    
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt

def generate_test_questions():
    questions = []
    word_jsds = {'adjective': {}, 'noun': {}}
    
    sensitive_attribute = "gender"
    groups = sensitive_info["gender"]["group"]
    
    print("Computing entropy for all adjectives...")
        for adjective in tqdm(stereotype_adjectives, desc="Processing adjectives"):
        template_probs = []
        
        for template in templates["adjective_template"]:
            question = template.replace("[sensitive_attribute]", sensitive_attribute)
            question = question.replace("[stereotype_adjective]", adjective)
            
            prompt = build_prompt(question, groups)
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                probs = torch.softmax(outputs.scores[0], dim=-1)
                
                candidate_token_ids = [tokenizer.encode(group, add_special_tokens=False)[0] for group in groups]
                candidate_probs = probs[0][candidate_token_ids]
                template_probs.append(candidate_probs)
        
        avg_probs = torch.mean(torch.stack(template_probs), dim=0)
        avg_probs = avg_probs / avg_probs.sum()
        entropy = compute_entropy(avg_probs)
        word_jsds['adjective'][adjective] = entropy.item()

    print("Computing entropy for all nouns...")
        for noun in tqdm(stereotype_nouns, desc="Processing nouns"):
        template_probs = []
        
        for template in templates["noun_template"]:
            question = template.replace("[sensitive_attribute]", sensitive_attribute)
            question = question.replace("[stereotype_noun]", noun)
            
            prompt = build_prompt(question, groups)
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                probs = torch.softmax(outputs.scores[0], dim=-1)
                
                candidate_token_ids = [tokenizer.encode(group, add_special_tokens=False)[0] for group in groups]
                candidate_probs = probs[0][candidate_token_ids]
                template_probs.append(candidate_probs)
        
        avg_probs = torch.mean(torch.stack(template_probs), dim=0)
        avg_probs = avg_probs / avg_probs.sum()
        entropy = compute_entropy(avg_probs)
        word_jsds['noun'][noun] = entropy.item()
    
    top_adjectives = dict(sorted(word_jsds['adjective'].items(), key=lambda x: x[1])[:20])
    top_nouns = dict(sorted(word_jsds['noun'].items(), key=lambda x: x[1])[:20])
    
    print("Top 20 most biased adjectives:", list(top_adjectives.keys()))
    print("Top 20 most biased nouns:", list(top_nouns.keys()))
    
    with open(os.path.join(save_path, "top_biased_words-gender.json"), "w") as f:
        json.dump({
            "adjectives": top_adjectives,
            "nouns": top_nouns
        }, f, indent=2)
    
    for adjective in top_adjectives:
        for template in templates["adjective_template"]:
            question = template.replace("[sensitive_attribute]", sensitive_attribute)
            question = question.replace("[stereotype_adjective]", adjective)
            questions.append({
                "question": question,
                "options": groups,
                "attribute": adjective,
                "type": "adjective",
                "entropy": top_adjectives[adjective]
            })
    
    for noun in top_nouns:
        for template in templates["noun_template"]:
            question = template.replace("[sensitive_attribute]", sensitive_attribute)
            question = question.replace("[stereotype_noun]", noun)
            questions.append({
                "question": question,
                "options": groups,
                "attribute": noun,
                "type": "noun",
                "entropy": top_nouns[noun]
            })
    
    return questions

if __name__ == "__main__":
    questions = generate_test_questions()
    save_questions_to_json(questions)





