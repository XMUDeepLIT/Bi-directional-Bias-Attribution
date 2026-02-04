import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from faig import compute_all_groups_faig

with open("gradient-debias-llama3.1-test/debias-model/top_biased_words-religion.json", "r") as f:
    biased_words = json.load(f)
    adjective_options = list(biased_words["adjectives"].keys())
    noun_options = list(biased_words["nouns"].keys())

bias_type = "religion"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model_name = "llm/Meta-Llama-3.1-8B-Instruct"
bias_neurons_save_path = "gradient-debias-llama3.1-test/bias_neuron_index/anti"
os.makedirs(bias_neurons_save_path, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("gradient-debias/synthetic_data_anti/template.json", "r") as f:
    templates = json.load(f)

with open("gradient-debias/synthetic_data_anti/sensitive_information.json", "r") as f:
    sensitive_info = json.load(f)

QUESTIONS_FILE_PATH = "gradient-debias-llama3.1-test/synthetic_data_anti"
if not os.path.exists(QUESTIONS_FILE_PATH):
    os.makedirs(QUESTIONS_FILE_PATH)
QUESTIONS_FILE_PATH = os.path.join(QUESTIONS_FILE_PATH, f"all_questions_{bias_type}.json")

def save_questions_to_json(questions, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(questions)} questions to: {file_path}")

def load_questions_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {file_path}")
    return questions

def generate_test_questions():
    adjective_questions = []
    noun_questions = []
    
    sensitive_attribute = "religion"
    groups = sensitive_info["religion"]["group"]
    
    for template in templates["adjective_template"]:
        question_group = []
        for group in groups:
            question = template.replace("[sensitive_attribute]", sensitive_attribute)
            question = question.replace("[group]", group)
            
            question_group.append({
                "question": question,
                "type": "adjective"
            })
        adjective_questions.append(question_group)
    
    for template in templates["noun_template"]:
        question_group = []
        for group in groups:
            question = template.replace("[sensitive_attribute]", sensitive_attribute)
            question = question.replace("[group]", group)
            
            question_group.append({
                "question": question,
                "type": "noun"
            })
        noun_questions.append(question_group)
    
    return {
        "adjective_questions": adjective_questions,
        "noun_questions": noun_questions
    }

def build_faig_prompt(question):
    if any(word in question.lower() for word in ["adjective"]):
        options = adjective_options
    else:
        options = noun_options
        
    messages = [
        {
            "role": "user", 
            "content": 
                f"""
                According to the religion of the person, fill in the blank in the sentence using the following options.
                Sentence: {question}
                Options: {options}
                Only provide a single word from the options, nothing else.
                """
        }
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


test_questions = generate_test_questions()
save_questions_to_json(test_questions, QUESTIONS_FILE_PATH)

print(f"Total {len(test_questions['adjective_questions'])} adjective test questions")
print(f"Total {len(test_questions['noun_questions'])} noun test questions")

print("Computing FAIG neuron importance scores for all questions...")
all_faig_scores = []

for question_group in tqdm(test_questions['adjective_questions']):
    group_prompts = [build_faig_prompt(item["question"]) for item in question_group]
    group_faig_scores = compute_all_groups_faig(group_prompts, model, tokenizer, adjective_options)
    all_faig_scores.append(group_faig_scores)

for question_group in tqdm(test_questions['noun_questions']):
    group_prompts = [build_faig_prompt(item["question"]) for item in question_group]
    group_faig_scores = compute_all_groups_faig(group_prompts, model, tokenizer, noun_options)
    all_faig_scores.append(group_faig_scores)

print("Computing global FAIG neuron importance mean...")
stacked_faig = torch.stack(all_faig_scores)
mean_faig_scores = torch.mean(stacked_faig, dim=0)

faig_file = os.path.join(bias_neurons_save_path, f"faig_{bias_type}.json")
with open(faig_file, 'w') as f:
    json.dump({
        "model": model_name,
        "faig_score": mean_faig_scores.tolist()
    }, f, indent=2)
print(f"Saved FAIG scores to: {faig_file}")
print("faig_score", len(mean_faig_scores.tolist()))

top_faig_indices = torch.topk(mean_faig_scores, 10).indices
print("Top 10 FAIG neuron indices:", top_faig_indices.tolist())
print("Top 10 FAIG neuron values:", mean_faig_scores[top_faig_indices].tolist())


