import torch

def calculate_final_faig(prompt, model, tokenizer, candidate_words, n_steps=20, split_point=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    linear_layer = model.lm_head

    with torch.set_grad_enabled(True):
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        gen_hs = gen_outputs.hidden_states

        new_token_hs = [step_hidden[-1][:, -1, :] for step_hidden in gen_hs]
        generated_hidden_states = torch.stack(new_token_hs, dim=1)
        generated_hidden_states = generated_hidden_states.detach().requires_grad_(True)
        
        generated_tokens = gen_outputs.sequences[0][inputs["input_ids"].shape[1]:]
        original_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        filled_word = original_response.strip()
        
        print(f"Generated word: {filled_word}")

        candidate_ids_list = []
        max_token_length = 0
        for word in candidate_words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            candidate_ids_list.append(ids)
            max_token_length = max(max_token_length, len(ids))
            if len(ids) > 1:
                print(f"Warning: word '{word}' contains multiple tokens: {ids}")
        
        all_token_ids = []
        for ids in candidate_ids_list:
            all_token_ids.extend(ids)
        all_token_ids = torch.tensor(all_token_ids, device=generated_hidden_states.device)
        
        scaled_acts = generated_hidden_states.detach().clone()
        scaled_acts.requires_grad_(True)
        logits = linear_layer(scaled_acts)
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        
        def compute_multi_token_entropy(scaled_acts):
            logits = linear_layer(scaled_acts)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            
            candidate_word_probs = []
            offset = 0
            for ids in candidate_ids_list:
                word_prob = probs[:, ids[0]].clone()
                
                for token_id in ids[1:]:
                    word_prob = word_prob * probs[:, token_id]
                
                candidate_word_probs.append(word_prob)
            
            candidate_word_probs = torch.stack(candidate_word_probs, dim=1)
            sum_p = candidate_word_probs.sum(dim=-1, keepdim=True)
            norm_p = candidate_word_probs / (sum_p + 1e-12)
            
            ent = -(norm_p * (norm_p + 1e-12).log()).sum(dim=-1)
            return 1 / (ent + 1e-12)
            
        faig_scores = compute_faig_with_custom_function(
            generated_hidden_states, 
            linear_layer,
            compute_multi_token_entropy,
            n_steps=n_steps
        )

    return faig_scores, generated_hidden_states, original_response, linear_layer, filled_word

def compute_faig_with_custom_function(hidden_states, linear_layer, entropy_function, n_steps=20):
    B, T, D = hidden_states.shape
    assert B == 1, "FAIG only supports batch_size=1"

    orig_act = hidden_states[0].detach()
    dtype = next(linear_layer.parameters()).dtype
    orig_act = orig_act.to(dtype)

    alphas = torch.linspace(0, 1, steps=n_steps, device=hidden_states.device, dtype=dtype).view(n_steps, 1, 1)
    scaled = orig_act.unsqueeze(0) * alphas
    scaled.requires_grad_(True)

    ent_inv = entropy_function(scaled)
    
    grads = torch.autograd.grad(ent_inv.sum(), scaled)[0]
    avg_grads = grads.mean(dim=0)

    faig = (orig_act * avg_grads).mean(dim=0)
    return faig





