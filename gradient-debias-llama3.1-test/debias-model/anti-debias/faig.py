import torch


def compute_group_faig(hidden_states_list, linear_layer, option_token_ids, options, n_steps=20):
    n_questions = len(hidden_states_list)
    assert n_questions > 1
    
    orig_acts = [hs[0].detach() for hs in hidden_states_list]
    dtype = next(linear_layer.parameters()).dtype
    device = next(linear_layer.parameters()).device
    orig_acts = [act.to(dtype).to(device) for act in orig_acts]
    
    alphas = torch.linspace(0, 1, steps=n_steps, device=device, dtype=dtype).view(n_steps, 1, 1)
    
    def compute_probability_differences(scaled_acts_list):
        device = scaled_acts_list[0].device
        n_questions = len(scaled_acts_list)
        dtype = scaled_acts_list[0].dtype
        
        total_js_div = torch.zeros(n_steps, device=device, dtype=dtype)
        
        for step in range(n_steps):
            all_probs = []
            for i in range(n_questions):
                logits_i = linear_layer(scaled_acts_list[i])
                last_logits_i = logits_i[step, -1, :]
                temperature = 0.1
                probs_i = torch.softmax(last_logits_i / temperature, dim=-1)
                
                prob_vector = torch.zeros(len(options), device=device, dtype=dtype)
                for j, token_id in enumerate(option_token_ids):
                    if isinstance(token_id, (list, tuple)):
                        word_prob = probs_i[token_id[0]].clone()
                        for next_token_id in token_id[1:]:
                            word_prob = word_prob * probs_i[next_token_id]
                        prob_vector[j] = word_prob
                    else:
                        prob_vector[j] = probs_i[token_id]
                
                prob_vector = prob_vector / (prob_vector.sum() + 1e-12)
                all_probs.append(prob_vector)
            
            all_probs = torch.stack(all_probs)
            
            mean_dist = torch.mean(all_probs, dim=0)
            
            amplification_factor = 1
            for i in range(n_questions):
                epsilon = 1e-10
                p = all_probs[i] + epsilon
                q = mean_dist + epsilon
                
                kl_div = torch.sum(p * (torch.log(p) - torch.log(q))) * amplification_factor
                
                total_js_div[step] += kl_div / n_questions
        
        return total_js_div
    
    scaled_acts_list = [act.unsqueeze(0) * alphas for act in orig_acts]
    for scaled_acts in scaled_acts_list:
        scaled_acts.requires_grad_(True)
        assert scaled_acts.requires_grad
    
    try:
        with torch.enable_grad():
            total_diff = compute_probability_differences(scaled_acts_list)
            
            assert total_diff.requires_grad
            
            grads_list = torch.autograd.grad(total_diff.sum(), scaled_acts_list)
            
    except RuntimeError as e:
        print("Error computing gradients:", str(e))
        return torch.zeros_like(orig_acts[0][0])
    
    avg_grads = [grad.mean(dim=0) for grad in grads_list]
    
    faig = torch.zeros_like(orig_acts[0][0])
    for orig_act, avg_grad in zip(orig_acts, avg_grads):
        faig += (orig_act * avg_grad).mean(dim=0)
    faig = faig / n_questions
    
    return faig


def compute_all_groups_faig(prompts, model, tokenizer, options, n_steps=20):
    hidden_states_list = []
    generated_tokens_list = []
    
    device = next(model.parameters()).device
    
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
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
            generated_hidden_states = torch.stack(new_token_hs, dim=1).to(device)
            
            hidden_states_list.append(generated_hidden_states)
            generated_tokens_list.append(option_token_ids)
    
    faig = compute_group_faig(
        hidden_states_list,
        model.lm_head,
        option_token_ids,
        options,
        n_steps=n_steps
    )
    
    return faig







