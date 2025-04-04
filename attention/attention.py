from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def aggregate_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:],
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)

def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M') # base model - for instruct model, use proper chat format
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M')

def decode(tokens):
    '''Turn tokens into text with mapping index'''
    full_text = ''
    chunks = []
    for i, token in enumerate(tokens):
        text = tokenizer.decode(token)
        full_text += text
        chunks.append(text)
    return full_text, chunks

def get_prompt_attention(prompt, start_layer=0, end_layer=None):
    '''Process only the prompt tokens and their attention patterns without generating completions'''
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    
    # Run the model with just the prompt - no generation, with no gradient tracking
    with torch.no_grad():
        outputs = model(
            tokens,
            output_attentions=True,
            return_dict=True
        )
    
    # Get attention matrices from the model output
    attention = outputs.attentions
    if end_layer is None:
        end_layer = model.config.num_hidden_layers - 1
    
    # Process attention for visualization 
    attn_matrices = []
    
    # Process token attention across selected layers
    layer_attns = []
    for layer_idx in range(start_layer, end_layer + 1):
        if layer_idx >= len(attention):
            break  # Prevent index out of range
        layer_attn = attention[layer_idx]
        # Average over heads
        layer_avg = layer_attn.squeeze(0).mean(dim=0)
        layer_attns.append(layer_avg)
    
    # Average over all layers
    if layer_attns:
        combined_attn = torch.stack(layer_attns).mean(dim=0)
        
        # For each token, add its attention pattern to our matrix
        for i in range(len(tokens[0])):
            attn_matrices.append(combined_attn[i])
    
    # Create attention matrix from all the attention vectors
    attn_m = heterogenous_stack(attn_matrices)
    
    # Get token text for display
    decoded, tokenized = decode(tokens[0])
    return decoded, tokenized, attn_m, model.config.num_hidden_layers

def get_completion(prompt):
    '''Get full text, token mapping, and attention matrix for a completion'''
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        tokens,
        max_new_tokens=50,
        output_attentions=True,
        return_dict_in_generate=True,
        early_stopping=True,
        length_penalty=-1
    )
    sequences = outputs.sequences
    attn_m = heterogenous_stack([
        torch.tensor([
            1 if i == j else 0
            for j, token in enumerate(tokens[0])
        ])
        for i, token in enumerate(tokens[0])
    ] + list(map(aggregate_attention, outputs.attentions)))
    decoded, tokenized = decode(sequences[0])
    return decoded, tokenized, attn_m

def show_matrix(xs):
    for x in xs:
        line = ''
        for y in x:
            line += '{:.4f}\t'.format(float(y))
        print(line)
