import torch
import numpy as np
from transformers import AutoModelForCausalLM


def no_init(loading_code):
    '''
    no_init_weights is used in from_pretrained to speed up loading large models.
    However, torch-built-in modules like torch.nn.Linear are heavily used in models of transformers,
    while its weights initialization cannot be disabled by no_init_weights.
    '''
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


def get_model(**kwargs):
    return no_init(lambda: AutoModelForCausalLM.from_pretrained(**kwargs))


def ppl_generate(input_texts, model, tokenizer, choices_list, device=None):
    loss_list = []
    # here we set batch size to 1
    for choices, text in zip(choices_list, input_texts):
        filled_texts = [text+choice for choice in choices]
        loss = [_evaluate_loss(filled_text, model, tokenizer, device)[0] for filled_text in filled_texts]
        # inference one sentence per time
        loss_list.append(loss)
    lm_loss_list = np.array(loss_list)
    preds = lm_loss_list.argmin(axis=1).tolist()
    # print(preds)
    return preds


def _evaluate_loss(input_texts, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        # note here we assume padding is performed on the right, left padding token will affect position_id in gpt2
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())
        ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))
        lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(-1).cpu().numpy()
    return ce_loss / lens