from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import re

field_getter = App()

def add_numbering(text):
    sentences = text.split('\n')
    numbered_text = ''

    for i, sentence in enumerate(sentences, 1):
        numbered_text += f'{i}. {sentence}\n'

    return numbered_text

@field_getter.add("q")
def get_q(entry):
    return entry['question']


@field_getter.add("a")
def get_a(entry):
    return entry['answer']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question}\nA: Let's think step by step\n".format(ice_prompt="{ice_prompt}", question=get_q(entry))


@field_getter.add("dq_q")
def get_dq_q(entry):
    return entry['question']


@field_getter.add("dq_a")
def get_dq_a(entry):
    return entry['answer']


@field_getter.add("dq_qa")
def get_qa(entry):
    return f"{get_dq_q(entry)}\t{get_dq_a(entry)}"


@field_getter.add("dq_gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question}\nA: Let's think step by step\n".format(ice_prompt="{ice_prompt}", question=get_dq_q(entry))


@field_getter.add("complex_qa")
def get_complex_qa(entry):
    ans = get_a(entry)
    #ans = remove_brackets_content(ans)
    pattern = r"####\s*(-?\d+)"
    replacement = r"The answer is \1"
    new_ans = re.sub(pattern, replacement, ans)
    return "Question: {question}\nA: Let's think step by step\n{answer}".format(question = get_q(entry), answer = new_ans)
    #return "Question:{question}\n Answer:{answer}".format(question = get_q(entry), answer = get_a(entry))
    
@field_getter.add("index_qa")
def get_index_qa(entry):
    ans = get_a(entry)
    pattern = r"####\s*(-?\d+)"
    replacement = r"The answer is \1."
    new_ans = re.sub(pattern, replacement, ans)
    new_ans = re.sub('\n',' ',new_ans)
    return "Question: {question}\nA:Let's think step by step\n{answer}".format(question = get_q(entry), answer = new_ans)

class DatasetWrapper(ABC):
    name = "gsm8k"
    ice_separator = "\n"
    # a_prefix = "\nA:Let's think step by step\n"
    a_prefix = ""
    question_field = "question"
    answer_field = "answer"
    hf_dataset = "gsm8k"
    hf_dataset_name = "main"
    field_getter = field_getter
