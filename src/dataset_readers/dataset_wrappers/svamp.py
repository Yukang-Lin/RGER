from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import re

field_getter = App()

def add_numbering(text):
    text = re.sub(r'\n{2,}', r'\n', text)
    sentences = text.split('\n')  
    numbered_text = ''

    sentences = sentences[1:]

    for i, sentence in enumerate(sentences, 1):
        numbered_text += f'{i}. {sentence}\n'  

    return numbered_text

def remove_brackets_content(text):
    pattern = r"<<.*?>>"  
    result = re.sub(pattern, "", text)  
    return result

@field_getter.add("q")
def get_q(entry):
    return entry['Body'] + ' ' + entry['Question']

@field_getter.add("a")
def get_a(entry):
    # return entry['complex_cot']
    return entry['Rationale1']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("dq_q")
def get_dq_q(entry):
    return entry['question']

@field_getter.add("dq_a")
def get_dq_a(entry):
    return entry['answer']

@field_getter.add("dq_qa")
def get_dq_qa(entry):
    return f"{get_dq_q(entry)}\t{get_dq_a(entry)}"

@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question}\nA: Let's think step by step\n".format(ice_prompt="{ice_prompt}", question=get_q(entry))


@field_getter.add("dq_gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question}\nA: Let's think step by step\n".format(ice_prompt="{ice_prompt}", question=get_dq_q(entry))

@field_getter.add("complex_qa")
def get_complex_qa(entry):
    ans = get_a(entry)
    ans = remove_brackets_content(ans)
    pattern = r"####\s*(-?\d+)"
    replacement = r"The answer is \1"
    new_ans = re.sub(pattern, replacement, ans)
    return "{question}\n{answer}".format(question = get_q(entry), answer = new_ans)
    
@field_getter.add("index_qa")
def get_index_qa(entry):
    ques = get_q(entry)
    ans = get_a(entry)
    return "Question: {question}\nA: Let's think step by step\n{answer}".format(question=ques, answer = ans)

class DatasetWrapper(ABC):
    name = "svamp"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "answer"
    hf_dataset = "index_data/svamp/svamp"  # You can also use the dataset downloaded in the file "index_data"
    hf_dataset_name = "main"
    field_getter = field_getter
