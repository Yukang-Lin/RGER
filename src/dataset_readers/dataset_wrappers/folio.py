from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import re

field_getter = App()

@field_getter.add("q")
def get_q(entry):
    return entry['question']


@field_getter.add("a")
def get_a(entry):
    return entry['rationale']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question}\nA: ".format(ice_prompt="{ice_prompt}", question=get_q(entry))


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
    return "{ice_prompt}{question}\nA: ".format(ice_prompt="{ice_prompt}", question=get_dq_q(entry))


@field_getter.add("index_qa")
def get_index_qa(entry):
    return "Question: {question}\nA: {answer}".format(question = get_q(entry), answer = get_a(entry))

class DatasetWrapper(ABC):
    name = "folio"
    ice_separator = "\n"
    a_prefix = ""
    question_field = "question"
    answer_field = "rationale"
    hf_dataset = "folio"
    hf_dataset_name = "main"
    field_getter = field_getter
