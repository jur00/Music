from unidecode import unidecode
import re

from contextlib import contextmanager
import os
import stat

def jaccard_similarity(test, real):
    intersection = set(test).intersection(set(real))
    union = set(test).union(set(real))
    return len(intersection) / len(union)

def neutralize(string):
    return unidecode(re.sub(r'[^a-zA-Z 0-9À-ú]+', '', string)).lower()

def create_name_string(rb_data, i, name_parts):
    return neutralize(' '.join([v for k, v in rb_data[i].items() if k in name_parts]).strip())

def check_original(rb_data, i):
    return rb_data[i]['Album'] in ['', 'original mix']

def add_or_remove_original(comp_name):
    return re.sub('original mix', '', comp_name) if 'original mix' in comp_name else f'{comp_name} original mix'

@contextmanager
def set_dir(path):
    origin = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def on_rm_error(func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod(path, stat.S_IWRITE)
    os.unlink(path)

def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return ValueError('File Name does not exist.')


