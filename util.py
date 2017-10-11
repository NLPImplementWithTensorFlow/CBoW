import numpy as np
import random

def mk_dict(data_path):
    with open(data_path, "r") as fs:
        lines = fs.readlines()
    
    words = []
    [[words.append(word) for word in line.split(" ")] for line in lines]
    words = list(set(words))
    return words

def read_dict(read_path):
    with open(read_path, "r") as fs:
        lines = fs.readlines()

    dict_ = [word.split("\n")[0] for word in lines]
    return dict_

def save_dict(dict_, save_path):
    content = "\n".join(dict_)
    with open(save_path, "w") as fs:
        fs.write(content)

def read_sentence(read_path):
    with open(read_path, "r") as fs:
        lines = fs.readlines

    return [line.split("\n")[0] for line in lines]

def convert_sentence2index(sentence, dict_, max_time_step):
    indexs = [dict_.index(word) for word in sentence.split(" ")]
    while len(indexs) != max_time_step and len(indexs) <= max_time_step:
        indexs.append(len(dict_)) # Add <EOS>
    return np.array(indexs[:max_time_step])

def mk_train_func(batch_size, max_time_step, sentence_read_path, dict_read_path):
    dict_ = read_dict(dict_read_path)
    sentence = np.array(read_sentence(sentence_read_path))
    sentence = sentence[:-(sentence.shape[0]%batch_size)]

    dump = []
    def func():
        t_len = len(sentence)
        while True:
            if len(dump) != t_len:
                random_selected_index = random.sample(ange(len(sentence)), batch_size)
                choiced_s = [convert_sentence2index(s, dict_, max_time_step) for s in sentence[random_selected_index].tolist()]
                dump += choiced_s
                np.delete(converted, random_selected_index)
            else:
                

            
            yield 

