import numpy as np
import random

def mk_metadata(dict_path):
    dict_ = read_dict(dict_path)
    with open("metadata.tsv", "w") as fs:
        fs.write("Index\tLabel\n")
        for index, label in enumerate(dict_):
            fs.write("%d\t%s\n" %(index, label))

def mk_dict(data_path):
    with open(data_path, "r") as fs:
        lines = fs.read().lower().split("\n")
    
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
        lines = fs.read().lower().split("\n")

    return lines

def convert_sentence2index(sentence, dict_, max_time_step):
    indexs = [dict_.index(word) for word in sentence.split(" ")]
    num_words = len(indexs)
    while len(indexs) != max_time_step and len(indexs) <= max_time_step:
        indexs.append(len(dict_)) # Add <EOS>
    return np.array(indexs[:max_time_step]), num_words

def extract_a_word(sentences, word_nums):
    choiced_label_idx = [random.choice(range(word_num)) for word_num in word_nums]
    label = [sentence[idx] for sentence, idx in zip(sentences, choiced_label_idx)]
    input_ = []
    for sentence, idx in zip(sentences, choiced_label_idx):
        c_sentence = list(sentence[:])
        del(c_sentence[idx])
        input_.append(c_sentence)

    return input_, label

def mk_train_func(batch_size, max_time_step, sentence_read_path, dict_read_path):
    dict_ = read_dict(dict_read_path)
    sentence = np.array(read_sentence(sentence_read_path))
    sentence = sentence[:-(sentence.shape[0]%batch_size)]

    def func():
        dump = []
        word_num = []
        t_len = len(sentence)
        while True:
            if len(dump) != t_len:
                random_selected_index = random.sample(range(len(sentence)), batch_size)
                converted = [convert_sentence2index(s, dict_, max_time_step) for s in sentence[random_selected_index].tolist()]
                choiced_s = [tuple_[0] for tuple_ in converted]
                dump += choiced_s
                choiced_word_num = [tuple_[1] for tuple_ in converted]
                word_num += choiced_word_num
                np.delete(sentence, random_selected_index)
            else:
                if type(np.array([])) != type(dump):
                    dump = np.array(dump)
                    word_num = np.array(word_num)

                idx = random.sample(range(len(dump)), batch_size)
                choiced_s = dump[idx].tolist()
                choiced_word_num = word_num[idx].tolist()

            yield extract_a_word(choiced_s, choiced_word_num)
    return func
