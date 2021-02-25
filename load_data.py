"""Load SemEval 2020 Task 11 data

Loaded data is kept in the format:
[
    {'id': <article_id>,
     'text': <full text of article>,
     'fragments': [
        (start, end, type),
        (...)
        ]
    }
    ...
]

"""
import os
from functools import partial
import pickle
import json
import pandas as pd
from abc import abstractmethod
from fragment_utils import (
    sentence_index_for_fragment_index,
    sentence_indexes_for_fragment,
    get_fragment,
    get_sentences_content,
    split_sentences_multi
)
from fragment import Fragment
from spanfilter import SpanFilter
import pipeline_config
from label_tool import normalize_label, read_labels_from_file

logger = pipeline_config.logger

TRAINING_ARTICLES_2020_DIR = 'propaganda_detection/datasets/train-articles'

# Accept that data is lost. In practice this is about losing labels when parts of articles
# are removed for other reasons.
ACCEPT_DATA_LOSS = True

class DataLoader(object):

    def __init__(self, label_identifier=None):
        self.dir = dir
        self.data = []
        self.labels = []
        self.label_identifier = label_identifier
        if label_identifier is not None:
            self.labels = read_labels_from_file(label_identifier)
        self.config = pipeline_config.config
        #if task_name:
        #    self.task_config = self.config['data'][task_name]
        self.read_articles = {}

    def read_article(self, article_id: str) -> str:
        p = os.path.join(self.task_config['dir'], 'train-articles', f'article{article_id}.txt')
        if article_id in self.read_articles.keys():
            print(f" ERROR :  re-reading article {article_id}")
        self.read_articles[article_id] = True
        with open(p, 'r', encoding='utf8') as f:
            return f.read()

    def save_pkl(self, data, labels=None):
        """Save tuple of (labels, data)"""
        if not labels:
            labels = self.labels
        with open(self.config.get_output_data_dir() + self.task_config['pkl_file'], 'wb') as f:
            pickle.dump((labels, data), f)

        # Don't bother configuring name of labels file
        with open(self.config.get_output_data_dir() + 'labels.pkl', 'wb') as f:
            pickle.dump(labels, f)

    def load_labels_pkl(self):
        """Return labels"""
        # Don't bother configuring name of labels file
        labels_pkl_file = self.config.get_output_data_dir() + 'labels.pkl'
        if not os.path.exists(labels_pkl_file):
            raise FileNotFoundError(f'Failed to find labels file {labels_pkl_file}')
        else:
            with open(labels_pkl_file, 'rb') as f:
                labels = pickle.load(f)
                self.labels = labels
                return labels

    def load(self):
        """Return labels, data if set, else return load_fragment_pkl()
        FIXME: generalize, any type of fragment annotation (including sentences) should be loadable.
        """
        if self.labels and self.data:
            return self.labels, self.data
        else:
            return self.load_fragment_pkl()

    def load_pkl(self):
        """Returns tuple of (labels, data), where data ia a list of article fragments"""
        pkl_file = self.config.get_output_data_dir() + self.task_config['pkl_file']
        if not os.path.exists(pkl_file):
            return None
        else: 
            with open(pkl_file, 'rb') as f:
                return pickle.load(f)

    def save_fragment_pkl(self, labels, data):
        """Save tuple of (labels, data)"""
        with open(self.config.get_output_data_dir() + self.task_config['pkl_fragment_file'], 'wb') as f:
            pickle.dump((labels, data), f)

    def load_fragment_pkl(self):
        """Returns tuple of (labels, data), where data is a list of sequences (id, text, fragments)"""
        pkl_fragment_file = self.config.get_output_data_dir() + self.task_config['pkl_fragment_file']
        if not os.path.exists(pkl_fragment_file):
            return None, None
        else:
            with open(pkl_fragment_file, 'rb') as f:
                self.labels, self.data = pickle.load(f)
                return self.labels, self.data

    def set_data(self, data):
        self.data = data

    def get_data(self, label_identifier=None):
        pkl_file = self.config.get_output_data_dir() + self.task_config['pkl_fragment_file']
        # FIXME: disabled reading from pickle file, too many operational problems.
        if False and os.path.exists(pkl_file):
            return self.load_pkl()
        else:
            labels, data = self._get_data(label_identifier=label_identifier)
            self.save_pkl(data, labels=labels)
            return labels, data

    def label_to_int(self, label):
        return self.labels.index(label)

    def int_to_label(self, n):
        if not self.labels or n >= len(self.labels):
            raise IndexError(f'index {n} too large for {self.labels}')
        return self.labels[n]


    @abstractmethod
    def _get_data(self, label_identifier=None):
        """Override in subclass"""
        pass


def read_article(article_id: str) -> str:
    """
    FIXME: using hard-coded path to directory with training articles
    :param self:
    :param article_id:
    :return:
    """
    global TRAINING_ARTICLES_2020_DIR
    p = os.path.join(TRAINING_ARTICLES_2020_DIR, f'article{article_id}.txt')
    with open(p, 'r', encoding='utf8') as f:
        return f.read()


class DataLoader2020_task2(DataLoader):
    def __init__(self):
        super(DataLoader2020_task2, self).__init__(task_name='2020-task11-2')

    def _get_data(self, label_identifier=None):
        """Original label_identifier is 'semeval2020', translate to 'semeval2021'
        If label_identifier is set, we translate here in _get_data, so that happens
        only when parsing the original dataset
        """
        self.labels = read_labels_from_file(label_identifier)

        # File containing labelled data
        label_file = os.path.join(self.task_config['dir'], self.task_config['label_file'])
        with open(label_file, 'r') as f:
            lst = f.readlines()

        self.data = []
        prev_article_id = -1
        fragments = []
        for l in lst:
            article_id, p_type, start, end = l.strip().split('\t')

            article_id = int(article_id)
            if article_id == prev_article_id:
                # FIXME: hardcoded exceptions for label_identifiers
                if label_identifier != 'semeval2020':
                    for n_p_type in normalize_label(p_type):
                        fragments.append(Fragment(int(start), int(end), n_p_type))
                else:
                    fragments.append(Fragment(int(start), int(end), p_type))
            else:
                if prev_article_id != -1:
                    # Add the previous article
                    #print("APPENDING article ", prev_article_id)
                    fragments.sort()  # Sort on first tuple element, that is, 'start'
                    self.data.append({
                        'id': prev_article_id,
                        'article': self.read_article(prev_article_id),
                        'fragments': fragments
                    })

                # Prepare the new one
                prev_article_id = article_id
                fragments = []
                if label_identifier != 'semeval2020':
                    for n_p_type in normalize_label(p_type):
                        fragments.append(Fragment(int(start), int(end), n_p_type))
                else:
                    fragments.append(Fragment(int(start), int(end), p_type))

        if fragments != []:
            #print("Appending ", prev_article_id)
            self.data.append({
                'id': prev_article_id,
                'article': self.read_article(prev_article_id),
                'fragments': fragments
            })
        print("Total fragments = ", sum([len(d['fragments']) for d in self.data]))
        return self.labels, self.data


class DataLoader2021_task2(DataLoader):
    def __init__(self, task_name='2021-task6-2'):
        super(DataLoader2021_task2, self).__init__(task_name=task_name)
        self.labels = read_labels_from_file("semeval2021")
        print("DATALOADER 2021", "+"*20)

    def _get_data(self, label_identifier="semeval2021"):
        self.labels = read_labels_from_file(label_identifier)
        self.data = []
        training_set_file = os.path.join(self.task_config['dir'], self.task_config['label_file'])
        file_list = [x.strip() for x in training_set_file.split(',')]
        for file_name in file_list:
            with open(file_name, 'r', encoding='utf8') as f:
                json_data = json.load(f)

            for item in json_data:
                id = item['id']
                txt = item['text']
                labels = item['labels']  # labels = [{start:, end:, technique:, text_fragment}, ...]
                # We skip that 'text_fragment' as it can be derived from text and start/end
                fragments = []
                for frag in labels:
                    if frag['text_fragment'].strip() == '':
                        # Deal with a '\n' fragment by skipping it
                        continue
                    fragments.append(Fragment(frag['start'], frag['end'], frag['technique']))
                fragments.sort()
                self.data.append({'id': id, 'article': txt, 'fragments': fragments})
        return self.labels, self.data


class DataLoader2021_task3(DataLoader):
    def __init__(self, task_name='2021-task6-3'):
        super(DataLoader2021_task3, self).__init__(task_name=task_name)
        self.labels = read_labels_from_file("semeval2021_3")
        print("DATALOADER 2021 - 3", "+"*20)

    def _get_data(self, label_identifier=None):
        training_set_file = os.path.join(self.task_config['dir'], self.task_config['label_file'])
        self.data = []
        file_list = [x.strip() for x in training_set_file.split(',')]
        for file_name in file_list:
            with open(file_name, 'r', encoding='utf8') as f:
                json_data = json.load(f)

            for item in json_data:
                id = item['id']
                txt = item['text']
                labels = item['labels']  # list of labels for the fragment
                image = item['image']  # reference to image file
                self.data.append({'id': id, 'article': txt, 'labels': labels, 'image': image})

        return self.labels, self.data


def loadJSON2021(path: str):
    """
    :param path: Comma separated list of paths to JSON files
    :return:
    """
    data = []
    file_list = [x.strip() for x in path.split(',')]
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf8') as f:
            json_data = json.load(f)

        for item in json_data:
            id = item['id']
            txt = item['text']
            fragments = []

            if not 'image' in item.keys():
                # Variation 1: labels =
                labels = item['labels']  # labels = [{start:, end:, technique:, text_fragment}, ...]
                # We skip that 'text_fragment' as it can be derived from text and start/end
                for frag in labels:
                    if frag['text_fragment'].strip() == '':
                        # Deal with a '\n' fragment by skipping it
                        continue
                    fragments.append(Fragment(frag['start'], frag['end'], frag['technique']))
            else:
                # Variation 2:
                start = 0
                end = len(txt)
                for l in item['labels']:
                    fragments.append(Fragment(start, end, l))
                # There is an image, but we ignore it
                image = item['image']

            fragments.sort()
            data.append({'id': id, 'article': txt, 'fragments': fragments})
    return data

def loadTAB2020(path: str):
    """Original label_identifier is 'semeval2020', translate to 'semeval2021'
    If label_identifier is set, we translate here in _get_data, so that happens
    only when parsing the original dataset
    """
    # self.labels = read_labels_from_file(label_identifier)

    # File containing labelled data
    #label_file = os.path.join(self.task_config['dir'], self.task_config['label_file'])
    with open(path, 'r') as f:
        lst = f.readlines()

    data = []
    prev_article_id = -1
    fragments = []
    for l in lst:
        article_id, p_type, start, end = l.strip().split('\t')

        article_id = int(article_id)
        if article_id == prev_article_id:
            # FIXME: hardcoded exceptions for label_identifiers
            #if label_identifier != 'semeval2020':
            #    for n_p_type in normalize_label(p_type):
            #        fragments.append(Fragment(int(start), int(end), n_p_type))
            #else:
            fragments.append(Fragment(int(start), int(end), p_type))
        else:
            if prev_article_id != -1:
                # Add the previous article
                # print("APPENDING article ", prev_article_id)
                fragments.sort()  # Sort on first tuple element, that is, 'start'
                data.append({
                    'id': prev_article_id,
                    'article': read_article(prev_article_id),
                    'fragments': fragments
                })

            # Prepare the new one
            prev_article_id = article_id
            fragments = []
            #if label_identifier != 'semeval2020':
            #    for n_p_type in normalize_label(p_type):
            #        fragments.append(Fragment(int(start), int(end), n_p_type))
            #else:
            #    fragments.append(Fragment(int(start), int(end), p_type))

    if fragments != []:
        # print("Appending ", prev_article_id)
        data.append({
            'id': prev_article_id,
            'article': read_article(prev_article_id),
            'fragments': fragments
        })
    print("Total fragments = ", sum([len(d['fragments']) for d in data]))
    return data


class CSVLoader(DataLoader):
    def __init__(self):
        super(CSVLoader, self).__init__()
        self.ready_made_data = True

    def set_csv_file(self, csv_file: str):
        self.csv_file = csv_file

    def get_dataframe(self):
        df = pd.read_csv(self.csv_file)
        print(f'CSVLoader: getting training data in dataframe from {self.csv_file} - len={len(df)}')
        return df
        
    def _get_data(self, label_identifier=None):
        raise NotImplementedError("This method should not be called for CSVLoader")



def XXXtask11_2020_2():
    dl = DataLoader2020_task2()
    labels, data = dl.load_fragment_pkl()
    if not data or pipeline_config.config['data']['overwrite']:
        sfilter = SpanFilter()
        labels, data = dl.get_data()
        new_data = []

        for d in data:
            id = d['id']
            fragments = d['fragments']
            article = d['article']
            #article = dl.read_article(article_id)
            if sfilter.check_fragments(article, fragments):
                article, fragments = sfilter.filter_multisentence_segment(article, fragments)
                article, fragments = sfilter.filter_multicategory_segment(article, fragments)
                #article, fragments = sfilter.filter_one_label(article, fragments)

                new_data.append({'id': id, 'article': article, 'fragments': fragments})
        dl.save_fragment_pkl(labels, new_data)
    return dl


def XXXredux():
    dl = DataLoader2020_task2()
    label_identifier = pipeline_config.config.pipeline_config['data']['labels']
    dl.get_data(label_identifier=label_identifier)
    return dl


def XXXtask6_combi():
    dl1 = redux()
    dl2 = task6_2021_1_2()

    dl2.data.extend(dl1.data)
    return dl2

def XXXtask6_1_2_train_dev():
    dl1 = task6_2021_1_2()
    dl2 = task6_2021_1_2_dev()

    dl2.data.extend(dl1.data)
    return dl2

def XXXtask6_3_train_dev():
    dl1 = task6_2021_3()
    dl2 = task6_2021_3_dev()

    dl2.data.extend(dl1.data)
    return dl2

def translate(labels: list, data: list) -> tuple:
    """Translate one set of labels to another"""
    new_data = []
    for d in data:
        id = d['id']
        article = d['article']
        n_fragments = []
        for start, end, p_type in d['fragments']:
            for n_p_type in normalize_label(p_type):
                n_fragments.append((start, end, n_p_type))
        new_data.append({'id': id, 'article': article, 'fragments': n_fragments})
    new_labels = read_labels_from_file(pipeline_config.config.pipeline_config['data']['labels'])
    return new_labels, new_data


def task6_2021_1_2():
    dl = DataLoader2021_task2()
    label_identifier = pipeline_config.config.pipeline_config['data']['labels']
    labels, data = dl.load_fragment_pkl()
    if not data or pipeline_config.config['data']['overwrite']:
        labels, data = dl.get_data(label_identifier=label_identifier)
        new_data = []
        for d in data:
            id = d['id']
            article = d['article']
            fragments = d['fragments']
            new_data.append({'id': id, 'article': article, 'fragments': fragments})
        dl.save_fragment_pkl(labels, new_data)
    return dl


def task6_2021_1_2_dev():
    dl = DataLoader2021_task2(task_name='2021-task6-2-dev')
    label_identifier = pipeline_config.config.pipeline_config['data']['labels']
    labels, data = dl.load_fragment_pkl()
    if not data or pipeline_config.config['data']['overwrite']:
        labels, data = dl.get_data(label_identifier=label_identifier)
        new_data = []
        for d in data:
            id = d['id']
            article = d['article']
            fragments = d['fragments']
            new_data.append({'id': id, 'article': article, 'fragments': fragments})
        dl.save_fragment_pkl(labels, new_data)
    return dl


def task6_2021_3():
    dl = DataLoader2021_task3()
    labels, data = dl.load_fragment_pkl()
    if not data or pipeline_config.config['data']['overwrite']:
        labels, data = dl.get_data()
        dl.labels = labels
        new_data = []
        for d in data:
            id = d['id']
            article = d['article']
            start = 0
            end = len(article)
            fragments = []
            for l in d['labels']:
                fragments.append((start, end, l))
            image = d['image']
            new_data.append({'id': id, 'article': article, 'fragments': fragments, 'image': image})
        dl.save_fragment_pkl(labels, new_data)
        dl.data = new_data
        dl.save_pkl(new_data, labels=labels)
        dl.save_fragment_pkl(labels, new_data)
    return dl

def task6_2021_3_dev():
    dl = DataLoader2021_task3(task_name='2021-task6-3-dev')
    labels, data = dl.load_fragment_pkl()
    if not data or pipeline_config.config['data']['overwrite']:
        labels, data = dl.get_data()
        dl.labels = labels
        new_data = []
        for d in data:
            id = d['id']
            article = d['article']
            start = 0
            end = len(article)
            fragments = []
            for l in d['labels']:
                fragments.append((start, end, l))
            image = d['image']
            new_data.append({'id': id, 'article': article, 'fragments': fragments, 'image': image})
        dl.save_fragment_pkl(labels, new_data)
        dl.data = new_data
        dl.save_pkl(new_data, labels=labels)
        dl.save_fragment_pkl(labels, new_data)
    return dl


########## Filters #########################################
"""
In configuration, filters are defined under 'data' and presented as a list.
The 'filter_' prefix is not presented, it serves only to clarify here in code what kind of functions these are.
"""

def check_change(func):
    """Decorator for filters: logs change in number of items"""
    def wrapper_check_change(dataloader: DataLoader):
        length_before = len(dataloader.data)
        output_dataloader = func(dataloader)
        length_after = len(output_dataloader.data)
        if length_before == length_after:
            msg = f'Number of items unchanged'
        elif length_before > length_after:
            msg = f'removed {length_before - length_after} to new total of {length_after} items'
        else:
            msg = f'added {length_after - length_before} to new total of {length_after} items'
        logger.info(f'{func.__name__}: {msg}')
        return output_dataloader
    return wrapper_check_change


@check_change
def filter_sentence_splitter(dataloader: DataLoader) -> DataLoader:
    """
    Split articles into sets of sentences covering fragments.


    :param dataloader:
    :return: dataloader
    """
    new_data = []
    for d in dataloader.data:
        id = d['id']
        article = d['article']
        fragments = d['fragments']
        lst = split_sentences_multi(id, article, fragments, include_empty=False)
        new_data.extend(lst)
    dataloader.data = new_data
    return dataloader

@check_change
def filter_sentence_splitter_with_empty(dataloader: DataLoader) -> DataLoader:
    """
    Split articles into sets of sentences covering fragments.


    :param dataloader:
    :return: dataloader
    """
    new_data = []
    for d in dataloader.data:
        id = d['id']
        article = d['article']
        fragments = d['fragments']
        lst = split_sentences_multi(id, article, fragments, include_empty=True)
        new_data.extend(lst)
        # If the article length is small, also include the original article
        #if len(article) < 400 and len(lst) > 1:
        #    new_data.append(d)
    dataloader.data = new_data
    return dataloader


@check_change
def filter_eliminate_short(dataloader: DataLoader) -> DataLoader:
    """Filter out items with too few characters as they are unlikely to represent a technique"""
    MIN_CHARACTERS = 10
    new_data = []
    for d in dataloader.data:
        if len(d['article']) > MIN_CHARACTERS:
            new_data.append(d)
    dataloader.data = new_data
    return dataloader


@check_change
def filter_eliminate_long(dataloader: DataLoader) -> DataLoader:
    """Filter out items with too many characters as they are unlikely to represent a technique"""
    MAX_CHARACTERS = 300
    new_data = []
    for d in dataloader.data:
        if len(d['article']) <= MAX_CHARACTERS:
            new_data.append(d)
    dataloader.data = new_data
    return dataloader


@check_change
def filter_translate_2020_2021_1_2(dataloader: DataLoader) -> DataLoader:
    """Normalize 2020 label data to 2021 label data"""
    new_data = []
    for d in dataloader.data:
        new_fragments = []
        for f in d['fragments']:
            new_label = normalize_label(f.label)
            # We ony move labels if there is a translation for them, otherwise we ignore the fragment
            if new_label:
                f.label = new_label[0]
                new_fragments.append(f)
        d['fragments'] = new_fragments
        new_data.append(d)
    dataloader.set_data(new_data)
    # print(dataloader.data[:10])
    return dataloader

@check_change
def filter_lowercase(dataloader: DataLoader) -> DataLoader:
    """Converts text of all items to lowercase"""
    new_data = []
    for d in dataloader.data:
        d['article'] = d['article'].lower()
        new_data.append(d)
    dataloader.data = new_data
    return dataloader

@check_change
def filter_duplicate_fragments(dataloader: DataLoader) -> DataLoader:
    """
    Extract fragments from sentences and add them to the dataset.

    Standard data contains (sets of) sentences covering fragments.
    Now we add the fragments phrases as independent data items.
    :return:
    """
    MIN_CHARS = 10
    added_data = []
    skipped = 0
    for d in dataloader.data:
        fragments = d['fragments']
        for index, (start, end, p_type) in enumerate(fragments):
            add_id = d['id'] + '_' + str(index)
            add_article = d['article'][start:end]
            add_fragments = [(0, len(add_article), p_type)]
            if len(add_article) > MIN_CHARS:
                added_data.append({'id': add_id, 'article': add_article, 'fragments': add_fragments})
            else:
                skipped += 1

    if skipped > 0:
        logger.info(f'duplicate_fragments: skipped {skipped} fragments as they are shorter than {MIN_CHARS} chars')
    dataloader.data = dataloader.data + added_data
    return dataloader


if __name__ == '__main__':
    dl = task11_2020_2()
    #dl = task6_2021_2()
    labels, data = dl.get_data()
    new_labels, new_data = translate(labels, data)
    count = 0
    for id, v in new_data.items():
        #print(v)
        count = count + 1
        if count > 10:
            break
    print("done")

