"""Deal with labels
"""

import yaml
import regex as re

NORM_LABEL_FILE = 'label_norm.yaml'
_LABELS: None = None

def read_labels_from_file(label_identifier) -> list:
    """
    Read labels from a file.

    :param label_identifier: identifies the file with labels by a name, e.g. semeval2020
    :return: list of labels
    """
    global _LABELS
    if _LABELS is None:
        _LABELS = read_label_definitions(NORM_LABEL_FILE)
    filename = _LABELS['label_files'][label_identifier]
    print(f"Reading {filename}")
    with open(filename, 'r') as f:
        labels = [l.strip() for l in f.readlines() if l.strip() != '']
    return labels


def read_label_definitions(filename: str) -> dict:
    """Read yaml file with label definitions.

    Format of YAML file:

    label_files:
        file1: location
        file2: location

    translate:
        old: new

    from_group:
        old:
            - new1
            - new2

    """
    with open(filename, 'r') as f:
        translate = yaml.load(f, Loader=yaml.SafeLoader)
    return translate


def normalize_label(label: str) -> list:
    """Normalize label to a standard set. Return a list"""
    global _LABELS
    if _LABELS is None:
        _LABELS = read_label_definitions(NORM_LABEL_FILE)
    if label in _LABELS['translate'].keys():
        return [_LABELS['translate'][label]]
    elif label in _LABELS['label_to_group'].keys():
        # Originally: return all (new) labels that came from splitting an old one
        #return _LABELS['label_to_group'][label]
        # Now: ignore labels that have been split, so as not to learn false positives
        return []
    else:
        print("Untranslated: ", label)
        return [label]


def label_to_symbol(label: str, all_labels: list) -> str:
    """Convert a label to start and end of a special symbol to use as input or output for encoder/decoder"""
    index = all_labels.index(label)
    in_symbol = f"[i-{index}]"
    out_symbol = f"[o-{index}]"
    return in_symbol, out_symbol


def symbol_to_label(symbol: str, all_labels: list) -> str:
    """Convert a label to a special symbol to use as input or output for encoder/decoder"""
    m = re.search("[i-(\d+)]", symbol)
    n = re.search("[o-(\d+)]", symbol)
    if m is None and n is None:
        raise ValueError(f"Symbol {symbol} fails to match symbol regex")
    elif m is not None:
        return all_labels[m.group(1)]
    else:
        return all_labels[n.group(1)]



if __name__ == '__main__':
    import pprint

    labels = read_labels_from_file('semeval2020')
    pprint.pprint([l for l in labels])

    labels2 = read_labels_from_file('semeval2021')
    pprint.pprint([normalize_label(l) for l in labels])

