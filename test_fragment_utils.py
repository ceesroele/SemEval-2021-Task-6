from fragment_utils import (
    preceding_non_labelled_sentences,
    get_fragment,
    next_nl_or_end,
    split_sentences_multi_until,
    split_sentences_multi,
    split_in_sentences,
    merge_short_sentences,
    calibrate,
    encode,
    decode,
    surrounding_word,
    next_token,
    tokenize_string,
    spacy_tokenize,
    insert_tags_list,
    insert_tags
)

from fragment import Fragment

# Need some real data
#from load_data import redux

article = 'abcd\n1234\npqrs'
article_extra = 'abcd\n\n1234\n\npqrs'

f1 = Fragment(1, 2, 'f1')
f2 = Fragment(6, 8, 'f2')
f3 = Fragment(11, 13, 'f3')

article_2 = 'aaaaa\n\nbbbbb\nccccc\n\n01234'
fragments_2 = [
    Fragment(2, 6, 'foo'),
    Fragment(4, 8, 'bar'),
    Fragment(21, 23, 'foo')
]


def test_next_nl_or_end():
    s = 'abcd\n\npqrs'
    n = next_nl_or_end(s)
    assert n == 6
    s = 'abcd\n\npqrs\n'
    n = next_nl_or_end(s, 4)
    assert n == 11
    s = 'abcd\n\npqrs'
    n = next_nl_or_end(s, 8)
    assert n == 10
    s = 'abcd\n\npqrs\n\nvwxyz'
    n = next_nl_or_end(s, 4)
    assert n == 12
    assert s[:n] == 'abcd\n\npqrs\n\n'
    assert s[n:] == 'vwxyz'

    n2 = next_nl_or_end(article_2, 9)
    assert n2 == 13

def test_preceding_non_labelled_sentences():
    id = '111'
    section = get_fragment(article_extra, f2)
    sentences, start = preceding_non_labelled_sentences(id, article_extra, [f2])
    new_article = article_extra[start:]
    assert sentences == ['abcd\n\n']
    assert new_article == '1234\n\npqrs'
    assert start == 6
    section_after = get_fragment(article_extra[start:], f2, offset=start)
    assert section == section_after


def test_split_sentences_multi_until():
    id = '111'
    start_text, start_fragments, continuation_text, rest_fragments, new_start = \
        split_sentences_multi_until(article_2, fragments_2, start_index=0)

    assert start_text == 'aaaaa\n\nbbbbb\n'
    assert start_fragments == [Fragment(2, 6, 'foo'), Fragment(4, 8, 'bar')]
    assert continuation_text == 'ccccc\n\n01234'
    assert rest_fragments == [Fragment(8, 10, 'foo')]
    assert new_start == 13
    assert article_2[new_start] == 'c'
    assert article_2[new_start-1] == '\n'



def test_split_sentences_multi():
    id = '111'
    part = get_fragment(article_2, fragments_2[2])
    assert part == '12'

    data = split_sentences_multi(id, article_2, fragments_2, include_empty=True) #, start_index=0, data=[])

    print(data)

    assert data[0] == {'id': '111x', 'article': 'aaaaa\n\nbbbbb\n', 'fragments': [Fragment(2, 6, 'foo'), Fragment(4, 8, 'bar')]}

    assert data[1] == {'id': '111_0p0', 'article': 'ccccc\n\n', 'fragments': []}

    #assert get_fragment(data[2]['article'], data[2]['fragments'][0]) == '12'
    assert data[2] == {'id': '111_0x', 'article': '01234', 'fragments': [Fragment(1, 3, 'foo')]}

    # Add a fragment that starts after the previous fragments,
    # but starts before the end of the sentence included with the previous fragments.
    fragments_2.append(Fragment(9, 10, 'tolstoy'))
    fragments_2.sort()

    data = split_sentences_multi(id, article_2, fragments_2, include_empty=True)
    assert data[0] == {
        'id': '111x',
        'article': 'aaaaa\n\nbbbbb\n',
        'fragments': [Fragment(2, 6, 'foo'), Fragment(4, 8, 'bar'), Fragment(9, 10, 'tolstoy')]}

    assert data[1] == {'id': '111_0p0', 'article': 'ccccc\n\n', 'fragments': []}

    assert get_fragment(data[2]['article'], data[2]['fragments'][0]) == '12'
    assert data[2] == {'id': '111_0x', 'article': '01234', 'fragments': [Fragment(1, 3, 'foo')]}

def test_real_article():

    fragments = [Fragment(543, 683, 'Causal Oversimplification'),
                 Fragment(582, 611, 'Loaded Language'), Fragment(835, 844, 'Loaded Language'),
                 Fragment(1476, 1483, 'Loaded Language'), Fragment(1929, 2341, 'Appeal to authority'),
                 Fragment(2045, 2051, 'Loaded Language'), Fragment(2984, 3018, 'Name calling/Labeling'),
                 Fragment(3045, 3084, 'Loaded Language'), Fragment(3262, 3319, 'Exaggeration/Minimisation'),
                 Fragment(3286, 3319, 'Bandwagon'), Fragment(3286, 3319, 'Reductio ad hitlerum'),
                 Fragment(4433, 4498, 'Doubt'), Fragment(4580, 4590, 'Loaded Language'), Fragment(4583, 4590, 'Repetition'),
                 Fragment(6524, 6539, 'Loaded Language'), Fragment(6813, 6839, 'Loaded Language'),
                 Fragment(7009, 7036, 'Exaggeration/Minimisation'), Fragment(7134, 7152, 'Loaded Language')]

    article_file_name = "propaganda_detection/datasets/train-articles/article111111135.txt"

    with open(article_file_name, 'r') as f:
        article = f.read()

    data = split_sentences_multi(id, article, fragments, include_empty=True)
    for f in fragments:
        print(f"{f.label} [{get_fragment(article, f)}]")
    assert len(data) == 47



def test_split_in_sentences():
    s = split_in_sentences('')
    assert s == []

    s = split_in_sentences('a\n')
    assert s == ['a\n']

    s = split_in_sentences('aaa\n')
    assert s == ['aaa\n']

    s = split_in_sentences('aaa\n\n')
    assert s == ['aaa\n\n']


    a = 'aaa\nbbb\n123'
    b = a + '\n'
    c = a + '\n\n'
    s = split_in_sentences(a)
    assert s == ['aaa\n', 'bbb\n', '123']

    s2 = split_in_sentences(b)
    assert s2 == ['aaa\n', 'bbb\n', '123\n']

    s3 = split_in_sentences(c)
    assert s3 == ['aaa\n', 'bbb\n', '123\n\n']

def test_calibrate():
    match_text = 'hello sad big world!'
    orig_text = 'goodbye sad big world!'
    fragment = Fragment(6, 13, 'foo')

    assert fragment.extract(match_text) == 'sad big'
    res = calibrate(fragment, match_text, orig_text, distance=3)
    print('new fragment = ', res)
    assert res == Fragment(8, 15, 'foo')

    match_text = ' hello sad big world!'
    orig_text = 'hello sad big world!'
    fragment = Fragment(1, 6, 'foo')

    assert fragment.extract(match_text) == 'hello'
    res = calibrate(fragment, match_text, orig_text, distance=3)
    print('new fragment = ', res)
    assert res == Fragment(0, 5, 'foo')

def test_surrounding_word():
    s = ' hello, world!'

    index = 10
    assert s[index] == 'r'
    start, end = surrounding_word(s, index)
    assert start == 8
    assert end == 13
    assert s[start:end] == 'world'

    index = 10
    assert s[index] == 'r'
    start, end = surrounding_word(s, index, with_line_end=True)
    assert start == 8
    assert end == 14
    assert s[start:end] == 'world!'

    index = 7
    assert s[index] == ' '
    res = surrounding_word(s, index)
    assert res is None

    index = 4
    assert s[index] == 'l'
    start, end = surrounding_word(s, index)
    assert start == 1
    assert end == 6
    assert s[start:end] == 'hello'



def test_decode():
    all_labels = ['one', 'two']
    s = "abc [i-0] def [o-0] ghi [i-1] jk [o-1]"
    article, fragments, errors = decode(s, all_labels)
    print(article)
    assert len(article) == 14
    assert len(fragments) == 2
    assert fragments[0].extract(article) == 'def'
    assert fragments[1].extract(article) == 'jk'

    all_labels1 = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'elven',
                   'twelves', 'thirteen', 'fourteen','fifteen','sixteen', 'seventeen']
    s1 = 'How to [i-1] vaccinate [o-1], ahem test'
    article1, fragments1, errors1 = decode(s1, all_labels1)
    print(article1)
    assert len(article1) == 27
    assert len(fragments1) == 1
    print(fragments1[0])
    assert fragments1[0].extract(article1) == 'vaccinate'

    s2 = '[i-16]  Got 195,000 + Americans [i-8] KILLED [o-8].\nFought scientists and doctors'
    article2, fragments2, errors2 = decode(s2, all_labels1)
    print(article2)
    assert len(article2) == 62
    assert len(fragments2) == 1
    print(fragments2[0])
    assert fragments2[0].extract(article2) == 'KILLED'


def test_next_token():
    s = " Hello, world!"

    t = next_token(s)
    assert t == 'Hello'

    index = 6
    assert s[index] == ','
    t = next_token(s[index:])
    assert t == 'world'

    assert next_token(s, index=index) == 'world'


def test_tokenize_string():
    s = " Hello, world!"

    lst = tokenize_string(s)

    assert len(lst) == 2
    assert lst[0] == 'Hello'
    assert lst[1] == 'world'

    s2 = '[i-16]  Got 195,000 + Americans [i-8] KILLED [o-8].\nFought scientists and doctors'
    lst = tokenize_string(s2)
    print(lst)

    assert len(lst) == 12

    assert lst[0] == '[i-16]'
    # number is split at comma, no problem as long as this happens in both original and prediction
    assert lst[3] == '000'


def test_spacy_tokenize():
    s = '[i-16]  Got 195,000 + Americans [i-8] KILLED [o-8].\nFought scientists and doctors'
    lst = spacy_tokenize(s)

    assert len(lst) == 15
    assert lst[0] == '[i-16]'


def test_insert_tags_list():
    orig_text = 'President Trump is Mentally Sick, Yes -- Truly ill in how he manages his Financial Life and his handling of Whitehouse policy and management !!!'
    tagged_text = '[i-16] President Trump is [i-8] Mentally Sick [o-8], Yes -- [i-8] Truly ill [o-8] in how he manages his Financial Life and his handling of Whitehouse policy and management!!! [o-16]'

    res = insert_tags_list(tagged_text, orig_text)
    print('insert tags list', res)

    assert len(res) == 34
    assert res[-1] == '[o-16]'

    out_text = insert_tags(orig_text, res)

    print('out text', out_text)

    assert len(out_text) == 188

    orig_text = 'A whoremonger, a pervert, and a pedophile walk into a bar.\n'
    tagged_text = 'A [i-10] [i-8] whoremonger [o-10] [o-8], a [i-10] pervert [o-10], and a [i-10] pedile [o-10] wins into a bar.\n'

    res = insert_tags_list(tagged_text, orig_text)
    print('insert tags list', res)

    assert len(res) == 19
    assert (res[1] == '[i-10]' and res[2] == '[i-8]') or (res[1] == '[i-8]' and res[2] == '[i-10]')

    out_text = insert_tags(orig_text, res)

    assert False


def test_merge_short_sentences():
    s = "\"LAW AND ORDER\"\n\nPLEADED GUILTY\n\nCONVICTED\n\nCONVICTED\n\nCONVICTED\n\nCONVICTED\n\nINDICTED"
    lst = split_in_sentences(s)

    out_lst = merge_short_sentences(lst)
    print('outlst', out_lst)

    assert len(out_lst) == 3

    assert out_lst[2] == 'CONVICTED\n\nCONVICTED\n\nCONVICTED\n\nCONVICTED\n\nINDICTED'

def test_find_all_labels_matching_or_not():
    s = ""
