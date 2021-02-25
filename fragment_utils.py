"""Utility functions for fragments
"""

from fragment import Fragment
from label_tool import label_to_symbol, symbol_to_label
import regex as re
import random
import spacy
nlp = spacy.load("en_core_web_lg")


def is_tag(token):
    """Determine if a token is a tag of the format like [i-10] or [o-7]"""
    is_tag_re = re.compile('\[[io]-\d+\]', re.IGNORECASE)
    return is_tag_re.match(token)

def is_start_tag(token):
    """Determine if a token is a tag of the format like [i-10] or [o-7]"""
    is_tag_re = re.compile('\[i-\d+\]', re.IGNORECASE)
    return is_tag_re.match(token)

def is_end_tag(token):
    """Determine if a token is a tag of the format like [i-10] or [o-7]"""
    is_tag_re = re.compile('\[o-\d+\]', re.IGNORECASE)
    return is_tag_re.match(token)



def sentence_index_for_fragment_index(fragment_index: int, sentences: list) -> int:
    """Index of sentence within sentences list in which `fragment_index` is located."""

    cur_start = 0
    cur_end = -1
    total_length = sum([len(s) for s in sentences]) + len(sentences) - 1
    a = '\n'.join(sentences)

    assert total_length == len(a), f"total length {total_length}, length article {len(a)}"

    # print([(x[0], x[1]) for x in enumerate(a)])

    for i, s in enumerate(sentences):
        cur_end = cur_start + len(s)  # +1 for the '\n'
        # print(i, s, cur_start, cur_end)
        # cur_end is now at the '\n' that separates sentences
        # if a[cur_end] != '\n':
        # print(a[cur_end], cur_start, cur_end, '--', a[cur_start:cur_end], '++', '(', s, ')')
        if cur_start <= fragment_index <= cur_end:
            # print("sentence index: ", fragment_index, 'i=',i, s, cur_start, cur_end, sentences)
            return i
        else:
            #            orig_article = '\n'.join(sentences)
            #            print(f'Got character: _{orig_article[fragment_index]}_ at {fragment_index}')
            #            assert fragment_index != cur_end, f"Index {fragment_index} should not be at newline in {orig_article[:fragment_index+1]}"
            cur_start = cur_end + 1  # start of new sentence is next position after '\n'

    if cur_end < total_length:
        print("Something wrong with indexes")
    raise IndexError(f'Failed to find fragment index {fragment_index} in sentences len={cur_end}/{total_length}')


def sentence_indexes_for_fragment(fragment: Fragment, sentences: list) -> list:
    """Get the start and end indexes in the whole article for the sentences encompassing a fragment."""
    start_sentence_index = sentence_index_for_fragment_index(fragment.start, sentences)
    end_sentence_index = sentence_index_for_fragment_index(fragment.end, sentences)
    return list(range(start_sentence_index, end_sentence_index +1))


def get_fragment(article: str, fragment: Fragment, offset=0) -> str:
    """Gets a fragment from an article

    :param article: String from which to get the fragment
    :param fragment: (start, end, p_type) tuple where end is exclusive
    :param offset: if article is truncated, subtract offset from the fragment indices
    :return: String indicated by the fragment definition
    """
    f = fragment - offset
    return article[f.start:f.end]

def get_sentences_content(fragment: Fragment, sentences: list) -> str:
    """Get the content of the sentences that together contain a fragment"""
    a = '\n'.join(sentences)
    #f = get_fragment(a, fragment)
    s_is = sentence_indexes_for_fragment(fragment, sentences)
    return '\n'.join([sentences[x] for x in s_is])


def offset_fragments(fragments, offset):
    """Correct indices of fragments with an offset.

    When an article is stripped of its beginning, all fragments need to be reindexed
    with the length of what was stripped.
    """
    res = []
    for f in fragments:
        assert f.start >= offset, f'{str(f)} >= {offset}, fragments = {str(fragments)}'
        res.append(f - offset)  # Note: substraction from the Fragment subtracts from its start and end indexes
    return res

def count_fragments(data: list) -> int:
    n = 0
    for d in data:
        n += len(d['fragments'])
    return n


def split_sentences_multi(id: str, article: str, fragments: list, include_empty=False, level=0):
    """Split an article into multiple sections, each with its own list of fragments

    :param: include_empty: include items for which there are no fragments with labels (that is, no labels match)
    """
    #print("****** ID = ", id, 'level', level, 'start index', start_index, 'len article', len(article), "*******")
    data = []
    number_of_fragments = len(fragments)
    fragment_texts = [get_fragment(article, f) for f in fragments]

    #print('original fragments', fragment_texts, fragments)
    if fragments == []:
        # Just ignore this
        print(f"Ignoring article with length {len(article)} for empty fragments")
        # return data
    else:
        # forcibly sort fragments, shouldn't be necessary as they are supposed to be sorted already...
        fragments.sort()

        # First deal with sentences without fragment
        sentences, start = preceding_non_labelled_sentences(id, article, fragments)

        # Note that here 'sentences' really only contains the strings, not [id, string] pairs
        data.extend([{'id': f'{id}p{i}', 'article': s, 'fragments': []} for i, s in enumerate(sentences)])

        # Now add the first sections that contains labels.
        # Note that we must deal with the 'start' offset coming from preceding_non_labelled_sentences
        # in both article and fragments.
        start_text, start_fragments, rest_text, rest_fragments, new_start = \
            split_sentences_multi_until(
                article[start:],
                offset_fragments(fragments, start)
            )
        data.append({'id': f'{id}x', 'article': start_text, 'fragments': start_fragments})

        # Now recursively deal with the rest
        if rest_text == '':
            # Nothing to be done, just return the input data
            pass
        elif rest_fragments == []:
            # if there are no more fragments, split article in sentences up to the end
            new_data = [{'id': f'{id}r{i}', 'article': s, 'fragments': []} for i, s in enumerate(split_in_sentences(rest_text)) if s.strip() != '']
            #print("split in sentences, new data: ", new_data)
            data.extend(new_data)
        else:
            #print("now recursively dealing with ", rest_text, rest_fragments)
            data.extend(split_sentences_multi(f'{id}_{level}', rest_text, rest_fragments, include_empty=include_empty, level=level+1))

        # Remove empty fragments
        if not include_empty:
            new_data = [d for d in data if d['fragments'] != []]
            data = new_data

        new_fragment_texts = [get_fragment(d['article'], f) for d in data for f in d['fragments']]

        for t in new_fragment_texts:
            if level == 0 and t not in fragment_texts:
                found = False
                for ft in fragment_texts:
                    if ft.startswith(t):
                        found = True
                        print(f"+++ Output fragment {[t]} is truncated \n--- from {[ft]}")
                        break
                if not found:
                    print(f"=== Got output level {level} fragment text {str([t])} not in input fragments ", fragment_texts,
                          ' ** total output is ', new_fragment_texts)
                    print("All original ", len(fragments), "fragments: ", fragments) # , "original text: ", article)


        n = count_fragments(data)

        #print(f"Level {level} output of {n} fragments for input of input: {number_of_fragments}")

    return data


def next_nl_or_end(s, n=0):
    """Next newline or end of text"""
    # first identify starting newlines, we pass them
    start = n
    while start < len(s) - 1 and s[start] == '\n':
        start += 1
    p = s.find('\n', start + 1)
    if p > -1:
        # Another newline found, continue until no more newlines or end of string
        while p < len(s) and s[p] == '\n':
            p += 1
        return p
    else:
        # No (more) newlines found, return length of string as end index
        return len(s)


def split_sentences_multi_until(article: str, fragments: list, start_index=0):
    """Split into sentences until sentence not covered with any label

    :param article:
    :param fragments:
    :param start_index:
    :param X:
    :param y:
    :return:
    """
    index = 0
    fragment_texts = [get_fragment(article, f) for f in fragments]
    if fragments == []:
        # FIXME: to be implemented
        print("FIXME:  not dealing with empty fragments")
        assert False, "Should never get here with empty fragments"
    else:
        new_start = 0
        max_end_so_far = 0
        while index < len(fragments):
            max_end_so_far = max(max_end_so_far, fragments[index].end)
            new_start = next_nl_or_end(article, max_end_so_far)
            if index + 1 < len(fragments) and new_start > fragments[index+1].start:
                index += 1
            else:
                break
        start_text = article[:new_start]
        rest_text = article[new_start:]
        start_fragments = offset_fragments(fragments[:index+1], start_index)
        rest_fragments = offset_fragments(fragments[index+1:], new_start)

        start_fragment_texts = [get_fragment(start_text, f) for f in start_fragments]

        for t in start_fragment_texts:
            if t not in fragment_texts:
                found = False
                for ft in fragment_texts:
                    if ft.startswith(t):
                        found = True
                        print(f"+++ MULTI UNTIL Output fragment {[t]} is truncated \n--- from {[ft]}")
                        break
                if not found:
                    print(f"=== MULTI UNTIL Got output level fragment text {str([t])} not in input fragments ", fragment_texts,
                          ' ** total output is ', start_fragment_texts)
                    print("All original ", len(fragments), "fragments: ", fragments) # , "original text: ", article)


        return start_text, start_fragments, rest_text, rest_fragments, new_start


def preceding_non_labelled_sentences(id: str, article: str, fragments: list):
    if fragments == []:
        return [], 0
    else:
        # fragments are sorted by start field, get the first. Fragment is a (start,end,p_type) tuple
        first_index = fragments[0].start
        if first_index > 0:
            substr = article[:first_index]
            if substr.find('\n') > 0:
                r = substr.rindex('\n')
                # Note: here we remove any newlines from after sentences,
                # where elsewhere they are left.
                # Reason for leaving them in is that this way fragment indices can be reconstructed.
                sentences = split_in_sentences(substr[:r+1]) #[x for x in substr[:r].split('\n') if x != '']
                return sentences, r+1
            else:
                return [], 0
        else:
            return [], 0


def merge_short_sentences(lst: list) -> list:
    """Merge subsequent sentences that consist of only a single word."""
    last_is_short = False
    out_lst = []
    for s in lst:
        if s.find(' ') == -1:
            if last_is_short:
                out_lst[-1] = out_lst[-1] + s
            else:
                last_is_short = True
                out_lst.append(s)
        else:
            last_is_short = False
            out_lst.append(s)
    return out_lst


def split_in_sentences(s):
    """Split into sentences, without loosing newlines. Original newlines remain after each sentence"""
    if s == '':
        return []
    elif s.find('\n') == -1:
        return [s]
    else:
        if s[-1] == '\n':
            i = len(s) -1
            while s[i] == '\n':
                i -= 1
            if not s[:i].find('\n'):
                # Only closing newlines, return s as one sentence
                return [s]
            else:
                j = s[:i].rfind('\n')
                lst = split_in_sentences(s[:j+1])
                rest = s[j+1:]
                lst.append(rest)
        else:
            j = s.rfind('\n')
            lst = split_in_sentences(s[:j + 1])
            rest = s[j + 1:]
            lst.append(rest)
        return lst


def fragment_from_text(article, match, p_type=None):
    """Find a fragment in an article and return its (start, end, label) definition"""
    def _fragment_from_text(article, text_fragment, p_type=None):
        n = len(text_fragment)
        # For searching text, transform all to lowercase
        start = article.lower().find(text_fragment.lower())
        if start == -1:
            return None
        else:
            return (start, start+n, p_type, text_fragment)

    if type(match) == str:
        return _fragment_from_text(article, match, p_type)
    else:
        for t in match:
            res = _fragment_from_text(article, t, p_type, t)
            if res != None:
                # We return the first match
                return res


def surrounding_word(s: str, index: int, with_line_end=False):
    """Find the start and end index of the word around the current index.

    :param s String in which we seek word
    :param index Current index within the string, identifying a position within the sought word
    :param with_line_end If set, a trailing line end character will be included with the word

    Returns: (start, end) of word identified by 'index' or None if index is not a position in a word."""
    is_word = re.compile('\w')
    #print(f' s=[{s}], index={index}')
    if index >= len(s):
        index = len(s) - 1

    if is_word.match(s[index]) is None:
        return None
    else:
        start = index
        end = index
        # 'start' is inclusive, that is, index of part of the word
        while start - 1 > 0 and is_word.match(s[start - 1]):
            start -= 1
        # 'end' is exclusive, that is, its index is NOT part of the word
        while end < len(s) and (is_word.match(s[end]) or (with_line_end and s[end] in ['.', '?', '!'])):
            end += 1
        return start, end


def next_token(s, index=0):
    tok = re.compile('^[^a-zA-Z0-9\[]*(\[[io]-\d+\]|\w+)\W.*', re.MULTILINE | re.DOTALL | re.IGNORECASE)
    m = tok.match(s[index:] + ' ')  # Add a space to enforce a match with \W at the end of the string
    if m is None:
        # no next token
        return None
    else:
        return m.group(1)


def tokenize_string(s):
    lst = []
    index = 0
    while index < len(s):
        found_next = next_token(s, index)
        if found_next is None:
            break
        else:
            print(index, found_next)
            lst.append(found_next)
            index = s.find(found_next, index) + len(found_next)
    return lst


def spacy_tokenize(s):
    """Tokenize a string using spacy. Turn symbol-fragmented tags into single tokens"""
    io_dd_regex = re.compile(r'^[io]-\d+$')
    doc = [str(x) for x in nlp(s)]
    output = []
    offset = 0
    index = 0
    while index < len(doc):
        if doc[index] == '[':
            if index + 2 < len(doc) and \
                    io_dd_regex.match(doc[index + 1]) and \
                    doc[offset + index + 2] == ']':
                output.append('['+doc[index + 1] + ']')
                index += 2  # we took out two extra symbols from the list
            else:
                output.append(doc[index])
        else:
            output.append(doc[index])
        index += 1
    return output

def look_back():
    pass

def look_ahead():
    pass


def insert_tags_list(tagged_text, orig_text):
    """Insert the tags from the model prediction into the original text

    Model prediction generates noise in texts, like word changes and loose characters."""
    print("orig text = ", orig_text)
    print("tagged text = ", tagged_text)
    tagged = spacy_tokenize(tagged_text)
    orig = spacy_tokenize(orig_text)

    before = None
    after = None
    insert_index = 0
    for index, word in enumerate(tagged):
        if is_tag(word):
            minus = 1
            while index - minus + 1 > 0 and is_tag(tagged[index - minus]):
                minus += 1

            if index - minus == -1:
                before = None
            else:
                before = tagged[index - minus]

            add = 1
            while index + add < len(tagged) and is_tag(tagged[index + add]):
                add += 1
            if index + add == len(tagged):
                after = None
            else:
                after = tagged[index + add]

            if before is None:
                #print(f"Nothing before {word} at index={index}, place it at beginning of orig")
                orig = [word] + orig
            elif after is None:
                #print(f"Nothing after {word} at index={index}, place it at end of orig")
                orig = orig + [word]
            else:
                # Potentially extend the span
                inc = 0
                while insert_index + inc + 1 < len(orig) and orig[insert_index + inc] != after:
                    inc += 1

                if orig[insert_index + inc] == after:
                    #print(f"Inserting {word}, before={before} and after={after}")
                    orig = orig[:insert_index + inc] + [word] + orig[insert_index + inc:]
                    insert_index += inc + 1

                else:
                    # We didn't find the following token
                    print(f"Didn't find following token for {word}, searching for further tokens")
                    # We set the increase to 1, so, no option to extend the span
                    inc = 0
                    # Now we look further ahead to see if there is a match
                    max_look_ahead = 4
                    look_ahead = 1
                    ignore = 0
                    while look_ahead < max_look_ahead and insert_index + inc + look_ahead < len(orig) and \
                            insert_index + inc + look_ahead + ignore < len(tagged):
                        orig_ahead_word = orig[insert_index + inc + look_ahead]
                        while is_tag(tagged[insert_index + inc + look_ahead + ignore]) and \
                                insert_index + inc + look_ahead + ignore < len(tagged):
                            ignore += 1
                        tagged_ahead_word = tagged[insert_index + inc + look_ahead + ignore]

                        if orig_ahead_word == tagged_ahead_word:
                            # Instead of 'after' matching, we find a match several words latter
                            # and treat that now as if 'after' had matched
                            print(f'Look ahead ({look_ahead}) matched: {orig[insert_index + look_ahead]}')
                            orig = orig[:insert_index + inc] + [word] + orig[insert_index + inc:]
                            print('orig = ', orig)
                            insert_index += inc + 1
                            break

                        look_ahead += 1

        else:
            # do nothing with non-tags, move on to the next token
            pass

    return orig


def insert_tags(orig_text: str, token_list: list) -> str:
    """insert tokens into actual text, respecting the original whitespace."""
    last_index = 0
    for token in token_list:
        # Assume all tokens are literally in the original string
        if not is_tag(token):
            try:
                n = orig_text.index(token, last_index)
                #print(f"Found '{token}' at {n} in '{orig_text}' with last_index={last_index}")
                last_index = n + len(token)
                # Skip whitespace
                #while last_index + 1 < len(orig_text) and orig_text[last_index + 1] in [' ']:
                #    last_index += 1
            except ValueError:
                print(f"Strange, token {token} not found in {orig_text[last_index:]}")
        else:
            extended_token = ' ' + token + ' '
            orig_text = orig_text[:last_index] + extended_token + orig_text[last_index:]
            #print(f"last index updated for '{token}' from {last_index} to {last_index + len(extended_token)}")
            last_index += len(extended_token)
    return orig_text


def calibrate(fragment: Fragment, match_text: str, orig_text:str, distance=3) -> Fragment:
    """Calibrates a fragment to another text, assuming it to be nearly right.

    Effectively, it moves a fragment a short distance to better match the original text"""
    fragment_text = fragment.extract(match_text).lower()

    print(f'INPUT FRAGMENT = [{fragment_text}]')

    first_word_re = re.compile('\A\W*(\w+)\W.*', re.MULTILINE | re.DOTALL)
    # Include trailing end of line interpunction
    last_word_re = re.compile('.*\W(\w+[!.?]?)\W*\Z', re.MULTILINE | re.DOTALL)

    m = first_word_re.match(fragment_text+' ')
    assert m is not None, f"First word matching failed for [{fragment_text} ] for {fragment}"
    first_word = m.group(1)

    # Deal with aberrant single letter words, skip them unless they are 'i' or 'a'
    # Commented out: this had a negative effect
    if len(first_word) == 1 and first_word not in ['i', 'a']:
        fragment.start += 1
        fragment_text = fragment_text[1:]
        m = first_word_re.match(fragment_text+' ')
        assert m is not None, f"First word matching failed (b) for [{fragment_text} ] for {fragment}"
        first_word = m.group(1)

    n = last_word_re.match(' ' + fragment_text)
    assert n is not None, f"Last word matching failed for [ {fragment_text}] for {fragment}"
    last_word = n.group(1)

    have_set_first_word = False
    have_set_last_word = False

    start, end = fragment.start, fragment.end
    startpos = max(start - distance, 0)
    for i in range(startpos, start + distance):
        if orig_text[i:].lower().startswith(first_word):
            start = i
            have_set_first_word = True
            break

    endpos = min(end + distance, len(orig_text))
    for i in range(end - distance, endpos):
        if orig_text[:i].lower().endswith(last_word):
            end = i
            have_set_last_word = True
            break

    if not have_set_first_word:
        res = surrounding_word(orig_text, start)
        if res is None:
            print("starting in empty space")
        else:
            start = res[0]
    if not have_set_last_word:
        res = surrounding_word(orig_text, end, with_line_end=True)
        if res is None:
            print("ending in empty space")
        else:
            end = res[1]

    fragment.start = start
    fragment.end = end

    print(f'OUTPUT FRAGMENT = [{fragment.extract(orig_text)}]')

    return fragment



def encode(article: str, fragments: list, all_labels: list, dropout=0.0, random_mask=0.0) -> str:
    """Indicate fragments in an article through Begin and End identifiers per label

    :param dropout Probability of leaving out labels
    """
    label_types = list({f.label for f in fragments})
    label_dict = {l: [None] * len(article) for l in label_types}
    symbols = {l: label_to_symbol(l, all_labels) for l in label_types}
    _IN_, _OUT_ = 0, 1

    for f in fragments:
        label_dict[f.label][f.start] = symbols[f.label][_IN_]
        label_dict[f.label][f.end - 1] = symbols[f.label][_OUT_]

    # We now have a matrix with the article in the first column and then columns for each label
    # Let's put it together into a single list

    lst = []
    for i in range(len(article)):
        # First we write IN symbols
        for l in label_types:
            sym = label_dict[l][i]
            guess = random.uniform(0, 1)
            if sym is not None and sym.startswith('[i-'):
                if guess >= (1 - random_mask):
                    lst.extend([' <mask> '])
                else:
                    lst.extend([' ', sym, ' '])
        # And last we write the OUT symbols
        lst.append(article[i])
        for l in label_types:
            sym = label_dict[l][i]
            guess = random.uniform(0, 1)
            if sym is not None and sym.startswith('[o-'):
                if guess >= (1 - random_mask):
                    lst.extend([' <mask> '])
                else:
                    lst.extend([' ', sym, ' '])

    s = "".join(lst)
    return s


def decode(s: str, all_labels: list) -> tuple:
    """Convert a string with Begin and End markers into an article and a list of fragments."""
    article = []  # article is built up character by character
    label_dict = {label_index: None for label_index in range(len(all_labels))}
    re_start = "^\[i-(\d+)\]\s?"
    re_end = "^\s?\[o-(\d+)\]"
    index = 0
    fragments = []
    errors = 0  # number of errors during processing
    while index < len(s):
        m = re.match(re_start, s[index:])
        n = re.match(re_end, s[index:])
        if m is not None:
            # Set start marker for the label to the current length of the article
            label_index = int(m.group(1))
            if label_dict[label_index] is not None:
                print(f"Ignoring unclosed open marker to open new for '{all_labels[label_index]}'/{label_index} in\n(({s}))")
                errors += 1
            label_dict[label_index] = len(article)
            index += len(m.group(0))
        elif n is not None:
            label_index = int(n.group(1))
            cur_start = label_dict[label_index]
            if cur_start is None:
                # We found an end marker for which there is no start marker
                print(f"Ignoring end marker without start marker for '{all_labels[label_index]}'/{label_index} in\n"
                      f"(({s}))")
                errors += 1
            else:
                cur_end = len(article)
                while cur_start < cur_end and article[cur_start] in [' ', '\n', '\t']:
                    cur_start += 1
                if cur_end > cur_start:
                    f = Fragment(cur_start, cur_end, all_labels[label_index])
                    fragments.append(f)
                label_dict[label_index] = None
            index += len(n.group(0))
        else:
            # Regular character. We can just add, but then we have to do bookkeeping
            article.append(s[index])
            index += 1

    # Check if there are loose ends, that is, open markers without closure
    for l_index in range(len(all_labels)):
        if label_dict[l_index] is not None:
            print(f"Ignoring unclosed marker for label '{all_labels[l_index]}'/{l_index} in\n(({s}))")
            errors += 1

    return ''.join(article), fragments, errors

def find_all_labels_matching_or_not(s, all_labels):
    """Find all tags beginning or ending a span, whether they have a matching tag or not."""
    labels = []
    match_tag_re = re.compile(r"\[[io]-(\d+)\]")
    m = match_tag_re.match(s)
    if m is not None:
        found_tag = m.group(0)
        label_id = int(m.group(1))
        labels.append(all_labels[label_id])
        shifted_index = s.find(found_tag) + len(found_tag)
        labels.extend(find_all_labels_matching_or_not(s[shifted_index:], all_labels))
    return labels
