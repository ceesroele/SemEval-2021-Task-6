"""

"""
from dataclasses import dataclass

@dataclass
class Fragment:
    start: int
    end: int
    label: str

    def __gt__(self, other):
        return self.start > other.start

    def __lt__(self, other):
        return self.start < other.start

    def __ge__(self, other):
        return self.start >= other.start

    def __le__(self, other):
        return self.start <= other.start

    def __len__(self):
        return self.end - self.start

    def __add__(self, n: int):
        return Fragment(self.start + n, self.end + n, self.label)

    def __sub__(self, n: int):
        return Fragment(self.start - n, self.end - n, self.label)

    def __and__(self, other):
        if self.start < other.start:
            a, b = self, other
        else:
            a, b = other, self
        if b.start >= a.end:
            return None
        else:
            start = b.start
            end = min(a.end, b.end)
            #intersection_start = b.start - a.start
            #intersection_end = end - a.start
            #txt = a.text[intersection_start:intersection_end]
            return Fragment(start, end, f'{a.label},{b.label}')

    def __or__(self, other):
        if self.start < other.start:
            a, b = self, other
        else:
            a, b = other, self
        if b.start >= a.end:
            start = a.start
            end = max(a.end, b.end)
            #text = a.text + '*' * (b.start - a.end) + b.text
            labels = f'{a.label},{b.label}'
        else:
            start = a.start
            end = max(a.end, b.end)
            #b_start = a.start + len(a.text) - b.start
            #text = a.text + b.text[b_start:]
            labels = f'{a.label},{b.label}'
        return Fragment(start, end, labels)

    def extract(self, article):
        """Slice the text string for this fragment out of the article"""
        return article[self.start:self.end]


if __name__ == '__main__':
    f = Fragment(20, 38, 'ha!')
    g = f + 10
    h = f - 10

    print('intersection: f & g', f & g)

    print(f)
    f -= 5
    print(f)

    print(len(f))
    lst = [f, g, h]
    print(lst)
    print(sorted(lst))

    print('conjunction: f | h', f | h)
