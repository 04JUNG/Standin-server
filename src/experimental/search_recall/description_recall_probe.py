"""Compare Korean phrase retrieval across the same whole-pose library."""

import re
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

def grams(value):
    return Counter(word[i:i+n] for word in re.findall(r'[가-힣a-z0-9]+', value.lower())
                   for n in (2, 3) for i in range(len(word)-n+1))


class DescriptionIndex:
    def __init__(self, records):
        self.ids = tuple(row['pose_id'] for row in records)
        docs = [grams(' '.join([row.get('description', {}).get('text', ''),
                               *(tag.get('label', '') for group in row.get('groups', {}).values()
                                 for tag in group if tag['field'] not in ('intent', 'emotion'))]))
                for row in records]
        frequency = Counter(term for doc in docs for term in doc)
        self.terms = {term: i for i, term in enumerate(term for term, count in frequency.items()
                                                       if 2 <= count <= len(docs) * .85)}
        self.idf = np.array([np.log1p((len(docs) - frequency[t] + .5) / (frequency[t] + .5))
                             for t in self.terms], dtype=np.float32)
        rows, columns, values = [], [], []
        for i, doc in enumerate(docs):
            for term, count in doc.items():
                if (j := self.terms.get(term)) is not None:
                    rows.append(i)
                    columns.append(j)
                    values.append((1 + np.log(count)) * self.idf[j])
        matrix = csr_matrix((values, (rows, columns)), shape=(len(docs), len(self.terms)), dtype=np.float32)
        norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
        self.matrix = matrix.multiply(1 / norms.clip(1e-6)[:, None]).tocsr()

    def search(self, text):
        query = np.zeros(len(self.terms), dtype=np.float32)
        for term, count in grams(text).items():
            if (j := self.terms.get(term)) is not None:
                query[j] = (1 + np.log(count)) * self.idf[j]
        norm = np.linalg.norm(query)
        return (self.matrix @ query) / norm if norm > 0 else None
