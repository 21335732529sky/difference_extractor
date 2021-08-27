import pandas as pd
import random
import json
import os
import torch
from collections import defaultdict, namedtuple, Counter
from sklearn.model_selection import train_test_split

SETTING = namedtuple("SETTING", ['N', 'K', 'Q'])


def fix_idx(idx, txt):
    bl = idx;
    while txt[bl] != ' ':
        bl -= 1
        if bl <= 0:
            bl = 0;
            break
    br = idx
    while txt[br] != ' ':
        br += 1
        if br >= len(txt):
            break
    if abs(idx - bl) < abs(idx - br):
        idx = bl
    else:
        idx = br

    return idx

class FewshotDataset:
    def __init__(self,
                 data_dir,
                 data_size='all',
                 label_split=None):
        self.texts, self.labels, self.splits, self.additional_data = self.read_data(data_dir)
        self.labels = [list(l) if isinstance(l, str) else l for l in self.labels]
        self.groups = self.make_group(self.labels)
        self.all_labels = sorted(list(self.groups.keys()))
        self.data_size = data_size
        self.setting = None
        self.freeze = False
        self.multi_label = False

        self.split_labels(label_split)
        if data_size == 'critical':
            self.drop_data(num_remain=5)

    def read_data(self, data_dir):
        raise NotImplementedError()

    def make_additional_info(self, idx_support, idx_query):
        raise NotImplementedError()

    def make_group(self, labels):
        group = defaultdict(list)
        labels_tmp = [list(l) if isinstance(l, str) else l for l in labels]
        for i, l in enumerate(labels_tmp):
            for j in l:
                group[j].append(i)

        return group

    def split_labels(self, label_split):
        min_samples = self.setting.K + 1 if self.setting is not None else 3
        self.train_labels, self.test_labels = train_test_split(self.all_labels,
                                                               test_size=0.6,
                                                               shuffle=True,
                                                               random_state=42)
        self.test_labels, self.dev_labels = train_test_split(self.test_labels,
                                                             test_size=0.5,
                                                             shuffle=True,
                                                             random_state=42)
        self.train_labels = [l for l in self.train_labels if len(self.groups[l]) > min_samples]
        self.dev_labels = [l for l in self.dev_labels if len(self.groups[l]) > min_samples]
        self.test_labels = [l for l in self.test_labels if len(self.groups[l]) > min_samples]
        print(len(self.train_labels), len(self.dev_labels), len(self.test_labels), flush=True)

    def drop_data(self, num_remain=5):
        for key in self.groups.keys():
            random.shuffle(self.groups[key])
            self.groups[key] = self.groups[key][:num_remain]

    def set_task_setting(self, N=2, K=2, Q=2):
        self.setting = SETTING(N=N, K=K, Q=Q)
        if self.task_type == 'all':
            self.split_labels(self.task_type, None)
        self.setting = SETTING(N=min(N, len(self.train_labels)), K=K, Q=Q)

    def set_query_size(self, Q):
        self.setting = SETTING(N=self.setting.N,
                               K=self.setting.K,
                               Q=Q)

    def make_task(self, mode='train', pivot=None):
        if mode == 'train':
            pool = self.train_labels
        elif mode == 'dev':
            pool = self.dev_labels
        elif mode == 'test':
            pool = self.test_labels
        classes_support = random.sample(pool, k=min(self.setting.N, len(pool)))
        if pivot is not None:
            tar_l, tar_i = pivot
            classes_support[0] = tar_l

        idx_all = [random.sample(self.groups[l], k=min(len(self.groups[l]), self.setting.K + self.setting.Q)) for l in
                   classes_support]
        idx_support = [([d for d in self.groups[l] if len(self.labels[d]) == 1])[:self.setting.K] for l in
                       classes_support]
        ch = [i for i, lst in enumerate(idx_support) if len(lst) > 0]
        idx_all = [idx_all[i] for i in ch]
        idx_support = [idx_support[i] for i in ch]
        classes_support = [classes_support[i] for i in ch]
        while not all([len(self.labels[i[0]]) == 1 for i in idx_support]):
            idx_all = [random.sample(self.groups[l], k=min(len(self.groups[l]), self.setting.K + self.setting.Q)) for l
                       in classes_support]
            idx_support = [e[:self.setting.K] for e in idx_all]

        if self.setting.Q > self.setting.N:
            idx_query = sum([e[self.setting.K:self.setting.K + self.setting.Q // self.setting.N] for e in idx_all], [])
        else:
            idx_query = sum([e[self.setting.K:self.setting.K + 1] for e in idx_all], [])
            random.shuffle(idx_query)
            idx_query = idx_query[:self.setting.Q]
        if pivot is not None:
            idx_support[0][0] = tar_i
        random.shuffle(idx_query)
        idx_query = idx_query[:self.setting.Q]
        if self.freeze:
            idx_query = [self.groups[l][0] for l in self.groups.keys()]
        support_set = [[self.texts[i] for i in j] for j in idx_support]
        query_set = [self.texts[i] for i in idx_query]
        if self.freeze:
            labels = [0, ] * len(idx_query)
        else:
            labels = [[classes_support.index(l) for l in self.labels[i] if l in classes_support] for i in idx_query]
        return support_set, query_set, labels, idx_support, idx_query

    def get_train_iterator(self, max_iter=10000):
        if self.setting is None:
            raise RuntimeError('You must specify the meta-training setting.')
        c = 0
        while c < max_iter:
            support_set, query_set, labels, idxs, idxq = self.make_task(mode='train')
            additional = self.make_additional_info(idxs, idxq)
            yield support_set, query_set, labels, additional
            c += 1

    def get_eval_iterator(self, max_iter=10000, mode='dev', pivot=None):
        if self.setting is None:
            raise RuntimeError('You must specify the meta-training setting.')
        c = 0
        if self.task_type == 'all':
            max_iter = len(self.splits[mode])
        while c < max_iter:
            if self.task_type == 'disjoint':
                support_set, query_set, labels, idxs, idxq = self.make_task(mode=mode, pivot=pivot)
                additional = self.make_additional_info(idxs, idxq)
            elif self.task_type == 'all':
                if self.setting.Q >= len(self.all_labels):
                    idxs = [random.sample(self.groups[l], self.setting.K) for l in self.all_labels]
                    idxq = self.splits[mode][c:c + 5]
                    support_set = [[self.texts[i] for i in j] for j in idxs]
                    query_set = [self.texts[i] for i in idxq]
                    labels = [[self.all_labels.index(a) for a in self.labels[i]] for i in idxq]
                    additional = self.make_additional_info(idxs, idxq)
                    c += 4
                else:
                    idxq = self.splits[mode][c:c + 5]
                    tar_labels = [self.labels[i] for i in idxq]
                    if self.setting.N >= len(self.all_labels):
                        tar_labels = self.all_labels
                        idxs = [random.sample(self.groups[l], self.setting.K) for l in tar_labels]
                    elif len(set(sum(tar_labels, []))) <= self.setting.N:
                        tar_labels = set(sum(tar_labels, []))
                        rem_set = set(self.all_labels) - tar_labels
                        rem_set = list(rem_set)
                        while len(tar_labels) < self.setting.N:
                            label = random.choice(rem_set)
                            tar_labels.add(label)
                            rem_set.remove(label)
                        tar_labels = list(tar_labels)
                        idxs = [random.sample(self.groups[l], self.setting.K) for l in tar_labels]
                    else:
                        tar_labels_b = set([tl[0] for tl in tar_labels])
                        tar_labels = sum([tl[1:] for tl in tar_labels], [])
                        tar_labels = list(set(tar_labels))
                        while len(tar_labels_b) < self.setting.N:
                            idx = random.randint(0, len(tar_labels) - 1)
                            tar_labels_b.add(tar_labels[idx])
                            tar_labels.remove(tar_labels[idx])
                        tar_labels = list(tar_labels_b)
                        idxs = [random.sample(self.groups[l], self.setting.K) for l in tar_labels]
                    support_set = [[self.texts[i] for i in j] for j in idxs]
                    query_set = [self.texts[i] for i in idxq]
                    labels = [[tar_labels.index(a) for a in self.labels[i] if a in tar_labels] for i in idxq]
                    additional = self.make_additional_info(idxs, idxq)
                    c += 4

            yield support_set, query_set, labels, additional
            c += 1


class HuffpostDataset_ICLR(FewshotDataset):
    def read_data(self, data_dir, add_span_tags=False):
        with open(os.path.join(data_dir, 'huffpost.json')) as f:
            dataset = [json.loads(line.strip()) for line in f]
        index = list(range(len(dataset)))
        texts = [' '.join(d['text']) for d in dataset]
        if add_span_tags:
            texts = ['[F]' + texts + '[/F]']
        labels = [d['label'] for d in dataset]
        splits = {}
        train_idx, test_idx = train_test_split(index,
                                               random_state=42,
                                               shuffle=True,
                                               test_size=0.2,
                                               stratify=labels)
        dev_idx, test_idx = train_test_split(test_idx,
                                             random_state=42,
                                             shuffle=True,
                                             test_size=0.5,
                                             stratify=[labels[i] for i in test_idx])
        splits['train'] = train_idx
        splits['dev'] = dev_idx
        splits['test'] = test_idx

        return texts, labels, splits, list()

    def make_additional_info(self, idxs, idxq):
        return {'support_labels': [self.labels[i[0]] for i in idxs],
                'query_labels': [self.labels[i] for i in idxq]}

    def split_labels(self, task_type, label_split):
        if task_type == 'all':
            min_samples = self.setting.K + 1 if self.setting is not None else 3
            self.train_labels = [l for l in self.all_labels if len(self.groups[l]) > min_samples]
            self.dev_labels = list(self.all_labels)
            self.test_labels = list(self.all_labels)
        elif task_type == 'disjoint':
            self.train_labels = list(range(0, 20))
            self.dev_labels = list(range(20, 25))
            self.test_labels = list(range(25, 41))


class FewRelDataset_ICLR(FewshotDataset):
    def __init__(self,
                 data_dir,
                 task_type='disjoint',
                 data_size='all',
                 label_split=None,
                 use_loc_info=False):
        super(FewRelDataset_ICLR, self).__init__(
            data_dir,
            task_type=task_type,
            data_size=data_size,
            label_split=label_split)
        self.use_loc_info = use_loc_info

    def read_data(self, data_dir):
        with open(os.path.join(data_dir, 'fewrel_bert_uncase.json')) as f:
            dataset = [json.loads(line.strip()) for line in f]
        index = list(range(len(dataset)))
        texts = []
        for d in dataset:
            s = []
            if min(d['head']) < min(d['tail']):
                s += d['text'][:d['head'][0]]
                s += ['[H]']
                s += d['text'][d['head'][0]:d['head'][1] + 1]
                s += ['[/H]']
                s += d['text'][d['head'][1] + 1:d['tail'][0]]
                s += ['[T]']
                s += d['text'][d['tail'][0]:d['tail'][1] + 1]
                s += ['[/T]']
                s += d['text'][d['tail'][1] + 1:]
            else:
                s += d['text'][:d['tail'][0]]
                s += ['[T]']
                s += d['text'][d['tail'][0]:d['tail'][1] + 1]
                s += ['[/T]']
                s += d['text'][d['tail'][1] + 1:d['head'][0]]
                s += ['[H]']
                s += d['text'][d['head'][0]:d['head'][1] + 1]
                s += ['[/H]']
                s += d['text'][d['head'][1] + 1:]
            texts.append(' '.join(s))
        labels = [d['label'] for d in dataset]
        loc = [(d['head'], d['tail']) for d in dataset]
        splits = {}
        train_idx, test_idx = train_test_split(index,
                                               random_state=42,
                                               shuffle=True,
                                               test_size=0.2,
                                               stratify=labels)
        dev_idx, test_idx = train_test_split(test_idx,
                                             random_state=42,
                                             shuffle=True,
                                             test_size=0.5,
                                             stratify=[labels[i] for i in test_idx])
        splits['train'] = train_idx
        splits['dev'] = dev_idx
        splits['test'] = test_idx

        return texts, labels, splits, loc

    def make_additional_info(self, idxs, idxq):
        ret = dict()
        ret['head_sup'] = torch.cuda.LongTensor([[self.additional_data[i][0] for i in idx] for idx in idxs])
        ret['tail_sup'] = torch.cuda.LongTensor([[self.additional_data[i][1] for i in idx] for idx in idxs])
        ret['head_que'] = torch.cuda.LongTensor([self.additional_data[i][0] for i in idxq])
        ret['tail_que'] = torch.cuda.LongTensor([self.additional_data[i][1] for i in idxq])

        return ret

    def split_labels(self, task_type, label_split):
        if task_type == 'all':
            min_samples = self.setting.K + 1 if self.setting is not None else 3
            self.train_labels = [l for l in self.all_labels if len(self.groups[l]) > min_samples]
            self.dev_labels = list(self.all_labels)
            self.test_labels = list(self.all_labels)
        elif task_type == 'disjoint':
            self.train_labels = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                                 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                                 59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                 76, 77, 78]
            self.dev_labels = [7, 9, 17, 18, 20]
            self.test_labels = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]


class FewRelDataset(FewRelDataset_ICLR):
    def split_labels(self, task_type, label_split):
        if task_type == 'all':
            min_samples = self.setting.K + 1 if self.setting is not None else 3
            self.train_labels = [l for l in self.all_labels if len(self.groups[l]) > min_samples]
            self.dev_labels = list(self.all_labels)
            self.test_labels = list(self.all_labels)
        elif task_type == 'disjoint':
            all_labels = list(self.groups.keys())
            self.train_labels, self.test_labels = train_test_split(all_labels,
                                                                   random_state=42,
                                                                   shuffle=True,
                                                                   test_size=.4)
            self.dev_labels, self.test_labels = train_test_split(self.test_labels,
                                                                 random_state=42,
                                                                 shuffle=True,
                                                                 test_size=.5)


class HuffpostDatset(HuffpostDataset_ICLR):
    def split_labels(self, task_type, label_split):
        if task_type == 'all':
            min_samples = self.setting.K + 1 if self.setting is not None else 3
            self.train_labels = [l for l in self.all_labels if len(self.groups[l]) > min_samples]
            self.dev_labels = list(self.all_labels)
            self.test_labels = list(self.all_labels)
        elif task_type == 'disjoint':
            all_labels = list(self.groups.keys())
            self.train_labels, self.test_labels = train_test_split(all_labels,
                                                                   random_state=42,
                                                                   shuffle=True,
                                                                   test_size=.4)
            self.dev_labels, self.test_labels = train_test_split(self.test_labels,
                                                                 random_state=42,
                                                                 shuffle=True,
                                                                 test_size=.5)


if __name__ == '__main__':
    a = FewRelDataset_ICLR('data')
