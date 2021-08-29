import torch
import random
import itertools
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
import time


def kl_div(param1, param2):
    mean1, cov1 = param1
    mean2, cov2 = param2
    bsz, seqlen, tag_dim = mean1.shape
    var_len = tag_dim * seqlen

    cov2_inv = 1 / cov2
    mean_diff = mean1 - mean2

    mean_diff = mean_diff.view(bsz, -1)
    cov1 = cov1.view(bsz, -1)
    cov2 = cov2.view(bsz, -1)
    cov2_inv = cov2_inv.view(bsz, -1)

    temp = (mean_diff * cov2_inv).view(bsz, 1, -1)
    KL = 0.5 * (torch.sum(torch.log(cov2), dim=1) - torch.sum(torch.log(cov1), dim=1) - var_len
                + torch.sum(cov2_inv * cov1, dim=1) + torch.bmm(temp, mean_diff.view(bsz, -1, 1)).view(bsz))
    return KL


def preprocess(support_set,
               query_set,
               labels,
               additional,
               tokenizer):
    input_args = {}

    input_args['support_inputs'] = [
        torch.tensor([lst[:128] for lst in tokenizer(sp, padding='max_length', max_length=128)['input_ids']]).type(
            torch.cuda.LongTensor) for sp in support_set]
    input_args['support_mask'] = [(e != 0).type(torch.cuda.LongTensor) for e in input_args['support_inputs']]
    input_args['query_inputs'] = torch.tensor(
        [lst[:128] for lst in tokenizer(query_set, padding='max_length', max_length=128)['input_ids']]).type(
        torch.cuda.LongTensor)
    input_args['query_mask'] = (input_args['query_inputs'] != 0).type(torch.cuda.LongTensor)

    input_args['labels'] = torch.tensor(labels).type(torch.cuda.LongTensor)
    if len(additional) != 0 and len(additional) == 4:
        input_args['head_sup'] = torch.cuda.LongTensor(additional['head_sup'])
        input_args['tail_sup'] = torch.cuda.LongTensor(additional['tail_sup'])
        input_args['head_que'] = torch.cuda.LongTensor(additional['head_que'])
        input_args['tail_que'] = torch.cuda.LongTensor(additional['tail_que'])
    return input_args


class CompressedDist(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim):
        super(CompressedDist, self).__init__()
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(input_dim, (input_dim + output_dim) // 2),
            torch.nn.ReLU(),
            torch.nn.Linear((input_dim + output_dim) // 2, output_dim),
            torch.nn.ReLU())
        '''
        for param in self.hidden.parameters():
            torch.nn.init.normal_(param, mean=.0, std=.1)
        '''
        self.mean_vec = torch.nn.Linear(output_dim, output_dim)
        self.std_vec = torch.nn.Linear(output_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, reps, n_samples=1):
        hidden = self.hidden(reps)
        mean = self.mean_vec(hidden)
        std = self.std_vec(hidden) ** 2 + 1e-8
        noise = torch.randn(n_samples, *reps.shape[:-1], self.output_dim).type(torch.cuda.FloatTensor)
        samples = std * noise + mean

        return samples.transpose(0, 1), (mean, std)


class DifferenceExtractor(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim):
        super(DifferenceExtractor, self).__init__()
        self.hidden = torch.nn.TransformerEncoderLayer(input_dim, 8, dim_feedforward=input_dim * 4, activation='gelu')
        self.hidden = torch.nn.TransformerEncoder(self.hidden, 1)
        self.hidden = torch.nn.DataParallel(self.hidden)
        self.output_dim = output_dim

    def forward(self, reps, n_samples=1):
        hidden = self.hidden(reps.transpose(0, 1))
        hidden = hidden.transpose(0, 1)

        return hidden


class ProtoNet(torch.nn.Module):
    def __init__(self,
                 bert_type='bert-base-uncased',
                 metric='distance',
                 gamma=1e-2,
                 alpha=1e-6,
                 diff_extractor=False,
                 maml=False):
        super(ProtoNet, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_type)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        print(self.tokenizer)
        null_emb = torch.FloatTensor(np.random.uniform(-0.01, 0.01, (6, 768)))
        null_emb = torch.nn.parameter.Parameter(null_emb)
        emb_cat = torch.nn.parameter.Parameter(
            torch.cat([self.bert.embeddings.word_embeddings.weight, null_emb], dim=0))
        self.bert.embeddings.word_embeddings.weight = emb_cat
        token_dict = {'additional_special_tokens': ['[H]', '[/H]', '[T]', '[/T]', '[F]', '[/F]']}
        self.tokenizer.add_special_tokens(token_dict)

        self.bert = torch.nn.DataParallel(self.bert)
        self.ls = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.bce_loss = torch.nn.BCELoss(reduction='none')
        self.nll = torch.nn.NLLLoss()
        self.relu = torch.nn.ReLU()
        self.alpha = alpha
        self.count = 0
        self.maml = False

        if maml:
            print('[Debug] MAML enabled')
            self.maml = True

        if diff_extractor:
            self.fnn = DifferenceExtractor(768, 768)
            self.fnn = torch.nn.DataParallel(self.fnn)
            self.p_phi = torch.nn.Sequential(
                torch.nn.Linear(768 * 2, 768 * 2 // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(768 * 2 // 4, 1),
                torch.nn.Sigmoid())
            self.p_phi = torch.nn.DataParallel(self.p_phi)

        self.cnn = None

        self.extract_diff = diff_extractor
        self.metric = metric
        self.sw = False
        self.gamma = gamma
        self.scaler_inner = GradScaler()

        self.W_key = torch.nn.Linear(768, 768)
        self.W_query = torch.nn.Linear(768, 768)

    def set_inner_opt(self, lr=1e-5, num_steps=1):
        par = list(self.p_theta.parameters()) + list(self.p_phi.parameters())
        self.inner_opt = AdamW(par, lr=lr * 100)
        self.inner_sch = get_linear_schedule_with_warmup(self.inner_opt,
                                                         num_warmup_steps=int(num_steps * 0.05),
                                                         num_training_steps=num_steps)

    def get_ib_params(self):
        ret = []
        if not self.decompose:
            ret += list(self.fnn.parameters())
        elif self.decompose:
            ret += list(self.decomposer.parameters())
        return ret

    def get_bert_params(self):
        return list(self.bert.parameters())

    def switch(self):
        self.sw = (not self.sw)

    def reset_maml(self):
        params = list(self.bert.parameters())
        with torch.no_grad():
            for p, g in zip(params, self.stored_grads):
                if g is None:
                    continue
                p.add_(g, gamma=self.gamma)
            if hasattr(self, 'fnn'):
                for p, g in zip(self.fnn.parameters(), self.stored_grads_ss):
                    if g is None:
                        continue
                    p.add_(g, gamma=self.gamma)

    def forward(self, support_set, query_set, label, additional):
        kwargs = preprocess(support_set, query_set, label, additional, self.tokenizer)
        N = len(kwargs['support_inputs'])
        K = kwargs['support_inputs'][0].shape[0]
        Q = len(kwargs['query_inputs'])
        labels = kwargs['labels']

        support_set_cat = torch.cat(kwargs['support_inputs'], dim=0)
        support_mask = (support_set_cat != 0).type(torch.LongTensor).cuda()
        query_set = kwargs['query_inputs']
        query_mask = (query_set != 0).type(torch.LongTensor).cuda()
        local_support_rep, support_rep = self.bert(input_ids=support_set_cat, attention_mask=support_mask)
        support_rep = local_support_rep[:, 0, :]
        if not self.maml:
            if self.rel_net:
                local_query_rep, query_rep = self.bert_query(input_ids=query_set, attention_mask=query_mask)
            else:
                local_query_rep, query_rep = self.bert(input_ids=query_set, attention_mask=query_mask)
            query_rep = local_query_rep[:, 0, :]

        instance_reps = support_rep.reshape((N * K, support_rep.shape[-1]))
        support_rep = support_rep.reshape((-1, K, support_rep.shape[-1])).mean(dim=1)
        if self.maml:
            self.meta_w = torch.zeros((N, support_rep.shape[-1]), requires_grad=True).type(torch.cuda.FloatTensor)
            self.meta_b = torch.zeros((N,), requires_grad=True).type(torch.cuda.FloatTensor)
            maml_label = torch.cuda.LongTensor([i // K for i in range(N * K)])
            logits_maml = torch.matmul(self.meta_w, instance_reps.t()).t() + self.meta_b
            loss_maml = self.nll(self.ls(logits_maml), maml_label)
            params = list(self.bert.parameters()) + [self.meta_w, self.meta_b]
            grads = torch.autograd.grad(loss_maml, params, create_graph=True)
            self.stored_grads = grads
            with torch.no_grad():
                for p, g in zip(params, grads):
                    p.add_(g, gamma=-self.gamma)
            local_query_rep, _ = self.bert(input_ids=query_set, attention_mask=query_mask)
            query_rep = local_query_rep[:, 0, :]
            support_rep = self.meta_w

        if self.extract_diff:
            support_rep_ex = support_rep.unsqueeze(0).expand((query_rep.shape[0], -1, -1))
            query_rep_ex = query_rep.unsqueeze(1)
            concated = torch.cat([support_rep_ex, query_rep_ex], dim=1)
            if self.maml:
                maml_support = support_rep.unsqueeze(0).expand((instance_reps.shape[0], -1, -1))
                maml_query = instance_reps.unsqueeze(1)
                concated_maml = torch.cat([maml_support, maml_query], dim=1)
                sampled_maml = self.fnn(concated_maml.detach(), n_samples=1)
                sampled_s_maml = sampled_maml[:, :N, :]
                sampled_q_maml = sampled_maml[:, N, :]
                if abs(self.alpha) > 1e-10:
                    self.mi_upper_bound(sampled_s_maml.unsqueeze(0), sampled_s_maml.unsqueeze(0), backprop_only=True,
                                        den=1)
                    self.step_inner()
                    kl_ss = self.mi_upper_bound(sampled_s_maml.unsqueeze(0), sampled_s_maml.unsqueeze(0))
                else:
                    kl_ss = 0
                logits = torch.matmul(sampled_s_maml, sampled_q_maml.transpose(-2, -1))
                idx_ = torch.cuda.LongTensor(list(range(N * K)))
                logits = logits[idx_, :, idx_] / (768 ** 0.5) + self.meta_b
                loss_maml_ss = self.nll(self.ls(logits), maml_label)
                loss_maml_ss = loss_maml_ss + self.alpha * kl_ss
                grads_ss = torch.autograd.grad(loss_maml_ss, self.fnn.parameters(), create_graph=True,
                                               allow_unused=True)
                self.stored_grads_ss = grads_ss
                with torch.no_grad():
                    for (name, p), g in zip(self.fnn.named_parameters(), grads_ss):
                        p.add_(g, gamma=-self.gamma)

            sampled_all_ng = self.fnn(concated.detach(), n_samples=16)
            sampled_s_ng = sampled_all_ng[:, :support_rep.shape[0], :]

            sampled_all = self.fnn(concated, n_samples=16)
            sampled_s = sampled_all[:, :support_rep.shape[0], :]
            sampled_q = sampled_all[:, support_rep.shape[0], :]

            if abs(self.alpha) > 1e-10:
                self.mi_upper_bound(sampled_s_ng.unsqueeze(0), sampled_s_ng.unsqueeze(0), backprop_only=True, den=1)
                self.step_inner()
                kl_ss = self.mi_upper_bound(sampled_s_ng.unsqueeze(0), sampled_s_ng.unsqueeze(0))
            else:
                kl_ss = 0
            lower_b = 0
            sampled_q = sampled_q.unsqueeze(1).expand((-1, sampled_s.shape[1], -1))
            support_rep = sampled_s
            query_rep = sampled_q

        else:
            support_rep = support_rep.unsqueeze(0)
            query_rep = query_rep.unsqueeze(1)

        if self.maml:
            if not self.extract_diff:
                support_rep = support_rep.squeeze()
                query_rep = query_rep.squeeze()
            else:
                query_rep = query_rep[:, 0, :]
            logits = torch.matmul(support_rep, query_rep.transpose(-2, -1))
            if self.extract_diff:
                idx_ = torch.cuda.LongTensor(list(range(Q)))
                logits = logits[idx_, :, idx_] / (768 ** 0.5) + self.meta_b
            else:
                logits = logits.t() + self.meta_b
        else:
            if self.multi_label:
                logits = ((support_rep * query_rep).sum(dim=-1) / (support_rep.shape[-1] ** 0.5))
            else:
                logits = -((support_rep - query_rep) ** 2).sum(dim=-1) ** 0.5

        if len(labels.shape) > 1:
            labels = labels.reshape((-1,))
        loss = self.nll(self.ls(logits), labels).mean()
        self.writer.add_scalar('Loss/main', loss.detach().cpu().numpy(), self.count)
        self.count += 1
        if self.extract_diff:
            loss += self.alpha * (kl_ss - lower_b)

        return loss

    def step_inner(self):
        torch.nn.utils.clip_grad_norm_(self.p_theta.parameters(), 1)
        self.inner_opt.step()
        self.inner_sch.step()

    def mi_upper_bound(self, sampled_1, sampled_2, backprop_only=False, den=1):
        kls = []
        var_mean, var_std = [0, 1]

        def _upper_bound(vectors, means, stds, vectors_2=None):
            choices = random.choices(list(range(vectors.shape[1])), k=16)
            choices = torch.cuda.LongTensor(choices)
            all_idx = torch.cuda.LongTensor(list(range(vectors.shape[1])))
            idx1_ex = all_idx.repeat_interleave(choices.shape[0])
            idx2_ex = choices.repeat(vectors.shape[1])
            cond_probs = self.normal_log_prob(vectors, means, stds, vectors_2=vectors_2)
            if vectors_2 is not None:
                vectors_2_ex = vectors_2[:, idx2_ex]
            else:
                vectors_2_ex = None
            log_probs = self.normal_log_prob(vectors[:, idx1_ex], means, stds, vectors_2=vectors_2_ex)
            mean_log_probs = log_probs.reshape((-1, vectors.shape[1], choices.shape[0])).mean(dim=-1)

            return (cond_probs - mean_log_probs).mean()

        if backprop_only:
            l_sigma = -self.normal_log_prob(sampled_1.detach(), var_mean, var_std, vectors_2=sampled_2.detach()).mean(
                dim=1).mean() / den
            self.scaler_inner.scale(l_sigma).backward(retain_graph=True)
            return None

        tar = list(itertools.combinations(range(sampled_1.shape[2]), 2))
        random.shuffle(tar)
        tar = tar[:32]

        for i, j in tar:
            s1 = sampled_1[:, :, i, :]
            kls.append(_upper_bound(s1, var_mean, var_std, vectors_2=sampled_2[:, :, j, :]))
        return torch.stack(kls).mean()

    def normal_log_prob(self, v, mean, std, vectors_2=None):
        if vectors_2 is not None:
            cated = torch.cat([v, vectors_2], dim=-1)
            probs = self.p_phi(cated)
            log_probs = torch.log(probs + 1e-6).squeeze(-1)
        else:
            var = std ** 2 + 1e-8
            log_scale = std.log()
            log_probs = (-(((v - mean) ** 2) / (2 * var)) - log_scale - np.log(2 * np.pi)).sum(dim=-1)
        return log_probs

    def calc_logits(self, support_rep, query_rep):
        if self.metric == 'distance':
            logits = -(((support_rep - query_rep) ** 2).sum(dim=-1) ** 0.5)
        elif self.metric == 'dot':
            logits = (support_rep * query_rep).sum(dim=-1) / (float(support_rep.shape[-1]) ** 0.5)
        return logits

    def predict(self, support_set, query_set, additional):
        kwargs = preprocess(support_set, query_set, [], additional, self.tokenizer)
        N = len(kwargs['support_inputs'])
        K = kwargs['support_inputs'][0].shape[0]
        Q = len(kwargs['query_inputs'])

        support_set_cat = torch.cat(kwargs['support_inputs'], dim=0)
        support_mask = (support_set_cat != 0).type(torch.LongTensor).cuda()
        query_set = kwargs['query_inputs']
        query_mask = (query_set != 0).type(torch.LongTensor).cuda()

        local_support_rep, support_rep = self.bert(input_ids=support_set_cat, attention_mask=support_mask)
        support_rep = local_support_rep[:, 0, :]
        if not self.maml:
            if self.rel_net:
                local_query_rep, query_rep = self.bert_query(input_ids=query_set, attention_mask=query_mask)
            else:
                local_query_rep, query_rep = self.bert(input_ids=query_set, attention_mask=query_mask)
            query_rep = local_query_rep[:, 0, :]
        instance_reps = support_rep.reshape((N * K, support_rep.shape[-1]))
        support_rep = support_rep.reshape((-1, K, support_rep.shape[-1])).mean(dim=1)
        if self.case_based:
            all_logits = torch.matmul(query_rep, support_rep.t()) / (support_rep.shape[-1] ** 0.5)
            all_probs = self.softmax(all_logits)
            probs = all_probs.reshape((-1, len(support_set), K)).sum(dim=-1)
        else:
            if self.maml:
                with torch.autograd.enable_grad():
                    self.meta_w = torch.zeros((N, support_rep.shape[-1]), requires_grad=True).type(
                        torch.cuda.FloatTensor)
                    self.meta_b = torch.zeros((N,), requires_grad=True).type(torch.cuda.FloatTensor)
                    maml_label = torch.cuda.LongTensor([i // K for i in range(N * K)])
                    logits_maml = torch.matmul(self.meta_w, instance_reps.t()).t() + self.meta_b
                    loss_maml = self.nll(self.ls(logits_maml), maml_label)
                    params = list(self.bert.parameters()) + [self.meta_w, self.meta_b]
                    grads = torch.autograd.grad(loss_maml, params, create_graph=True, allow_unused=True)
                    self.stored_grads = grads
                for p, g in zip(params, grads):
                    if g is None:
                        continue
                    p.add_(g, gamma=-self.gamma)
                local_query_rep, _ = self.bert(input_ids=query_set, attention_mask=query_mask)
                query_rep = local_query_rep[:, 0, :]
                support_rep = self.meta_w
            if self.extract_diff:
                support_rep_ex = support_rep.unsqueeze(0).expand((query_rep.shape[0], -1, -1))
                query_rep_ex = query_rep.unsqueeze(1)
                concated = torch.cat([support_rep_ex, query_rep_ex], dim=1)
                if self.maml:
                    with torch.autograd.enable_grad():
                        maml_support = support_rep.unsqueeze(0).expand((instance_reps.shape[0], -1, -1))
                        maml_query = instance_reps.unsqueeze(1)
                        concated_maml = torch.cat([maml_support, maml_query], dim=1)
                        sampled_maml = self.fnn(concated_maml.detach(), n_samples=1)
                        sampled_s_maml = sampled_maml[:, :N, :]
                        sampled_q_maml = sampled_maml[:, N, :]
                        if abs(self.alpha) > 1e-10:
                            kl_ss = self.mi_upper_bound(sampled_s_maml.unsqueeze(0), sampled_s_maml.unsqueeze(0))
                        else:
                            kl_ss = 0
                        logits = torch.matmul(sampled_s_maml, sampled_q_maml.transpose(-2, -1))
                        idx_ = torch.cuda.LongTensor(list(range(N * K)))
                        logits = logits[idx_, :, idx_] / (768 ** 0.5) + self.meta_b
                        loss_maml_ss = self.nll(self.ls(logits), maml_label)
                        loss_maml_ss = loss_maml_ss + self.alpha * kl_ss
                    grads_ss = torch.autograd.grad(loss_maml_ss, self.fnn.parameters(), create_graph=True,
                                                   allow_unused=True)
                    self.stored_grads_ss = grads_ss
                    with torch.no_grad():
                        for p, g in zip(self.fnn.parameters(), grads_ss):
                            if g is None:
                                continue
                            p.add_(g, gamma=-self.gamma)
                if concated.shape[0] < 8:
                    Q = concated.shape[0]
                    concated = concated.repeat(2, 1, 1)
                    sampled_all = self.fnn(concated, n_samples=1)
                    sampled_all = sampled_all[:Q]
                else:
                    sampled_all = self.fnn(concated, n_samples=1)
                sampled_s = sampled_all[:, :support_rep.shape[0], :]
                sampled_q = sampled_all[:, support_rep.shape[0], :]
                sampled_q = sampled_q.unsqueeze(1).expand((-1, sampled_s.shape[1], -1))
                support_rep = sampled_s
                query_rep = sampled_q
            else:
                support_rep = support_rep.unsqueeze(0)
                query_rep = query_rep.unsqueeze(1)

            if self.maml:
                if not self.extract_diff:
                    support_rep = support_rep.squeeze()
                    query_rep = query_rep.squeeze()
                else:
                    query_rep = query_rep[:, 0, :]
                logits = torch.matmul(support_rep, query_rep.transpose(-2, -1))
                if self.extract_diff:
                    idx_ = torch.cuda.LongTensor(list(range(Q)))
                    logits = logits[idx_, :, idx_] / (768 ** 0.5) + self.meta_b
                else:
                    logits = logits.t() + self.meta_b
            else:
                logits = -((support_rep - query_rep) ** 2).sum(dim=-1) ** 0.5
            probs = self.softmax(logits)

        return probs, support_rep


class MLMAN(torch.nn.Module):
    def __init__(self,
                 encoder='bert-base-uncased',
                 hidden_dim=100,
                 alpha=0,
                 lambda_=0.5,
                 rerank=False,
                 ns=False,
                 extract_diff=False,
                 add_null=False):
        super(MLMAN, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder)
        null_emb = torch.FloatTensor(np.random.uniform(-0.01, 0.01, (6, 768)))
        null_emb = torch.nn.parameter.Parameter(null_emb)
        emb_cat = torch.nn.parameter.Parameter( \
            torch.cat([self.bert_encoder.embeddings.word_embeddings.weight, null_emb], dim=0))
        self.bert_encoder.embeddings.word_embeddings.weight = emb_cat
        token_dict = {'additional_special_tokens': ['[H]', '[/H]', '[T]', '[/T]', '[F]', '[/F]']}
        self.tokenizer.add_special_tokens(token_dict)

        self.bert_encoder = torch.nn.DataParallel(self.bert_encoder)
        self.softmax = torch.nn.Softmax(dim=-1)


        self.emb_dim = (self.cnn.ebd_dim if self.cnn else 768)
        self.extract_diff = extract_diff

        self.lstm = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.linear1 = torch.nn.Linear((200 if encoder == 'lstm' else self.emb_dim) * 4, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim * 8, hidden_dim)
        if extract_diff:
            self.fnn = DifferenceExtractor(hidden_dim * 4, hidden_dim * 4)
            self.fnn = torch.nn.DataParallel(self.fnn)
            self.p_theta = CompressedDist(768, 768)
            self.p_phi = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 8, hidden_dim * 8 // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim * 2, 1),
                torch.nn.Sigmoid())
            self.p_phi = torch.nn.DataParallel(self.p_phi)
        self.v_attn = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.nll = torch.nn.NLLLoss()
        self.relu = torch.nn.ReLU()
        self.ls = torch.nn.LogSoftmax(dim=-1)
        self.rerank = rerank
        self.ns = ns
        self.add_null = add_null

        self.lambda_ = lambda_
        self.alpha = alpha

    def set_inner_opt(self, lr=1e-5, num_steps=1):
        par = list(self.p_phi.parameters())
        self.inner_opt = AdamW(par, lr=lr * 100)
        self.inner_sch = get_linear_schedule_with_warmup(self.inner_opt,
                                                         num_warmup_steps=int(num_steps * 0.05),
                                                         num_training_steps=num_steps)

    def step_inner(self):
        torch.nn.utils.clip_grad_norm_(self.p_theta.parameters(), 1)
        self.inner_opt.step()
        self.inner_sch.step()

    def local_maching_aggregation(self, support, query, query_length=[], support_lengths=[], instance_mask=None):
        N = len(support)
        Q = len(query)
        stack = []
        for sup, sls in zip(support, support_lengths):
            buf = []
            for i in range(sup.shape[0]):
                buf.append(sup[i][:sls[i]])
            buf = torch.cat(buf, dim=0)  # [L, D]
            stack.append(buf)
        support = torch.nn.utils.rnn.pad_sequence(stack, batch_first=True)  # [N, Ls, D]
        inv_mask = (support == 0).type(torch.LongTensor).prod(dim=-1)  # [N, Ls]
        inv_mask = inv_mask.type(torch.FloatTensor).unsqueeze(0).unsqueeze(-1).cuda()  # [1, N, Ls, 1]
        query = [query[i][:query_length[i]] for i in range(query.shape[0])]  # [Lq, D] * Q
        query = torch.nn.utils.rnn.pad_sequence(query, batch_first=True)  # [Q, Lq, D]
        inv_mask_q = (query == 0).type(torch.LongTensor).prod(dim=-1).unsqueeze(1).unsqueeze(2).type(
            torch.FloatTensor).cuda()  # [Q, 1, 1, Lq]

        query_ex = torch.cat([query.unsqueeze(1), ] * support.shape[0], dim=1)  # [Q, 1, Lq, D]
        support_ex = torch.cat([support.unsqueeze(0), ] * query.shape[0], dim=0)  # [1, N, Ls, D]

        attention_score = torch.matmul(support_ex, query_ex.transpose(2, 3))  # [Q, N, Ls, Lq]
        if instance_mask is None:
            instance_mask = torch.zeros((Q, N, 1, 1)).type(torch.FloatTensor).cuda()
        else:
            instance_mask = instance_mask.reshape((Q, N, 1, 1))
        attention_score += attention_score.detach().std() * instance_mask
        attn_query = torch.nn.Softmax(dim=2)(attention_score - 1e9 * (inv_mask + inv_mask_q))  # [Q, N, Ls, Lq]
        attn_support = torch.nn.Softmax(dim=3)(attention_score - 1e9 * (inv_mask + inv_mask_q))  # [Q, N, Ls, Lq]

        query_agg = torch.matmul(attn_query.transpose(2, 3), support_ex)  # [Q, N, Lq, D]
        support_agg = torch.matmul(attn_support, query_ex)  # [Q, N, Ls, D]

        query_local = torch.cat([query_ex, query_agg, abs(query_ex - query_agg), query_ex * query_agg],
                                dim=-1)  # [Q, N, Lq, 4D]
        support_local = torch.cat([support_ex, support_agg, abs(support_ex - support_agg), support_ex * support_agg],
                                  dim=-1)  # [Q, N, Ls, 4D]
        query_local = F.relu(self.linear1(query_local))  # [Q, N, Lq, H]
        support_local = F.relu(self.linear1(support_local))  # [Q, N, Ls, H]

        if type(support_lengths[0][0]) != int:
            support_lengths = [[sl.cpu().numpy() for sl in sls] for sls in support_lengths]  # [N, K]
        accum_lengths = [[0, ] + [sum(sls[:i + 1]) for i in range(len(sls))] for sls in support_lengths]

        support_padded = [torch.nn.utils.rnn.pad_sequence(
            [support_local[:, i, s:e, :].transpose(0, 1) for s, e in zip(als[:-1], als[1:])], batch_first=True) for
                          i, als in enumerate(accum_lengths)]  # [K, Ls, Q, H] * N
        support_stacked = torch.nn.utils.rnn.pad_sequence([e.transpose(0, 1) for e in support_padded],
                                                          batch_first=True)  # [N, Ls, K, Q, H]
        support_stacked = support_stacked.transpose(1, 3).transpose(0, 1)  # [Q, N, K, Ls, H]
        support_stacked = support_stacked.reshape(
            (-1, support_stacked.shape[3], support_stacked.shape[4]))  # [Q * N * K, Ls, H]

        query_local = torch.nn.utils.rnn.pad_sequence([query_local[i, :, :, :].transpose(0, 1) for i in range(Q)],
                                                      batch_first=True)  # [Q, Lq, N, H]
        query_local = query_local.transpose(1, 2)  # [Q, N, Lq, H]
        query_local = query_local.reshape((-1, *query_local.shape[2:]))  # [Q * N, Lq, H]

        query_local = self.dropout(query_local)  # [Q * N, Lq, H]
        support_padded = self.dropout(support_stacked)  # [Q * N * K, Ls, H]

        query_rep, _ = self.lstm(query_local)  # [Q * N, Lq, 2H]
        support_rep, _ = self.lstm(support_padded)  # [Q * N * K, Ls, 2H]
        support_rep = support_rep.reshape(
            (Q, N, support_rep.shape[0] // (N * Q), support_rep.shape[1], support_rep.shape[2]))  # [Q, N, K, Ls, 2H]
        query_rep = query_rep.reshape((Q, N, *query_rep.shape[1:]))  # [Q, N, Lq, 2H]

        support_rep = [[support_rep[:, i, j, :l, :] for j, l in enumerate(sls)] for j, sls in
                       enumerate(support_lengths)]  # [[Q, Ls, 2H] * K] * N
        support_rep = [torch.cat([torch.cat([e.max(dim=1)[0], e.mean(dim=1)], dim=-1).unsqueeze(0) for e in sup], dim=0)
                       for sup in support_rep]  # [K, Q, 4H] * N
        support_rep = torch.cat([e.unsqueeze(0) for e in support_rep], dim=0)  # [N, K, Q, 4H]
        support_rep = support_rep.transpose(1, 2).transpose(0, 1)  # [Q, N, K, 4H]
        query_rep = [query_rep[i, :, :l, :] for i, l in enumerate(query_length)]  # [N, Lq, 2H] * Q
        query_rep = torch.cat([torch.cat([e.max(dim=1)[0], e.mean(dim=1)], dim=-1).unsqueeze(0) for e in query_rep],
                              dim=0)  # [Q, N, 4H]

        if len(support_rep.shape) == 1:
            support_rep = support_rep.reshape((1, -1))

        return support_rep, query_rep

    def instance_maching_aggregation(self, support_rep, query_rep):
        K = support_rep.shape[2]
        query_rep_expanded = torch.cat([query_rep.unsqueeze(2), ] * K, dim=2)  # [Q, N, K, 4H]
        concat = torch.cat([support_rep, query_rep_expanded], dim=-1)  # [Q, N, K, 8H]

        attn_beta = self.linear2(concat)  # [Q, N, K, H]
        attn_beta = self.v_attn(attn_beta).squeeze(-1)  # [Q, N, K]
        attn_beta = torch.nn.Softmax(dim=-1)(attn_beta)  # [Q, N, K]
        class_proto = torch.sum(support_rep * attn_beta.unsqueeze(-1), dim=2)  # [Q, N, 4H]

        return class_proto

    def class_maching(self, class_proto, query_rep):
        proto_query = torch.cat([class_proto, query_rep], dim=-1)  # [Q, N, 8H]
        maching_score = self.linear2(proto_query)  # [Q, N, H]
        maching_score = self.v_attn(maching_score).squeeze()

        return maching_score

    def forward(self, support_set, query_set, label, additional):
        '''
        kwargs['suppor_sets']: N lists of support set instances. Each lists has K instances in N-way K-shot setting.
        '''
        kwargs = preprocess(support_set, query_set, label, additional, self.tokenizer)

        N = len(kwargs['support_inputs'])
        support_input = kwargs['support_inputs']
        support_label = kwargs['labels']
        support_lengths = kwargs['support_mask']
        support_lengths = [torch.sum(sl, dim=-1) for sl in support_lengths]
        query_lengths = torch.sum(kwargs['query_mask'], dim=-1)

        incon_loss = 0

        query_rep, query_rep_cls = self.bert_encoder.forward(input_ids=kwargs['query_inputs'],
                                                             attention_mask=kwargs['query_mask'])
        support_input_concat = torch.cat(support_input, dim=0)
        support_mask_concat = torch.cat(kwargs['support_mask'], dim=0)
        support_rep, support_rep_cls = self.bert_encoder.forward(input_ids=support_input_concat,
                                                                 attention_mask=support_mask_concat)

        support_rep = support_rep.reshape((len(support_input), len(support_input[0]), *support_rep.shape[1:]))

        instance_mask = None
        support_rep, query_rep = self.local_maching_aggregation(support_rep,
                                                                query_rep,
                                                                query_length=query_lengths,
                                                                support_lengths=support_lengths,
                                                                instance_mask=instance_mask)
        class_proto = self.instance_maching_aggregation(support_rep, query_rep)
        if self.extract_diff:
            concated = torch.cat([class_proto, query_rep], dim=1)

            sampled_all_ng = self.fnn(concated.detach(), n_samples=16)
            sampled_s_ng = sampled_all_ng[:, :N, :]

            sampled_all = self.fnn(concated, n_samples=16)
            class_proto_ss = sampled_all[:, :N, :]
            query_rep = sampled_all[:, N:, :]

            if abs(self.alpha) > 1e-10:
                self.mi_upper_bound(sampled_s_ng.unsqueeze(0), sampled_s_ng.unsqueeze(0), backprop_only=True, den=1)
                self.step_inner()
                kl_ss = self.mi_upper_bound(sampled_s_ng.unsqueeze(0), sampled_s_ng.unsqueeze(0))
            else:
                kl_ss = 0
        else:
            class_proto_ss = class_proto
            kl_ss = 0
        all_maching_scores = self.class_maching(class_proto_ss, query_rep)

        all_loss = self.nll(self.ls(all_maching_scores), support_label) + self.lambda_ * incon_loss
        all_loss += self.alpha * kl_ss

        return all_loss

    def get_sentence_representations(self, sentences, masks, method='cls'):
        if method == 'cls':
            _, reps = self.bert_encoder.forward(input_ids=sentences,
                                                attention_mask=masks)
        elif method == 'mean':
            reps, _ = self.bert_encoder.forward(input_ids=sentences,
                                                attention_mask=masks)
            masks = masks.type(torch.FloatTensor).cuda().unsqueeze(-1)
            reps = (reps * masks).sum(dim=1) / masks.sum(dim=1)

        return reps

    def nll_loss(self, logits, y):
        like = torch.log(F.softmax(logits, dim=-1) + 1e-8)
        return -torch.mean(torch.sum(like * y.type(torch.FloatTensor).cuda(), dim=-1))

    def _class_sim(self, class_reps):
        dots = torch.matmul(class_reps, class_reps.transpose(-2, -1))
        norms = (class_reps ** 2).sum(dim=-1, keepdim=True) ** 0.5
        cossim = dots / (norms * norms.transpose(-2, -1))
        n = max(0, len(cossim.shape) - 2)
        dis = 1
        for m in cossim.shape:
            dis *= m

        return cossim.sort(dim=-1)[0][:, :, -2].mean().cpu().numpy()

    def predict(self, support_set, query_set, additional):
        kwargs = preprocess(support_set, query_set, [], additional, self.tokenizer)
        N = len(kwargs['support_inputs'])
        support_input = kwargs['support_inputs']
        support_lengths = kwargs['support_mask']
        support_lengths = [torch.sum(sl, dim=-1) for sl in support_lengths]
        query_lengths = torch.sum(kwargs['query_mask'], dim=-1)

        query_rep, query_rep_cls = self.bert_encoder.forward(input_ids=kwargs['query_inputs'],
                                                             attention_mask=kwargs['query_mask'])
        support_input_concat = torch.cat(support_input, dim=0)
        support_mask_concat = torch.cat(kwargs['support_mask'], dim=0)
        support_rep, support_rep_cls = self.bert_encoder.forward(input_ids=support_input_concat,
                                                                 attention_mask=support_mask_concat)
        support_rep = support_rep.reshape((len(support_input), len(support_input[0]), *support_rep.shape[1:]))
        instance_mask = None

        p2 = time.time()
        support_rep, query_rep = self.local_maching_aggregation(support_rep,
                                                                query_rep,
                                                                query_length=query_lengths,
                                                                support_lengths=support_lengths,
                                                                instance_mask=instance_mask)
        class_proto = self.instance_maching_aggregation(support_rep, query_rep)
        if self.extract_diff:
            concated = torch.cat([class_proto, query_rep], dim=1)

            sampled_all = self.fnn(concated, n_samples=16)
            class_proto_ss = sampled_all[:, :N, :]
            query_rep = sampled_all[:, N:, :]
        else:
            class_proto_ss = class_proto
        all_maching_scores = self.class_maching(class_proto_ss, query_rep)

        logits = all_maching_scores
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = self.softmax(logits)
        return probs, self._class_sim(class_proto_ss)

    def mi_upper_bound(self, sampled_1, sampled_2, backprop_only=False, den=1):
        kls = []
        var_mean, var_std = [0, 1]

        def _upper_bound(vectors, means, stds, vectors_2=None):
            choices = random.choices(list(range(vectors.shape[1])), k=16)
            choices = torch.cuda.LongTensor(choices)
            all_idx = torch.cuda.LongTensor(list(range(vectors.shape[1])))
            idx1_ex = all_idx.repeat_interleave(choices.shape[0])
            idx2_ex = choices.repeat(vectors.shape[1])
            cond_probs = self.normal_log_prob(vectors, means, stds, vectors_2=vectors_2)
            if vectors_2 is not None:
                vectors_2_ex = vectors_2[:, idx2_ex]
            else:
                vectors_2_ex = None
            log_probs = self.normal_log_prob(vectors[:, idx1_ex], means, stds, vectors_2=vectors_2_ex)
            mean_log_probs = log_probs.reshape((-1, vectors.shape[1], choices.shape[0])).mean(dim=-1)
            return (cond_probs - mean_log_probs).mean()

        if backprop_only:
            l_sigma = -self.normal_log_prob(sampled_1.detach(), var_mean, var_std, vectors_2=sampled_2.detach()).mean(
                dim=1).mean() / den
            l_sigma.backward(retain_graph=True)
            return None

        tar = list(itertools.combinations(range(sampled_1.shape[2]), 2))
        random.shuffle(tar)
        tar = tar[:32]
        for i, j in tar:
            s1 = sampled_1[:, :, i, :]
            kls.append(_upper_bound(s1, var_mean, var_std, vectors_2=sampled_2[:, :, j, :]))
        return torch.stack(kls).mean()

    def normal_log_prob(self, v, mean, std, vectors_2=None):
        if vectors_2 is not None:
            cated = torch.cat([v, vectors_2], dim=-1)
            probs = self.p_phi(cated)
            log_probs = torch.log(probs + 1e-6).squeeze(-1)
        else:
            var = std ** 2 + 1e-8
            log_scale = std.log()
            log_probs = (-(((v - mean) ** 2) / (2 * var)) - log_scale - np.log(2 * np.pi)).sum(dim=-1)
        return log_probs

    def get_ib_params(self):
        ret = []
        ret += list(self.fnn.parameters())
        return ret

    def get_bert_params(self):
        ret = list(self.bert_encoder.parameters())
        ret += list(self.linear1.parameters())
        ret += list(self.linear2.parameters())
        ret += list(self.lstm.parameters())
        ret += list(self.v_attn.parameters())

        return ret
