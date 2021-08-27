import argparse
import numpy as np
import pickle
import torch
import os
import itertools
import yaml
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score
from data_processor import HuffpostDataset_ICLR, FewRelDataset_ICLR
from model import MLMAN, ProtoNet
from tqdm import tqdm
from pprint import pprint
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as pl


def evaluate(model, dataset, eval_steps=1000, mode='dev', multi_source=False, query_size=100):
    model.eval()

    train_Q = dataset.setting.Q
    dataset.set_query_size(query_size)
    preds = []
    golds = []
    mean_cossims = []
    task_accs = []

    bar = tqdm(dataset.get_eval_iterator(max_iter=eval_steps, mode=mode), total=eval_steps)
    for ss, qs, l, add in bar:
        probs, _ = model.predict(ss, qs, add, labels=l)
        if hasattr(model, 'maml') and model.maml:
            model.reset_maml()
        task_acc = accuracy_score(l, probs.argmax(dim=-1).cpu().numpy().tolist())
        preds.extend(probs.argmax(dim=-1).cpu().numpy().tolist())
        golds.extend(l)
        task_accs.append(task_acc)
        bar.set_description('Acc = {:.5f}'.format(np.array(task_accs).mean()))

    mlb = MultiLabelBinarizer()
    # golds = MultiLabelBinarizer().fit_transform(golds)
    # preds = MultiLabelBinarizer().fit_transform(preds)
    f1 = f1_score(golds, preds, average='micro')
    acc = accuracy_score(golds, preds)
    model.train()
    dataset.set_query_size(train_Q)
    return f1, 0


def train(model, dataset, setting):
    num_train_steps = setting['num_train_steps']
    num_eval_steps = setting['num_eval_steps']
    early_stop = setting['early_stop']
    eval_interval = setting['eval_interval']
    model_path = setting['model_path']
    lr = setting['lr']
    c = 0
    use_ib = 'info_bottle' in setting and setting['info_bottle']
    multi_source = setting['data'] == 'multi_source'
    print(use_ib, multi_source)
    additional_params = False

    if 'decompose' not in setting:
        setting['decompose'] = False
    if 'ss_rel' not in setting:
        setting['ss_rel'] = False

    if use_ib:
        bert_params = list(model.bert_encoder.parameters())
        if hasattr(model, 'bert_ss'):
            bert_params += list(model.bert_ss.parameters())
        optimizer = AdamW(bert_params, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(num_train_steps * 0.05),
                                                    num_training_steps=num_train_steps)
        ib_vars = model.get_ib_params()
        optimizer_ib = AdamW(ib_vars, lr=lr * 100)
        scheduler_ib = get_linear_schedule_with_warmup(optimizer_ib,
                                                       num_warmup_steps=int(num_train_steps * 0.05),
                                                       num_training_steps=num_train_steps)
        additional_params = True
    elif setting['model'] in ['proto', 'maml'] and (setting['ss_rel'] or setting['decompose']):
        bert_params = list(model.bert.parameters())
        # bert_params += list(model.bert_ss.parameters())
        optimizer = AdamW(bert_params, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(num_train_steps * 0.05),
                                                    num_training_steps=num_train_steps)
    elif setting['model'] == 'relnet':
        bert_params = list(model.bert.parameters())
        rel_params = list(model.rel_score.parameters())
        optimizer = AdamW(bert_params, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(num_train_steps * 0.05),
                                                    num_training_steps=num_train_steps)
        optimizer_rel = AdamW(rel_params, lr=lr * 100)
        scheduler_rel = get_linear_schedule_with_warmup(optimizer_rel,
                                                        num_warmup_steps=int(num_train_steps * 0.05),
                                                        num_training_steps=num_train_steps)
        is_relnet = True
    else:
        bert_params = model.parameters()
        optimizer = AdamW(bert_params, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(num_train_steps * 0.05),
                                                    num_training_steps=num_train_steps)
    if setting['ss_rel']:
        bert_params = model.get_bert_params()
        optimizer = AdamW(bert_params, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(num_train_steps * 0.05),
                                                    num_training_steps=num_train_steps)
        ib_vars = model.get_ib_params()
        optimizer_ib = AdamW(ib_vars, lr=lr)
        scheduler_ib = get_linear_schedule_with_warmup(optimizer_ib,
                                                       num_warmup_steps=int(num_train_steps * 0.05),
                                                       num_training_steps=num_train_steps)
        model.set_inner_opt(lr=lr, num_steps=num_train_steps)
        additional_params = True
    failed = 0
    best_score = -1
    print(is_relnet)

    while c < num_train_steps:
        for ss, qs, l, add in dataset.get_train_iterator(max_iter=eval_interval):
            loss = model(ss, qs, l, add)
            loss.backward(retain_graph=True)
            if setting['model'] == 'maml':
                model.reset_maml()
            torch.nn.utils.clip_grad_norm_(bert_params, max_norm=1)
            if additional_params:
                torch.nn.utils.clip_grad_norm_(ib_vars, max_norm=1)
            if is_relnet:
                torch.nn.utils.clip_grad_norm_(rel_params, max_norm=1)
            c += 1
            if c % 1 == 0:
                optimizer.step()
                if additional_params:
                    optimizer_ib.step()
                if is_relnet:
                    optimizer_rel.step()
                scheduler.step()
                if additional_params:
                    scheduler_ib.step()
                if is_relnet:
                    scheduler_rel.step()
                bert_norm = 0
                ib_norm = 0
                optimizer.zero_grad()
                if additional_params:
                    optimizer_ib.zero_grad()
                    model.inner_opt.zero_grad()
                if is_relnet:
                    optimizer_rel.zero_grad()

            print('\rstep = {}, loss = {:.5f}, grad_bert = {}, grad_ib = {}'.format(c, loss.detach().cpu().numpy(),
                                                                                    bert_norm, ib_norm), end='',
                  flush=True)
        print('')
        with torch.no_grad():
            dev_score, _ = evaluate(model, dataset, eval_steps=500, multi_source=multi_source)
        print('')
        print('step = {}, Dev accuracy = {:.4f}'.format(c, dev_score), flush=True)
        '''
    with torch.no_grad():
      analyse_dev(model, dataset, global_steps=c, query_size=(5 if setting['data'] == 'meddra' else 125))
    '''
        if best_score < dev_score:
            failed = 0
            best_score = dev_score
            torch.save(model.state_dict(), model_path)
        else:
            failed += 1
            if failed >= early_stop:
                print('Early stopping triggered.')
                break
    print('Best dev accuracy: {:.4f}'.format(best_score))
    return best_score


def test_model(model, dataset, setting):
    num_eval_steps = setting['num_eval_steps']
    model_path = setting['model_path']
    print('eval model path = ', model_path)
    odd_keys = model.load_state_dict(torch.load(model_path), strict=False)
    print(odd_keys)
    test_accs = []
    with torch.no_grad():
        for i in range(5):
            test_acc, _ = evaluate(model, dataset, eval_steps=num_eval_steps, mode='test', query_size=5)
            test_accs.append(test_acc)
            print('Seed #{}: Accuracy = {:.4f}'.format(i, test_acc), flush=True)
    print('Test accuracy = {:.4f} +- {:.4f}'.format(np.mean(test_accs), np.std(test_accs)))
    pickle.dump(test_accs,
                open('accuracies_{}.bin'.format(model_path.split('/')[-1] + '_N{}K{}'.format(args.test_N, args.test_K)),
                     'wb'))


def optimize_hyperparameters(args):
    results = {}
    gamma_list = [10 ** e for e in range(-6, 1, 2)]
    for lr, beta, gamma in itertools.product([1e-5, 3e-5, 5e-5], [10 ** e for e in range(-6, 1, 2)], gamma_list):
        args.lr = lr
        args.margin = 0
        args.gamma = (0 if args.ablation else gamma)
        args.beta = beta
        score = main(args)
        results[(lr, beta, gamma)] = score
    print(results)
    pprint(sorted(list(results.itmes()), key=lambda x: -x[-1]))


def main(args):
    setting = yaml.safe_load(open(args.config_file))
    if args.data == 'meddra':
        setting['task_type'] = 'all'
    setting['data'] = args.data
    setting['data_dir'] = args.data_dir
    setting['lr'] = args.lr
    diff_extractor = (setting['diff_extractor'] if 'diff_extractor' in setting else None)

    if not args.eval_mode or ('model_path' not in setting):
        setting['model_path'] = '{}_{}_{}_N{}_K{}_Q{}_{}'.format(args.data,
                                                                 setting['model'],
                                                                 setting['lr'],
                                                                 setting['meta_setting']['N'],
                                                                 setting['meta_setting']['K'],
                                                                 setting['meta_setting']['Q'],
                                                                 ("w_de" if diff_extractor else "none"))
        if abs(args.gamma) > 1e-8:
            setting['model_path'] += '_gamma{}'.format(args.gamma)
        setting['model_path'] = os.path.join('models', setting['model_path'])

    pprint(setting)

    if setting['data'] == 'huffpost':
        dataset = HuffpostDataset_ICLR(setting['data_dir'],
                                       data_size=setting['data_size'],
                                       label_split=(setting['label_split'] if 'label_split' in setting else None))
    elif setting['data'] == 'fewrel':
        dataset = FewRelDataset_ICLR(setting['data_dir'],
                                     task_type=setting['task_type'],
                                     data_size=setting['data_size'],
                                     label_split=(setting['label_split'] if 'label_split' in setting else None),
                                     use_loc_info=(setting['cnn'] if 'cnn' in setting else False))

    if args.eval_mode:
        setting['meta_setting']['N'] = args.test_N
        setting['meta_setting']['K'] = args.test_K
        print('Test at {}-way {}-shot setting'.format(args.test_N, args.test_K))
    dataset.set_task_setting(**setting['meta_setting'])
    if args.eval_mode and setting['model'] == 'maml':
        try:
            args.gamma = float(args.eval_model_path.split('gamma')[-1])
        except:
            args.gamma = 0
        print(args.gamma)

    bert_model = 'bert-base-uncased'

    if setting['model'] == 'mlman':
        model = MLMAN(encoder=bert_model,
                      ns=setting['ns'],
                      ss_rel=diff_extractor,
                      gamma=args.gamma)
    elif setting['model'] == 'proto':
        model = ProtoNet(bert_type=bert_model, gamma=args.gamma, diff_extractor=diff_extractor)
    elif setting['model'] == 'maml':
        model = ProtoNet(bert_type=bert_model, alpha=args.alpha, gamma=args.gamma, diff_extractor=diff_extractor,
                         maml=True)
    model = model.cuda()
    if args.eval_mode:
        if args.eval_model_path != '':
            setting['model_path'] = args.eval_model_path
        test_model(model, dataset, setting)
    else:
        score = train(model, dataset, setting)
    del model
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.yaml')
    parser.add_argument('--eval_model_path', type=str, default='')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--eval_mode', action='store_true')
    parser.add_argument('--analyse_mode', action='store_true')
    parser.add_argument('--iclr', action='store_true')
    parser.add_argument('--data', type=str, default='huffpost')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--margin', type=float, default=0)
    parser.add_argument('--test_N', type=int, default=5)
    parser.add_argument('--test_K', type=int, default=1)
    parser.add_argument('--const_inter', action='store_true')
    parser.add_argument('--stack', action='store_true')
    parser.add_argument('--magic_number', type=int, default=0)
    parser.add_argument('--optuna', action='store_true')
    parser.add_argument('--ablation', action='store_true')

    args = parser.parse_args()

    main(args)
