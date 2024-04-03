import numpy as np
from src.dists import avg, norm_hamming, get_params


def count_metrics(tp, fp, fn):
    precision = (tp / (tp + fp))
    recall = (tp / (tp + fn))
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def get_f1(y_true, y_pred):
    tp, fp, fn = 0, 0, 0
    for idx in range(len(y_true)):
        if y_pred[idx] == 1:
            if y_true[idx] == 1:
                tp += 1
            else:
                fp += 1
        elif y_pred[idx] == 0:
            if y_true[idx] == 1:
                fn += 1
    precision, recall, f1 = count_metrics(tp, fp, fn)
    return precision, recall, f1


def train_T(hash_name, res, train_size, from_t, to_t):
    for T in np.linspace(from_t, to_t, 10):
        orig_hashes = res[hash_name].orig_hashes[:train_size]
        param2disorted_hashes = res[hash_name].param2distorted_hashes
        prec_res = []
        recall_res = []
        f1_res = []

        for img_idx in range(train_size):
            y_true = []
            y_pred = []
            img_hash = orig_hashes[img_idx]
            # originals
            y_true += [int(img_idx == idx) for idx in range(train_size)]
            y_pred += [int(norm_hamming(img_hash, other_hash) < T) for other_hash in orig_hashes]
            for param in get_params(res):
                # distorts
                y_true += [int(img_idx == idx) for idx in range(train_size)]
                y_pred += [int(norm_hamming(img_hash, distorted_hash) < T) for distorted_hash in
                           param2disorted_hashes[param][:train_size]]
            precision, recall, f1 = get_f1(y_true, y_pred)
            prec_res.append(precision)
            recall_res.append(recall)
            f1_res.append(f1)
        print(f'T={round(T, 6)}: precision={avg(prec_res)}, recall={avg(recall_res)}, f1={avg(f1_res)}')


def test_T(hash_name, res, T, train_size, max_limit=None):
    orig_hashes = res[hash_name].orig_hashes[train_size:]
    param2disorted_hashes = res[hash_name].param2distorted_hashes
    if not max_limit:
        max_limit = len(res[hash_name].orig_hashes)
        print(max_limit)
    prec_res = []
    recall_res = []
    f1_res = []

    for img_idx in range(max_limit - train_size):
        y_true = []
        y_pred = []
        img_hash = orig_hashes[img_idx]
        # originals
        y_true += [int(img_idx == idx) for idx in range(max_limit - train_size)]
        y_pred += [int(norm_hamming(img_hash, other_hash) < T) for other_hash in orig_hashes]
        for param in get_params(res):
            # distorts
            y_true += [int(img_idx == idx) for idx in range(max_limit - train_size)]
            y_pred += [int(norm_hamming(img_hash, distorted_hash) < T) for distorted_hash in
                       param2disorted_hashes[param][train_size:]]
        precision, recall, f1 = get_f1(y_true, y_pred)
        prec_res.append(precision)
        recall_res.append(recall)
        f1_res.append(f1)
    return (avg(prec_res), avg(recall_res), avg(f1_res))


def train_T_all_res(hash_name, res_list, train_size, from_t, to_t):
    for T in np.linspace(from_t, to_t, 10):
        prec_res = []
        recall_res = []
        f1_res = []
        for res in res_list:
            orig_hashes = res[hash_name].orig_hashes[:train_size]
            param2disorted_hashes = res[hash_name].param2distorted_hashes

            for img_idx in range(train_size):
                y_true = []
                y_pred = []
                img_hash = orig_hashes[img_idx]
                # originals
                y_true += [int(img_idx == idx) for idx in range(train_size)]
                y_pred += [int(norm_hamming(img_hash, other_hash) < T) for other_hash in orig_hashes]
                for param in get_params(res):
                    # distorts
                    y_true += [int(img_idx == idx) for idx in range(train_size)]
                    y_pred += [int(norm_hamming(img_hash, distorted_hash) < T) for distorted_hash in
                               param2disorted_hashes[param][:train_size]]
                precision, recall, f1 = get_f1(y_true, y_pred)
                prec_res.append(precision)
                recall_res.append(recall)
                f1_res.append(f1)
        print(f'T={round(T, 6)}: precision={avg(prec_res)}, recall={avg(recall_res)}, f1={avg(f1_res)}')
