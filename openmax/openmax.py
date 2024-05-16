import torch
import numpy as np
import pickle

from utils.configs import config
from openmax.evt_fitting import weibull_tail_fitting
from openmax.openmax_utils import *


NCLASSES = config.num_species.sum()
ALPHA_RANK = 2
LABELS = range(NCLASSES)
WEIBULL_TAIL_SIZE = NCLASSES


class OpenMax:
    def __init__(self, net, fold):
        self.net = net
        self.dev = torch.device('cuda:0')
        self.fold = fold

        self.net.to(self.dev)
        self.net.eval()

    def compute(self, dl):
        openmax = []
        prob_u = []
        y_true = []

        with torch.no_grad():
            for data in dl:
                x = data[0].to(self.dev)
                y = data[1].cpu().numpy()
                out = self.net(x).cpu().numpy()

                for logits, label in zip(out, y):
                    temp_openmax, temp_prob_u = compute_openmax(
                        logits, self.fold)
                    openmax.append(temp_openmax)
                    prob_u.append(temp_prob_u)
                    y_true.append(label)

        openmax = np.asarray(openmax)
        prob_u = np.asarray(prob_u)
        y_true = np.asarray(y_true)

        y_true_bin = get_bin_labels(y_true)
        roc = compute_roc(y_true_bin, prob_u)
        roc_thresh = roc['thresholds']

        best_idx = np.argmax(roc['tpr'] - roc['fpr'])
        best_thresh = roc_thresh[best_idx]
        return openmax, best_thresh

    def predict(self, dl):
        y_pred = []
        openmax, thresh = self.compute(dl)

        for scores in openmax:
            temp = get_openmax_predict_int(scores, thresh)
            y_pred.append(temp)

        y_pred = np.asarray(y_pred)
        return y_pred


def create_model(net, dataloader, fold):
    device = torch.device('cuda:0')
    logits_correct = []
    label_correct = []
    with torch.no_grad():
        for data in dataloader:
            x = data[0].to(device)
            y = data[1].cpu().numpy()
            out = net(x).cpu().numpy()
            correct_index = get_correct_classified(y, out)
            out_correct = out[correct_index]
            y_correct = y[correct_index]

            for logits, label in zip(out_correct, y_correct):
                logits_correct.append(logits)
                label_correct.append(label)

    logits_correct = np.asarray(logits_correct)
    label_correct = np.asarray(label_correct)

    av_map = {}
    for label in LABELS:
        av_map[label] = logits_correct[label_correct == label]

    feature_mean = []
    feature_distance = []
    for label in LABELS:
        mean = compute_mean_vector(av_map[label])
        distance = compute_distance_dict(mean, av_map[label])
        feature_mean.append(mean)
        feature_distance.append(distance)

    weibull_model = build_weibull(
        mean=feature_mean,
        distance=feature_distance,
        tail=WEIBULL_TAIL_SIZE,
        fold=fold,
    )
    return weibull_model


def build_weibull(mean, distance, tail, fold):
    model_path = f'weibull_models/weibull_model_{fold}.pkl'
    weibull_model = {}

    for label in LABELS:
        weibull_model[label] = {}
        weibull = weibull_tail_fitting(
            mean[label], distance[label], tailsize=tail)
        weibull_model[label] = weibull

    with open(model_path, 'wb') as file:
        pickle.dump(weibull_model, file)
    return weibull_model


def query_weibull(label, weibull_model):
    return weibull_model[label]


def recalibrate_scores(weibull_model, activation_vector):
    ranked_list = np.flip(np.argsort(activation_vector))
    alpha_weights = [
        ((ALPHA_RANK+1) - i) / float(ALPHA_RANK) for i in range(1, ALPHA_RANK+1)
    ]
    # print(f'alpha_weights: {alpha_weights}')
    ranked_alpha = np.zeros_like(activation_vector)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    ranked_alpha *= 0.5

    openmax_scores = []
    openmax_scores_u = []

    for label in LABELS:
        weibull = query_weibull(label, weibull_model)
        av_distance = compute_distance(weibull['mean'], activation_vector)
        wscore = weibull['weibull_model'].w_score(av_distance)
        # print(f'wscore_{label}: {wscore}')
        modified_score = activation_vector[label] * \
            (1 - wscore*ranked_alpha[label])
        openmax_scores += [modified_score]
        openmax_scores_u += [activation_vector[label] - modified_score]

    openmax_scores = np.asarray(openmax_scores)
    openmax_scores_u = np.asarray(openmax_scores_u)
    # print(f'openmax_score: {openmax_scores}')
    # print(f'openmax_score_u: {openmax_scores_u}')
    # print(f'sum openmax_score_u: {np.sum(openmax_scores_u)}')
    # print('-' * 80)

    openmax_prob, prob_u = compute_openmax_probability(
        openmax_scores, openmax_scores_u)
    return openmax_prob, prob_u


def compute_openmax_probability(openmax_scores, openmax_scores_u):
    e_k = np.exp(openmax_scores)
    e_u = np.exp(np.sum(openmax_scores_u))
    openmax_arr = np.concatenate((e_k, e_u), axis=None)
    total_denominator = np.sum(openmax_arr)
    prob_k = e_k / total_denominator
    prob_u = e_u / total_denominator
    res = np.concatenate((prob_k, prob_u), axis=None)
    return res, prob_u


def compute_openmax(activation_vector, fold):
    model_path = f'weibull_models/weibull_model_{fold}.pkl'
    with open(model_path, 'rb') as file:
        weibull_model = pickle.load(file)
    openmax, prob_u = recalibrate_scores(weibull_model, activation_vector)
    return openmax, prob_u
