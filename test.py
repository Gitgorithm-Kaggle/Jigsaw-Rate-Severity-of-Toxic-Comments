import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class SIMPLE_FC(torch.nn.Module):
    def __init__(self, input_dim, fc_dim, drop=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.fc_1 = torch.nn.Linear(self.input_dim, self.fc_dim)
        self.fc_2 = torch.nn.Linear(self.fc_dim, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        return x.squeeze(dim=1)


class EarlyStopMonitor(object):
    def __init__(self, max_round, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = -1
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance
        self.is_best = False

    def early_stop_check(self, curr_val):
        self.epoch_count += 1

        if not self.higher_better:
            curr_val *= -1

        if self.last_best is None:
            self.last_best = curr_val
            self.best_epoch = self.epoch_count
            self.is_best = True

        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            self.is_best = True

        else:
            self.num_round += 1
            self.is_best = False

        return self.num_round >= self.max_round

def process(data):
    non_toxic, toxic = data.iloc[:, 1], data.iloc[:, 2]
    df_1 = pd.DataFrame({'Content': non_toxic,
                         'Label': 0})
    df_2 = pd.DataFrame({'Content': toxic,
                         'Label': 1})
    df = pd.concat([df_1, df_2], ignore_index=True)
    return df

def shuffle(data, raw_l):

    false_idx = np.arange(raw_l)
    true_idx = np.arange(raw_l, len(data))

    np.random.shuffle(false_idx)
    np.random.shuffle(true_idx)
    split = round(TRAIN_RATIO * raw_l)

    train_false, val_false = false_idx[:split], false_idx[split:]
    train_true, val_true = true_idx[:split], true_idx[split:]

    train_idx = np.concatenate((train_false, train_true), axis=0)
    val_idx = np.concatenate((val_false, val_true), axis=0)

    return train_idx, val_idx

def eval_epoch(val_indices, save):
    # np.random.shuffle(val_indices)
    pred_prob = np.zeros(len(val_indices))
    pred_label = np.zeros_like(pred_prob)
    true_labels = np.zeros_like(pred_prob)

    loss = 0
    num_instance = len(val_indices)
    num_batch = round(num_instance // BATCH_SIZE)
    with torch.no_grad():
        model.eval()

        for batch in range(num_batch):
            src_idx = batch * BATCH_SIZE
            dst_idx = min(src_idx + BATCH_SIZE, num_instance - 1)

            indices = val_idx[src_idx:dst_idx]
            emb = torch.index_select(val_emb, 0, torch.tensor(indices))
            label = np.take(label_l, indices)
            labels = torch.from_numpy(label).float()

            prob = model(emb).sigmoid()
            loss += model_criterion_eval(prob, labels).item()

            true_labels[src_idx:dst_idx] = label
            pred_prob[src_idx:dst_idx] = prob.cpu().numpy()
            pred_label[src_idx:dst_idx] = np.round(prob.cpu().numpy())

    auc_roc = roc_auc_score(true_labels, pred_prob)

    if save:
        df = pd.DataFrame({'CONTENT_INDICES' : val_indices,
                           'TRUE_LABEL' : true_labels,
                           'PRED_PROB' : pred_prob,
                           'PRED_LABEL' : pred_label,
                           'DIFF(ABS)' : abs(true_labels - pred_prob)})

        df.to_csv('./saved_data/{}.csv'.format(MODEL_NUM))

    return auc_roc, loss/num_instance

if not os.path.exists('processed'):
    os.mkdir('processed')
if not os.path.exists('saved_data'):
    os.mkdir('saved_data')
if not os.path.exists('saved_checkpoints'):
    os.mkdir('saved_checkpoints')

MODEL_PERFORMANCE_PATH = './saved_models/model_performance.csv'
VAL_EMBEDDING_PATH = './processed/BERT_validation_data.pt'
TEST_EMBEDDING_PATH = './processed/BERT_comments_to_score.pt'
VAL_CSV_PATH = './data/validation_data.csv'
TEST_CSV_PATH = './data/comments_to_score.csv'
TRAIN_RATIO = 0.7
EPOCH = 50
FC_DIM = 64
BATCH_SIZE = 64
LR = float(3e-4)

MODEL_NUM = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{MODEL_NUM}-SIMPLE_FC-{epoch}.pth'

raw_val_data = pd.read_csv(VAL_CSV_PATH)
# test_data = pd.read_csv(TEST_CSV_PATH)
val_emb = torch.load(VAL_EMBEDDING_PATH)
# test_emb = torch.load(TEST_EMBEDDING_PATH)
val_data = process(raw_val_data)

# split train:val
train_idx, val_idx = shuffle(val_data, len(raw_val_data))
label_l = val_data.Label.values

model = SIMPLE_FC(val_emb.shape[1], FC_DIM)
# model = model.to(device)
model_optimizer = torch.optim.Adam(model.parameters(), lr = LR)
model_criterion = torch.nn.BCELoss()
model_criterion_eval = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(max_round=5)
num_instance = len(train_idx)
num_batch = round(num_instance // BATCH_SIZE)
print('NUM INSTANCE = {} | BATCH SIZE = {} | NUM BATCH = {}'.format(num_instance, BATCH_SIZE, num_batch))

for epoch in range(EPOCH):
    np.random.shuffle(train_idx)
    for batch in range(num_batch):
        src_idx = batch * BATCH_SIZE
        dst_idx = min(src_idx + BATCH_SIZE, num_instance-1)

        indices = train_idx[src_idx:dst_idx]
        emb = torch.index_select(val_emb, 0, torch.tensor(indices))
        label = torch.from_numpy(np.take(label_l, indices)).float()
        prob = model(emb).sigmoid()
        loss = model_criterion(prob, label)
        loss.backward()
        model_optimizer.step()

    val_auc, val_loss = eval_epoch(val_idx, 0)
    print('EPOCH : {} | loss : {} | auc: {}'.format(format(epoch, '03'), val_loss, val_auc))

    if early_stopper.early_stop_check(val_auc):
        print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        best_epoch = early_stopper.best_epoch
        print(f'Loading the best model at epoch {best_epoch}')
        best_model_path = get_checkpoint_path(best_epoch)
        model.load_state_dict(torch.load(best_model_path))
        print(f'Loaded the best model at epoch {best_epoch} for inference')
        model.eval()
        os.remove(best_model_path)
        break
    else:
        if early_stopper.is_best:
            torch.save(model.state_dict(), get_checkpoint_path(epoch))
            print('Saved {}-PREDICTED-{}.pth'.format(MODEL_NUM, early_stopper.best_epoch))

            for i in range(epoch):
                try:
                    os.remove(get_checkpoint_path(i))
                    print('Deleted {}-PREDICT-{}.pth'.format(MODEL_NUM, i))
                except:
                    continue
val_auc, val_loss = eval_epoch(val_idx, 1)