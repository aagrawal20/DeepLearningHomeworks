from sklearn.preprocessing import OneHotEncoder
import random
from tensorflow.python.ops.gen_nn_ops import relu, elu
from tensorflow.python.ops.nn_ops import leaky_relu
from sklearn.model_selection import KFold
import numpy as np


# one hot encode
def one_hot_encode(labels):
    """ This function one hot encodes the labels for the data"""

    # create encoder
    enc_tr = OneHotEncoder()
    # fit the encoder with the labels
    enc_tr.fit(labels.reshape(-1, 1))
    # transform current labels to be one hot encoded
    labels = enc_tr.transform(labels.reshape(-1, 1)).toarray()

    return labels


# train-test split
def split_data(proportion, data, labels):  # TODO: update the function to not hardcode validation split
    """ This function splits the data based on the proportion"""

    # get the number of examples
    num_examples = data.shape[0]
    # create index to split on
    split_idx = int(proportion * num_examples)
    # split data train, test
    data_1, data_2 = data[:split_idx], data[split_idx:]
    # split labels train, test
    labels_1, labels_2 = labels[:split_idx], labels[split_idx:]

    return data_1, labels_1, data_2, labels_2


def generate_adam(default):
    lr_list = [0.01, 0.03, 0.001, 0.003, 0.0001, 0.0003]
    beta1_list = [0.9, 0.99, 0.999]
    beta2_list = [0.999, 0.9999, 0.99999]

    lr, b_1, b_2 = random.choice(lr_list), random.choice(beta1_list), random.choice(beta2_list)

    if default:
        return 0.001, 0.9, 0.999
    else:
        return lr, b_1, b_2

# implement K-Fold
def cross_validate(session, train_x_all, train_y_all, cross_entropy,input_p, output_p, init, optimizer, accuracy, split_size=5):
    results = []
    kf = KFold(n_splits=split_size)
    global val_min_loss, no_loss_drop
    for train_idx, val_idx in kf.split(train_x_all, train_y_all):
        train_x = train_x_all[train_idx]
        train_y = train_y_all[train_idx]
        val_x = train_x_all[val_idx]
        val_y = train_y_all[val_idx]

        print("\nStart training")
        session.run(init)
        batch_size = 100
        val_min_loss = 999
        no_loss_drop = 0
        for epoch in range(500):
            total_batch = train_x.shape[0] // batch_size
            ce_vals = []
            for i in range(total_batch):
                batch_x = train_x[i * batch_size:(i + 1) * batch_size]
                batch_y = train_y[i * batch_size:(i + 1) * batch_size]
                _, c = session.run([optimizer, metric], feed_dict={input_p: batch_x, output_p: batch_y})
                ce_vals.append(c)

            avg_val_ce = sum(ce_vals) / len(ce_vals)
            print('Fold CROSS ENTROPY: ' + str(avg_val_ce))

            # if val_min_loss > min(ce_vals):
                    # val_min_loss = min(ce_vals)
                    # no_loss_drop = 0
            # else:
                    # no_loss_drop += 1
            # if no_loss_drop > 5:
                    # print("Early Stopping the model training as the loss has not dropped in the last %d runs", no_loss_drop)
                    # break

        results.append(session.run([cross_entropy, accuracy], feed_dict={input_p: val_x, output_p: val_y}))
    return results