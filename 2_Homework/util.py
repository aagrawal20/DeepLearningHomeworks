from sklearn.preprocessing import OneHotEncoder

def one_hot_encode(labels):
    """ This function one hot encodes the labels for the data"""
    
    # create encoder
    encoder = OneHotEncoder()
    # fit the encoder
    encoder.fit(labels.reshape(-1,1))
    # transform current labels to be one hot encoded
    labels = encoder.transform(labels.reshape(-1,1)).toarray()
    
    return labels

# train - test split
def split_data(proportion, data, labels):
    """ This function splits the data based on the proportion"""
    
    # number of examples
    num_examples = data.shape[0]
    # split index
    split_index = int(proportion * num_examples)
    # train - test data split
    train_data, test_data = data[:split_index], data[split_index:]
    # train - test labels split
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    
    return train_data, train_labels, test_data, test_labels