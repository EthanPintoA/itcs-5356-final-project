import pickle

def load_pickle_data(train_path, val_path):
    with open(train_path, 'rb') as train_file:
        train_data = pickle.load(train_file)
    with open(val_path, 'rb') as val_file:
        val_data = pickle.load(val_file)
    return train_data, val_data
