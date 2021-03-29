import pickle
import os


class Serializable:

    def __init__(self):
        pass

    def save(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)

        pickle_out = open(path+'/'+filename, 'wb+')
        pickle.dump(self, pickle_out)
        pickle_out.close()

    @staticmethod
    def load(path):
        pickle_in = open(path, 'rb')
        obj = pickle.load(pickle_in)
        pickle_in.close()

        return obj
