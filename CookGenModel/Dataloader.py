class TaskSpecificFileIO(object):

    @staticmethod
    def read_csv_as_list(csv_file):

        f = open(csv_file,'r')
        return f.read().splitlines()[1:]
    
class Dataloader(object):

    data = None
    text_data = None
    label_data = None

    @staticmethod
    def prepare_data():
        data_items = TaskSpecificFileIO.read_csv_as_list('annotations_combined_v2.csv')

        #clean to the extend possibly, noisy because of how data is created
        X, Y = [], []
        action_list = []
        for item in data_items:
            text = item.split(',')[1:2]
            actions = item.split(',')[-1]
            if actions not in action_list:
                action_list.append(actions)
                X.append(text); Y.append(actions)

        Dataloader.text_data = X
        Dataloader.label_data = Y

        dataset = list(zip(X,Y))
        data = []
        n_data = len(dataset)
        for i in range(n_data):
            data_point = dataset[i]
            text = data_point[0]
            label = data_point[1]
            point_X = [i]
            point_Y = [0.0 for _ in action_list]; point_Y[action_list.index(label)] = 1.0
            data.append((point_X,point_Y))

        Dataloader.data = data
