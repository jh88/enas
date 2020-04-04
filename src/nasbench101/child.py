from nasbench import api


class Child():
    INPUT = 'input'
    OUTPUT = 'output'
    OPS = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']

    def __init__(self, nasbench, **kwargs):
        self.nasbench = nasbench

    def get_model_spec(self, arc):
        ops = [self.INPUT]
        matrix = [[0 for _ in range(7)] for _ in range(7)]

        for i in range(5):
            arc_index = arc[i * 2]
            arc_op = arc[i * 2 + 1]

            ops.append(self.OPS[arc_op])
            matrix[arc_index][i + 1] = 1

        ops.append(OUTPUT)

        for row in matrix:
            if not sum(row):
                row[-1] = 1

        model_spec = api.ModelSpec(matrix=matrix, ops=ops)

        return model_spec

    def get_accuracies(self, arc):
        model_spec = self.get_model_spec(arc)
        data = self.nasbench.query(model_spec)

        return data

    def build_valid_rl(self, key='test_accuracy'):
        data = self.get_accuracies(self.sample_arc)
        self.accuracy = data[key]
        
        self.train_acc = data['train_accuracy']
        self.valid_acc = data['validation_accuracy']
        self.test_acc = data['test_accuracy']

    def connect_controller(self, controller_model):
        self.sample_arc = controller_model.sample_arc
