from solution import Solution
from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'train input':{
        'datapath': 'fasttext_training_1115.csv',
        'label': 'label'
    },
    'test input':{
        'datapath': 'fasttext_test_1115.csv',
        'label': 'label'
    },
    'test input1':{
        'datapath': 'fasttext_test_1114.csv'
        'label': 'label'
    },
    'algorithm':{

    },
    'output':{
        "filename": 'a.csv',
    }
}

solution.set_params(params)
solution.run()
solution.run(Mode.Test)