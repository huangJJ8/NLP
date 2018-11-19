from solution import Solution
from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'train_input':{
        'datapath': 'fasttext_training.csv',
        'label': 'label'
    },
    'test_input':{
        'datapath': 'fasttext_test.csv',
        'label': 'label'
    },
    'fasttext':{

    },
    'output':{
        "filename": 'a.csv',
    }
}

solution.set_params(params)
solution.run()

params = {
    'train_input':{
        'datapath': 'fasttext_training.csv',
        'label': 'label'
    },
    'test_input':{
        'datapath': 'fasttext_test_unlabel.csv',
    },
    'fasttext':{

    },
    'output':{
        "filename": 'a.csv',
    }
}

solution.set_params(params)
solution.run(Mode.Test)