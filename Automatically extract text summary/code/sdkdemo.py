from solution import Solution
from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'train_input':{
        'datapath': 'test_en_label.csv'
    },
    'keywords':{

    },
    'output':{
        "filename": 'b.csv',
    }
}

solution.set_params(params)
solution.run()