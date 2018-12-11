from solution import Solution
from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'train_input':{
        'datapath': 'test_CN.csv'
    },
    'keywords':{

    },
    'output':{
        "filename": 'a.csv',
    }
}

solution.set_params(params)
solution.run()
