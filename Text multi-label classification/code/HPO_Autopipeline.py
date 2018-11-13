from mlstudiosdk.solution_gallery.solution.SolutionBase import SolutionBase
from mlstudiosdk.modules.utils.featureType import FeatureType
from mlstudiosdk.modules.benchmark.provider import FeProvider
from mlstudiosdk.modules.benchmark.provider import (ModeProvider, SearchSpaceProvider, IndustryProvider, MetricProvider,
                                                    LocalSearchProvider)

class FinanceMultipletextClassifier(SolutionBase):
    predefined_params = {
        'preprocessing': {
            'industry_field': IndustryProvider.Insurance.name,
            FeProvider.Imputer.name: {'process_columns': FeatureType.All, 'fillna': -1},
            FeProvider.UnimportantFeatureDropper.name: {'process_columns':FeatureType.Numerical}
        },
        'hparams': {
            "industry_field": IndustryProvider.Insurance.name,
            "mode": ModeProvider.Local.name,
            "search_space": SearchSpaceProvider.Medium.name,
        },
        'sampling': {
            "n_subsets": 1,
            "maxsize": 100000000,
            "size_subset": .1,
        },
        'hparams_input': {
            "industry_field": IndustryProvider.Insurance.name,
            "mode": ModeProvider.Local.name,
            "search_space": SearchSpaceProvider.Medium.name,
        },
        'hyperparam_filter': {
            "cv": 3,
            "mode": ModeProvider.Local.name,
            "metric": MetricProvider.LOG_LOSS.name,
            "search_algo": LocalSearchProvider.RandomSearch.name,
            "n_iter": 40,
            "n_jobs": -1,
            "n_best": 1,
        },
        'hyperparam_searcher': {
            "cv": 3,
            "mode": ModeProvider.Local.name,
            "metric": MetricProvider.LOG_LOSS.name,
            "search_algo": LocalSearchProvider.RandomSearch.name,
            "n_iter": 40,
            "n_jobs": -1,
        },
        'modeler': {
            "metric": MetricProvider.LOG_LOSS.name,
            "n_jobs": -1,
        },
    }

    def __init__(self):
        super().__init__()
        self.category_ = 'Automation'
        self.name_ = 'Finance_Text_Multiple_Label_Classifier'
        self.subsampling = True
        self.model()
        self.set_params(self.predefined_params)

    def form_params(self, solution=IndustryProvider.Insurance, **kwargs):
        return super().form_params(solution, **kwargs)
   
    def model(self):
        reader1 = self.myscheme.new_node("mlstudiosdk.modules.components.io.reader.Reader")
        reader1.set_title("train_input")
        reader2 = self.myscheme.new_node("mlstudiosdk.modules.components.io.reader.Reader")
        reader2.set_title("test_input")

        TextClassifierModel = self.myscheme.new_node("mlstudiosdk.modules.components.nlp.Text_Multiple_Label_Classifier.Text_Multiple_Label_Classifier")
        TextClassifierModel.set_title('sentiment_model_EN')

        self.myscheme.new_link(reader1, "Data", TextClassifierModel, "Train Data")
        self.myscheme.new_link(reader2, "Data", TextClassifierModel, "Test Data")

        writer = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.Writer")
        writer.set_title("output")

        self.myscheme.new_link(TextClassifierModel, "Data_OUT", writer, "Data")
        self.myscheme.new_link(TextClassifierModel, "predict_proba_OUT", writer, "Data")

        # visualization node
        eva_visualization = self.myscheme.new_node("mlstudiosdk.modules.components.visualization.evaluation_matrix.Evaluation")
        eva_visualization.set_title("evaluation_visualization")
        evaluation_writer = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.JsonWriter")
        evaluation_writer.set_title("evaluation_output")

        pred_stat_visualization = self.myscheme.new_node("mlstudiosdk.modules.components.visualization.data_statistics.Statistics")
        pred_stat_visualization.set_title("pred_statistics_visualization")
        pred_stat_writer = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.JsonWriter")
        pred_stat_writer.set_title("pred_statistics_output")

        # visualization link
        self.myscheme.new_link(reader2, "Data", pred_stat_visualization, "Data")
        self.myscheme.new_link(pred_stat_visualization, "Data", pred_stat_writer, "Data")

        self.myscheme.new_link(sentimentModel, "Evaluation Results", eva_visualization, "Result")
        self.myscheme.new_link(TextClassifierModel, "Metric Score", eva_visualization, "Metric Score")       
        self.myscheme.new_link(eva_visualization, "Evaluation", evaluation_writer, "Data")

