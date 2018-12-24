from mlstudiosdk.solution_gallery.solution.SolutionBase import SolutionBase

class Solution(SolutionBase):

    def __init__(self):
        super().__init__()
        self.model()

    def model(self):

        reader1 = self.myscheme.new_node("mlstudiosdk.modules.components.io.reader.Reader")
        reader1.set_title("train_input")

        Text_Extract_Summary = self.myscheme.new_node('Text_Extract_Summary.Text_Extract_Summary')
        Text_Extract_Summary.set_title('textrank')
        outputwriter = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.Writer")
        outputwriter.set_title("output")
        eva_visualization = self.myscheme.new_node(
            "mlstudiosdk.modules.components.visualization.evaluation_matrix.Evaluation")
        eva_visualization.set_title("evaluation_visualization")
        evaluation_writer = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.JsonWriter")
        evaluation_writer.set_title("evaluation_output")
        pred_stat_visualization = self.myscheme.new_node(
            "mlstudiosdk.modules.components.visualization.data_statistics.Statistics")
        pred_stat_visualization.set_title("pred_statistics_visualization")
        pred_stat_writer = self.myscheme.new_node("mlstudiosdk.modules.components.io.writer.JsonWriter")
        pred_stat_writer.set_title("pred_statistics_output")

#         self.myscheme.new_link(reader2, "Data", pred_stat_visualization, "Data")
        self.myscheme.new_link(pred_stat_visualization, "Data", pred_stat_writer, "Data")

        self.myscheme.new_link(Text_Extract_Summary, "Evaluation Results", eva_visualization, "Result")
        self.myscheme.new_link(Text_Extract_Summary, "Metric Score", eva_visualization, "Metric Score")
        # self.myscheme.new_link(sentimentModel, "Metric", eva_visualization, "Metric")
        self.myscheme.new_link(eva_visualization, "Evaluation", evaluation_writer, "Data")


        self.myscheme.new_link(reader1, "Data", Text_Extract_Summary, "Train Data")
#         self.myscheme.new_link(reader2, "Data", Text_Extract_Summary, "Test Data")
        self.myscheme.new_link(Text_Extract_Summary, "News", outputwriter, "Data")
