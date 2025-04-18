from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class metric():
    def __init__(self, args):
        self.args = args
        self.num_classes = args.num_classes
        self.model_name = self.args.modelName
    def get_all_metric(self, all_pred, all_true):

            multi_class_result_df = self.multi_class(all_pred, all_true)

            base_df = pd.DataFrame({
                'seed': [self.args.seed],
                'model_name': [self.model_name],
            })

            metrics_df = pd.concat([base_df, multi_class_result_df], axis=1)

            return metrics_df
        

    def plot_confusion_matrix(self, all_true, all_pre, class_names=None):
        """
        绘制混淆矩阵

        :param all_true: 真实标签 (list 或 numpy array)
        :param all_pre: 预测标签 (list 或 numpy array)
        :param class_names: 类别名称 (list, 可选)
        """
        # 计算混淆矩阵
        cm = confusion_matrix(all_true, all_pre)
        
        # 打印混淆矩阵
        print("Confusion Matrix:")
        print(cm)

        # # 可视化混淆矩阵
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        # disp.plot(cmap=plt.cm.Blues, values_format='d')
        # plt.title("Confusion Matrix")
        # plt.show()
        # plt.savefig("output.png", dpi=300)  # 保存为高分辨率 PNG 文件

    def multi_class(self, all_pred, all_true):

            # 计算评价指标
            report = classification_report(all_true, all_pred, output_dict=True, zero_division=0)

            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            macro_f1 = report['macro avg']['f1-score']
            accuracy = accuracy_score(all_true, all_pred)
            self.plot_confusion_matrix(all_pred, all_true)

            metrics_df = pd.DataFrame({
                'Precision': [precision],
                'Recall': [recall],
                'F1-Score': [macro_f1],
                'Accuracy': [accuracy]
            })

            return metrics_df.round(4)



class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"