import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn import svm
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import roc_curve, auc, confusion_matrix
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

class Transform():
    '''
    confuse 
    normalization
    drop
    delenan
    '''

    def __init__(self, seed_num=1) -> None:
        self.seed_num = seed_num

    def confuse(self, df):
        '''
        打乱x, y
        '''
        # tem_X, tem_Y = [], []
        # x = np.linspace(0, len(y)-1, len(y))
        # random.seed(self.seed_num)
        # random.shuffle(x)
        # for i in range(len(x)):
        #     tem_X.append(X.iloc[int(x[i]), :].tolist())
        #     tem_Y.append(y[int(x[i])])
        df = df.sample(
            frac=1, random_state=self.seed_num)
        self.confuse_index = np.array(df.index)
        return df.reset_index(drop=True)

    def normalization(self, X):
        '''
        归一化
        '''
        xmean = np.mean(X, axis=0)
        xstd = np.std(X, axis=0)
        # print(xmean, xstd)
        return (X-xmean)/xstd

    def drop(self, patient, dropColunm=['ESR', '轻链ka/λ', '免疫固定电泳', '蛋白电泳', '免疫分型', '尿蛋白'], startXIndex=4, YIndex=2):
        '''
        startXIndex选择X维度
        dropcolumn选择删除的维度
        '''
        X = patient.iloc[:, startXIndex:].drop(dropColunm, axis=1)
        y = patient.iloc[:, YIndex]
        return X, y

    def DeleNan(self, patient):
        '''
        删除多余nan行
        '''
        withNan = patient.isnull()
        lenP = withNan.iloc[:, 1].count() - withNan.iloc[:, 1].sum()
        patient = patient.iloc[:lenP, :]
        return patient


class PredictWay():
    def __init__(self, train_x, train_y,
                 test_x, test_y, keyvalue, seed_num, Y_index) -> None:
        self.trainx, self.trainy, self.testx, self.testy = [
            train_x, train_y, test_x, test_y
        ]
        self.seed_num = seed_num
        self.dict = {v: k for k, v in keyvalue.items()}
        self.param_gbdt = {
            'n_estimators': [100],
            # 'max_depth':8,
            # 'min_samples_split':5,
            'learning_rate': [0.1],
            'loss': ['log_loss'],
            'max_depth': [4],
            'min_samples_split': [2],
            'min_samples_leaf': [3],
            'max_features': [3],
            'random_state': [self.seed_num]
        }
        self.Y_index = Y_index

    def print(self, result, jud=False):
        self.matrix = confusion_matrix(self.testy, result)
        if jud:
            print(f'acc:{{:.4}}'.format(
                sum(np.array(self.testy) == result) / len(result)))
            print(f"precision:{{:.4}}".format(precision_score(
                np.array(self.testy), result, average='micro')))
            print(f"recall:{{:.4}}".format(recall_score(
                np.array(self.testy), result, average='micro')))
            print(f"f1:{{:.4}}".format(f1_score(np.array(self.testy), result,
                                                average='micro')))
            # print('auc:', auc(fpr,tpr))
            # print(fpr,tpr)
            # plt.plot(fpr,tpr)
        self.PrintPatient(self.testy, result, self.testx)
        return sum(np.array(self.testy) == result) / len(result)

    def PrintPatient(self, Y, result, testx):
        index = np.where(result != Y)
        self.false_map_index = self.Y_index[index]
        # false_map = [self.dict[i] for i in Y[index]]
        # true_map = [self.dict[i] for i in Y[np.where(result == Y)]]
        # print('错误样本标签：', false_map)
        # print('正确样本标签：', true_map)
        # print('准确率：', len(np.where(result == Y)[0])/len(Y))
        return len(np.where(result == Y)[0])/len(Y)

    def Svm(self,):
        self.svm = svm.SVC(random_state=self.seed_num).fit(
            self.trainx, self.trainy)
        # result = self.svm.predict_proba(self.testx)
        result = self.svm.predict(self.testx)
        # print(result)
        # self.svm_f, self.svm_t = self.get_roc(
        #     np.array(self.testy), result[:, 1])
        # self.print(result)
        return len(np.where(result == np.array(self.testy))[0])/len(self.testy)

    def lightgbm_cv(self,):
        data_train = lgb.Dataset(self.trainx, self.trainy, silent=True)
        return (lgb.cv(self.param_gbdt, data_train,
                       num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
                       metrics='binary_logloss',  verbose_eval=50, show_stdv=True, seed=self.seed_num))

    def GBDT(self, *param, **param_grid):
        GBDTreg = GradientBoostingClassifier(*param)
        GridSearch = GridSearchCV(GBDTreg, param_grid)
        self.gbdt = GridSearch.fit(self.trainx, self.trainy)
        result = self.gbdt.predict(self.testx)
        # print(GridSearch.best_params_)
        # print(GridSearch.best_score_)
        self.gbdt_f, self.gbdt_t = self.get_roc(
            np.array(self.testy), self.gbdt.decision_function(self.testx))
        self.print(result)
        return len(np.where(result == np.array(self.testy))[0])/len(self.testy)

    def decisionTree(self):
        dt_classifier = DecisionTreeClassifier(
            random_state=self.seed_num)
        self.decisiontree = dt_classifier.fit(self.trainx, self.trainy)
        result = self.decisiontree.predict(self.testx)
        self.dt_f, self.dt_t = self.get_roc(np.array(self.testy),
                                            self.decisiontree.predict_proba(self.testx)[:, 1])
        # # 可视化决策树的划分方式（文本形式）
        # tree_rules = export_text(
        #     dt_classifier, feature_names=self.trainx.columns.tolist())
        # # print(tree_rules)

        # # 可视化决策树的划分方式（图形形式）
        # plt.figure(figsize=(60, 40))
        # plot_tree(dt_classifier, feature_names=self.trainx.columns.tolist(),
        #           class_names=self.dict, filled=True, rounded=True)
        # plt.savefig('decisionTree.png')
        return len(np.where(result == np.array(self.testy))[0])/len(self.testy)

    def predict_test(self, test_x, test_y):
        decisiontree_result = self.decisiontree.predict(test_x)
        gbdt_result = self.gbdt.predict(test_x)
        svm_result = self.svm.predict(test_x)
        decisiontree_result = len(
            np.where(decisiontree_result == np.array(test_y))[0])/len(test_y)
        gbdt_result = len(
            np.where(gbdt_result == np.array(test_y))[0])/len(test_y)
        svm_result = len(
            np.where(svm_result == np.array(test_y))[0])/len(test_y)
        return {
            'decisiontree': decisiontree_result,
            'gbdt': gbdt_result,
            'svm': svm_result
        }

    def get_roc(self, Y, Y_hat):
        return None, None
        # print(Y, Y_hat)
        fpr, tpr, thresholds = roc_curve(Y, Y_hat)
        print('auc:', auc(fpr, tpr))
        return fpr, tpr

    def show_roc(self,):
        return
        plt.figure(figsize=(10, 10))
        plt.title('ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(self.dt_f, self.dt_t, '--*b', label="decitionTree")
        plt.plot(self.gbdt_f, self.gbdt_t, '-', label="GBDT")
        plt.plot(self.svm_f, self.svm_t, '-*', label="SVM")
        # plt.plot(self.dt_f, self.dt_t, '--*b', label="")
        plt.legend()
        plt.show()

def test_matric(ytest,result):
    DL_matrix = confusion_matrix(np.array(ytest),np.array(result))
    sensitive, specificity, ppv,npv=[],[],[],[]
    sum_T  =0
    for i in range(len(DL_matrix)):
        sum_T+=DL_matrix[i,i]
    for i in range(len(DL_matrix)):
        TP = DL_matrix[i,i]
        FN = sum(DL_matrix[i,:])-DL_matrix[i,i]
        FP = sum(DL_matrix[:, i]) - DL_matrix[i,i]
        TN = DL_matrix.sum()-TP-FP-FN
        sensitive.append(TP/(TP+FN))
        specificity.append(TN/(TN+FP))
        ppv.append(TP/(TP+FP))
        npv.append(TN/(TN+FN))
    print('sensitive:',sensitive)
    print('specificity:',specificity)
    print('ppv:',ppv)
    print('npv:',npv)
        # TP,FP,FN,TN = DL_matrix[0,0],DL_matrix[0,1],DL_matrix[1,0],DL_matrix[1,1]
        # P = TP/(TP+FP)
        # R = TP/(TP+FN)
        # F1 = 2*P*R/(P+R)
        # print('骨髓瘤患者：','P:',P,'R',R,'F1',F1)
        # P = TN/(TN+FN)
        # R = TN/(TN+FP)
        # F1 = 2*P*R/(P+R)
        # print('非骨髓瘤患者：','P:',P,'R',R,'F1',F1)

def get_k_fold_data(k, i, X: pd.DataFrame, y: pd.DataFrame, index: np.array):
    assert k > 1
    fold_size = len(X) // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        x_part, y_part = X.iloc[idx, :], y.iloc[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
            return_index = idx
        elif X_train is None:
            X_train, y_train = x_part, y_part
        else:
            X_train = pd.concat([X_train, x_part], axis=0)
            y_train = pd.concat([y_train, y_part], axis=0)
    return X_train, y_train, x_valid, y_valid, index[return_index]

