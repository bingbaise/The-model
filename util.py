from matplotlib.pylab import RandomState
import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.base import TransformerMixin
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn import svm
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from keras.utils import to_categorical
from metric import getMATRIX

def toFloat(x):
    if type(x) == str:
        try:
            x = float(x)
        except:
            x = np.nan
    return float(x)

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
                 test_x, test_y, seed_num, Y_index) -> None:
        self.trainx, self.trainy, self.testx, self.testy = [
            train_x, train_y, test_x, test_y
        ]
        self.seed_num = seed_num
        self.param_gbdt = {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'loss': ['log_loss'],
            'max_depth': [4],
            'min_samples_split': [2],
            'min_samples_leaf': [3],
            'max_features': [3],
            'random_state': [self.seed_num]
        }
        self.Y_index = Y_index
        self.show= True
        self.roc =False
        self.metrix = False
        self.dict_auc = {}



    def print(self, result, jud=False):
        '''
        输出acc、precision、recall、f1
        获得混淆矩阵
        返回准确率
        '''
        if jud:
            print(f'acc:{{:.4}}'.format(
                sum(np.array(self.testy) == result) / len(result)))
            print(f"precision:{{:.4}}".format(precision_score(
                np.array(self.testy), result, average='micro')))
            print(f"recall:{{:.4}}".format(recall_score(
                np.array(self.testy), result, average='micro')))
            print(f"f1:{{:.4}}".format(f1_score(np.array(self.testy), result,
                                                average='micro')))
        return sum(np.array(self.testy) == result) / len(result)

    def FalseIndex(self, Y, result):
        index = np.where(result != Y)
        self.false_map_index = self.Y_index[index]
        print(self.false_map_index)
    

    def build_dnn_model(self, input_shape, num_classes):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def model_train(self, model_name, model):
        if model_name == 'DNN':
            num_classes = len(np.unique(self.trainy))
            y_train_cat = to_categorical(self.trainy, num_classes=num_classes)
            y_test_cat = to_categorical(self.testy, num_classes=num_classes)
            model = self.build_dnn_model(input_shape=self.trainx.shape[1], num_classes=num_classes)
            self.model_fit = model.fit(self.trainx, y_train_cat, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            result = np.argmax(model.predict(self.testx), axis=1)
            matrix = confusion_matrix(self.testy, result)
            matrix = getMATRIX(matrix)
        else:
            # 处理其他模型
            self.model_fit = model.fit(self.trainx, self.trainy)
            result = self.model_fit.predict(self.testx)
            matrix = confusion_matrix(self.testy, result)
            matrix = getMATRIX(matrix)
        
        return matrix

    # def model_train(self,model_name, model):
    #     if model_name=='lgb':
    #         self.trainx.columns = [str(i) for i in range(len(self.trainx.columns))]
    #     self.model_fit = model.fit(self.trainx, self.trainy)
    #     result = self.model_fit.predict(self.testx)
    #     matrix = confusion_matrix(self.testy, result)
    #     matrix = getMATRIX(matrix)
        
    #     # print('lgbP:',stats.ttest_ind(self.trainy, self.lgb_model.predict(self.testx)))
    #     # self.get_roc(self.testy, 
    #     #              self.model_fit.predict_proba(self.testx)[:,1],
    #     #              model_name)
        return matrix

    def DNN2roc(self, Y, Y_hat):
        self.get_roc(Y, Y_hat,'DNN')


    def get_roc(self, Y, Y_hat,model_name):
        fpr, tpr, thresholds = roc_curve(Y, Y_hat)
        self.dict_auc[model_name] = [fpr, tpr]

    def show_roc(self,):
        plt.figure(figsize=(10, 10))
        plt.title('ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], '--', lw=1, color = 'r')
        f_t_AUCs = sorted(self.dict_auc.items(), 
                          key=lambda item: auc(item[1][0],item[1][1]),
                          reverse=True)
        for i in f_t_AUCs:
            plt.plot(i[1][0], i[1][1], label=f'{i[0]}:AUC={auc(i[1][0], i[1][1]):.4f}')
        plt.legend()
        plt.savefig('output.png')
        plt.show()

    @classmethod
    def test_matric(cls, label, result):
        DL_matrix = confusion_matrix(np.array(label),np.array(result))
        sensitive, specificity,ppv,npv = [],[],[],[]
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
        print(len(np.where(np.array(result) == np.array(label))[0]))
        acc = len(np.where(np.array(result) == np.array(label))[0])/len(label)
        f1 = f1_score(label, result)
        print('sensitive',sensitive,'\nspecificity', specificity,'\nppv', ppv,'\nnpv', npv, '\nacc', acc,'\nf1', f1)


def get_k_fold_data(k, i, X: pd.DataFrame, y: pd.DataFrame, index: np.ndarray):
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


