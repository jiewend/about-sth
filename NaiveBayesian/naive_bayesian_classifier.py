import json
import numpy
import pandas as pd
# https://www.cnblogs.com/leoo2sk/archive/2010/09/17/naive-bayesian-classifier.html
# https://www.ruanyifeng.com/blog/2013/12/naive_bayes_classifier.html

class NaiveBayesianCalssifier(object):
    '''
    朴素贝叶斯法
    '''
    def __init__(self, config=None):
        self.train_data = self.load_train_data()
        # 拉普拉斯平滑参数
        self.smooth = 0.1
    
    def load_train_data(self):
        df = pd.read_excel('./train_data.xlsx')
        train_data = [dict(it) for idx, it in df.iterrows()]
        return train_data
    
    def _caul_P_y_x(self, dic_x, df_train_data):
        feat_name  = dic_x['key']
        feat_value = dic_x['value']
        y_enum = sorted(list(set(list(df_train_data['y']))))
        train_data_part = [it for it in self.train_data if it[feat_name] == feat_value]
        P_y_if_x = {}
        for _y in y_enum:
            divid_top = len([it for it in train_data_part if it['y'] == _y]) + self.smooth
            divid_bottom = len(self.train_data) + self.smooth_bottom
            prob = divid_top / divid_bottom
            P_y_if_x[_y] = prob
        return P_y_if_x

    def fit(self):
        '''
        计算训练集中的条件概率
        '''
        # key = feat_name, value = dict(value=y, prop=num) 
        self.P_y_by_x = {}
        df_train_data = pd.DataFrame(self.train_data)
        self.smooth_bottom = sum([len(set(df_train_data[it])) for it in df_train_data.columns]) * self.smooth
        feat_names = [it for it in df_train_data.columns if it != 'y']
        for _fname in feat_names:
            feat_enum = sorted(list(set(list(df_train_data[_fname]))))
            for fv in feat_enum:
                dic_x = dict(key=_fname, value=fv)
                self.P_y_by_x[_fname] = self._caul_P_y_x(dic_x, df_train_data)

    def predict_by_single_feature(self, dic_x):
        feat_name  = dic_x['key']
        feat_value = dic_x['value']
        P_y_if_x = self.P_y_by_x[feat_name]
        for k, v in P_y_if_x.items():
            print('y:{}\tprob:{}'.format(k, v))
        y_hit = sorted(P_y_if_x.items(), key=lambda it:it[1])[-1]
        print('hit y : value {}, prob {}'.format(y_hit[0], y_hit[1]))

if __name__ == '__main__':
    nbc = NaiveBayesianCalssifier()
    nbc.fit()
    dic_x = dict(key='x1', value='s')
    nbc.predict_by_single_feature(dic_x)        
