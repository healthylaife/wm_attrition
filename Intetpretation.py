import pandas as pd
import shap
import tensorflow
from matplotlib import pyplot as plt

from model import make_model
from preprocessing import *
from sklearn.model_selection import KFold

if __name__ == '__main__':
    tensorflow.compat.v1.disable_v2_behavior()
    first_time = [True, True, True, True, True]
    observation_point_list = [30, 60, 90, 120, 180, 270]
    prediction_point_shift = 1.5


    for observation_point in observation_point_list:
        df = pd.read_csv("data/qlik_aug_final.csv", low_memory=False)
        prediction_point = (observation_point * prediction_point_shift)
        # df = exclude_inappropriate_patients(df, 0, observation_point - 60, prediction_point, 1)
        feature_df = filter_by_time_from_first_visit(df, 0, observation_point)
        label_df = filter_by_time_from_first_visit(df, 0, prediction_point)
        x_demo, x_temp, y = transform2seq(feature_df, label_df, df, prediction_point,
                                          list(range(0, observation_point + 1, 15)))

        #x_demo = x_demo[y[:, 1] < 2]
        #x_temp = x_temp[y[:, 1] < 2]


        k = 0
        for train_index, test_index in KFold(shuffle=True, random_state=73).split(x_demo):
            k += 1

            if k == 3:
                x_demo_train = x_demo[train_index, :]
                x_demo_test = x_demo[test_index, :]
                x_temp_train = x_temp[train_index, :]
                x_temp_test = x_temp[test_index, :, :]
                y_train = y[train_index]
                y_test = y[test_index]

                # train
                model = make_model(x_demo.shape[1], x_temp.shape[2], x_temp.shape[1], tr_flag=True)
                model.load_weights('./weights/model_weights_fold' + str(observation_point) + "----" + str(k) + 'finalllle.h5', by_name = True)

                explainer = shap.DeepExplainer(model, [x_demo, x_temp])
                shap_values = explainer.shap_values([x_demo, x_temp])

                np.savetxt('aSHAP_value-outcome-demo-' + str(observation_point) + '-' + str(k) + '.txt',
                           np.mean(np.abs(shap_values[0][0]), axis=0), fmt="%5.5f")

                np.savetxt('aSHAP_value-outcome-tempo-' + str(observation_point) + '-' + str(k) + '.txt',
                           np.sum(np.mean(np.abs(shap_values[0][1]), axis=0), axis=0), fmt="%5.5f")

                print(str(k) + " finished")
