import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.backend import clear_session

from model import make_model
from preprocessing import *

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def calculate_results(y_test, pred_test):
    print("AUROC: ", roc_auc_score(y_test, pred_test))
    print("AUPRC: ", average_precision_score(y_test, pred_test))
    print(classification_report(y_test, pred_test.round()))


if __name__ == '__main__':
    # tensorflow.compat.v1.disable_v2_behavior()
    first_time = [True, True, True, True, True]
    observation_point_list = [30, 60, 90, 120, 180, 270]
    prediction_point_shift = 1.5

    for observation_point in observation_point_list:
        df = pd.read_csv("data/qlik_aug_final.csv", low_memory=False)
        prediction_point = (observation_point * prediction_point_shift)
        # df = exclude_inappropriate_patients(df, 0, observation_point - 60, prediction_point, 1) #
        feature_df = filter_by_time_from_first_visit(df, 0, observation_point)
        label_df = filter_by_time_from_first_visit(df, 0, prediction_point)
        x_demo, x_temp, y = transform2seq(feature_df, label_df, df, prediction_point,
                                          list(range(0, observation_point + 1, 15)))

        # Normalization
        scaler = MinMaxScaler()
        scaler.fit(x_demo)
        x_demo = scaler.transform(x_demo)
        for k in range(x_temp.shape[2]):
            min = x_temp[:, :, k].min()
            max = x_temp[:, :, k].max()
            feature_range = max - min
            if feature_range != 0:
                x_temp[:, :, k] = (x_temp[:, :, k] - min) / feature_range

        # Count and print samples by their label
        unique, counts = np.unique(y[:, 0], return_counts=True)
        dict_attrition = dict(zip(unique, counts))
        attrition_sample_ratio = dict_attrition[0] / dict_attrition[1]
        print(dict_attrition)

        unique, counts = np.unique(y[:, 1], return_counts=True)
        dict_outcome = dict(zip(unique, counts))
        outcome_sample_ratio = dict_outcome[0] / dict_outcome[1]
        print(dict_outcome)

        class_weights = {'outcome': {0: 1, 1: outcome_sample_ratio, 2: 0, 3: 0},
                         'attrition': {0: 1, 1: 1}}

        # Split to train/test model
        k = 0
        for train_index, test_index in KFold(shuffle=True, random_state=73).split(x_demo):
            k = k + 1
            if k < 30:
                x_demo_train = x_demo[train_index, :]
                x_demo_test = x_demo[test_index, :]

                x_temp_train = x_temp[train_index, :]
                x_temp_test = x_temp[test_index, :, :]
                y_train = y[train_index]
                y_test = y[test_index]

                # pretraining
                model = make_model(x_demo.shape[1], x_temp.shape[2], x_temp.shape[1], tr_flag=True)
                if not first_time[k - 1]:
                    model.load_weights('./weights/model_weights_fold' + str(k) + '.h5')
                    first_time[k - 1] = False
                history = model.fit(x=[x_demo_train, x_temp_train], y=[y_train[:, 0], y_train[:, 1]],
                                    validation_data=([x_demo_test, x_temp_test], [y_test[:, 0], y_test[:, 1]]),
                                    epochs=50, verbose=0, batch_size=64,
                                    class_weight=class_weights)
                model.save_weights('./weights/model_weights_fold' + str(k) + '.h5')

                # train *** fine tune
                model = make_model(x_demo.shape[1], x_temp.shape[2], x_temp.shape[1], tr_flag=False)
                model.load_weights('./weights/model_weights_fold' + str(k) + '.h5')
                history = model.fit(x=[x_demo_train, x_temp_train], y=[y_train[:, 0], y_train[:, 1]],
                                    validation_data=([x_demo_test, x_temp_test], [y_test[:, 0], y_test[:, 1]]),
                                    epochs=10, verbose=0, batch_size=128,
                                    class_weight=class_weights)


                model.save_weights('./weights/model_weights_fold' + str(observation_point) + "----" + str(k) + 'finalllle.h5')
                pred_test = model.predict([x_demo_test, x_temp_test])
                pred_test_binary_0 = np.where(pred_test[0] > 0.5, True, False)
                pred_test_binary_1 = np.where(pred_test[1] > 0.5, True, False)
                y_test_binary = y_test.astype(bool)
                out_str = str(observation_point) + " - " + str(k) + ","

                p, r, f, s = precision_recall_fscore_support(y_test_binary[:, 0], pred_test_binary_0, average='binary')
                out_str += str(roc_auc_score(y_test[:, 0], pred_test[0])) + ","  # AUROC
                out_str += str(average_precision_score(y_test[:, 0], pred_test[0])) + ","  # AUPRC
                out_str += str(p) + ","  # Precision
                out_str += str(r) + ","  # Recall
                out_str += str(f) + ","  # F1

                indexes = np.where(y_test[:, 1].reshape(y_test[:, 1].shape[0]) < 2)[0]
                new_label = y_test[:, 1][indexes]
                new_label_binary = y_test[:, 1][indexes].astype(bool)
                new_pred = pred_test[1][indexes]
                new_pred_binary = pred_test_binary_1[indexes]
                p, r, f, s = precision_recall_fscore_support(new_label_binary, new_pred_binary, average='binary')
                out_str += str(roc_auc_score(new_label, new_pred)) + ","  # AUROC
                out_str += str(average_precision_score(new_label, new_pred)) + ","  # AUPRC
                out_str += str(p) + ","  # Precision
                out_str += str(r) + ","  # Recall
                out_str += str(f)  # F1
                print(out_str)
        clear_session()