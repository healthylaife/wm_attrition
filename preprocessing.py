import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

PEDSNET_FILE = "data/qs_pedsnet_data_20210115.csv"
QLIK_FILE = "data/obesity_psc17_20210114.csv"
QLIK_AUGMENTED_FILE = "data/Qlik_augmented.csv"

NAN_VALUE = 0


def filter_pedsnet(df):
    df = df[df['fact'].str.contains("Body mass index ") | df['fact'].str.contains("Body height Measured") | df[
        'fact'].str.contains("BMI for age z score NHANES 2000")]
    return df


def find_in_pedsnet(df, visit_occurrence_id):
    height = ""
    bmi_percentile = ""
    bmi_ratio = ""
    NHANES = ""

    target_df = df[df['visit_occurrence_id'] == visit_occurrence_id]

    if len(target_df[target_df['fact'] == 'Body height Measured']) > 0:
        height = target_df[target_df['fact'] == 'Body height Measured'].iat[0, 11]

    if len(target_df[target_df['fact'] == 'Body mass index (BMI) [Percentile] Per age and sex']) > 0:
        bmi_percentile = target_df[target_df['fact'] == 'Body mass index (BMI) [Percentile] Per age and sex'].iat[0, 11]

    if len(target_df[target_df['fact'] == 'Body mass index (BMI) [Ratio]']) > 0:
        bmi_ratio = target_df[target_df['fact'] == 'Body mass index (BMI) [Ratio]'].iat[0, 11]

    if len(target_df[target_df['fact'] == 'BMI for age z score NHANES 2000']) > 0:
        NHANES = target_df[target_df['fact'] == 'BMI for age z score NHANES 2000'].iat[0, 11]

    return height, bmi_percentile, bmi_ratio, NHANES


def add_more_info_to_qlik(qlik_df, pedsnet_df):
    qlik_df["height"] = ""
    qlik_df["bmi_percentile"] = ""
    qlik_df["bmi_ratio"] = ""
    qlik_df["NHANES"] = ""

    counter = 0
    for index, row in qlik_df.iterrows():
        height, bmi_percentile, bmi_ratio, NHANES = find_in_pedsnet(pedsnet_df, row["visit_occurrence_id"])
        qlik_df._set_value(index, "height", height)
        qlik_df._set_value(index, "bmi_percentile", bmi_percentile)
        qlik_df._set_value(index, "bmi_ratio", bmi_ratio)
        qlik_df._set_value(index, "NHANES", NHANES)
        counter += 1

    return qlik_df


def fill_na(df):
    df["buying_food"] = df["buying_food"].fillna("Don't know")
    df["food_did_not_last"] = df["food_did_not_last"].fillna("Don't know")
    df["race"] = df["race"].fillna("Refused")
    df["ethnic_group"] = df["ethnic_group"].fillna("Refused")

    default_value = np.nan
    df["lifestyle_sleep_score"] = df["lifestyle_sleep_score"].replace('Incomplete', default_value)
    df["lifestyle_activity_score"] = df["lifestyle_activity_score"].replace('Incomplete', default_value)
    df["lifestyle_nutrition_score"] = df["lifestyle_nutrition_score"].replace('Incomplete', default_value)
    df["lifestyle_behavior_score"] = df["lifestyle_behavior_score"].replace('Incomplete', default_value)
    df["lifestyle_total_score"] = df["lifestyle_total_score"].replace('Incomplete', default_value)

    df["psc17_externalizing_subscale"] = df["psc17_externalizing_subscale"].replace('Incomplete', default_value)
    df["psc17_internalizing_subscale"] = df["psc17_internalizing_subscale"].replace('Incomplete', default_value)
    df["psc17_attention_subscale"] = df["psc17_attention_subscale"].replace('Incomplete', default_value)
    df["psc17_total_score"] = df["psc17_total_score"].replace('Incomplete', default_value)
    return df


def filter_based_initial_visit(df, count=1):
    temp_df = df.copy()
    new_visit_type = ["visit_type_WEIGHT MANAGEMENT NEW PATIENT", "visit_type_TELEMED NP PCP EXTERNAL"]
    visit_type_df = pd.get_dummies(df["visit_type"], prefix='visit_type')
    for col in list(visit_type_df.columns):
        if col in new_visit_type:
            temp_df[col] = visit_type_df[col]
    temp_df["new_visit_count"] = temp_df["visit_type_WEIGHT MANAGEMENT NEW PATIENT"] + temp_df[
        "visit_type_TELEMED NP PCP EXTERNAL"]

    idx = temp_df.groupby("study_id").sum()["new_visit_count"].to_frame()
    idx = idx[idx["new_visit_count"] == count].index.values.tolist()
    return df[df["study_id"].isin(idx)]


def substitute_values(df):
    df['visit_type'] = df['visit_type'].replace(visit_dict)
    df['race'] = df['race'].replace(race_dict)
    df['ethnic_group'] = df['ethnic_group'].replace(ethnicity_dict)
    return df


def change_representation_type(df):
    df['visit_date'] = pd.to_datetime(df['visit_date'], format='%Y-%m-%d %H:%M:%S')

    dict = {'No': 0, 'Yes': 1}
    df['care_connect_visit'] = df['care_connect_visit'].replace(dict).astype(int)
    df['bariatric_visit'] = df['bariatric_visit'].replace(dict).astype(int)
    df['telehealth_visit'] = df['telehealth_visit'].replace(dict).astype(int)

    dict = {'Female': 0, 'Male': 1}
    df['sex'] = df['sex'].replace(dict).astype(int)

    dict = {'Never true': -1, "Don't know": -1, 'Not sure': -1, 'Sometimes true': 1, 'Often true': 1}
    df["buying_food"] = df["buying_food"].replace(dict).astype(int)
    df["food_did_not_last"] = df["food_did_not_last"].replace(dict).astype(int)

    race_df = pd.get_dummies(df["race"], prefix='race')
    for col in list(race_df.columns):
        df[col] = race_df[col]
    df.drop(["race"], axis=1, inplace=True)

    ethnic_df = pd.get_dummies(df["ethnic_group"], prefix='ethnic_group')
    for col in list(ethnic_df.columns):
        df[col] = ethnic_df[col]
    df.drop(["ethnic_group"], axis=1, inplace=True)

    diagnosis_code_df = pd.get_dummies(df["diagnosis_code"], prefix='diagnosis_code')
    for col in list(diagnosis_code_df.columns):
        df[col] = diagnosis_code_df[col]
    df.drop(["diagnosis_code"], axis=1, inplace=True)

    visit_type_df = pd.get_dummies(df["visit_type"], prefix='visit_type')
    for col in list(visit_type_df.columns):
        df[col] = visit_type_df[col]
    df.drop(["visit_type"], axis=1, inplace=True)

    df.drop(["department"], axis=1, inplace=True)

    df["lifestyle_sleep_score"] = df["lifestyle_sleep_score"].astype(float)
    df["lifestyle_activity_score"] = df["lifestyle_activity_score"].astype(float)
    df["lifestyle_nutrition_score"] = df["lifestyle_nutrition_score"].astype(float)
    df["lifestyle_behavior_score"] = df["lifestyle_behavior_score"].astype(float)
    df["lifestyle_total_score"] = df["lifestyle_total_score"].astype(float)

    df["psc17_externalizing_subscale"] = df["psc17_externalizing_subscale"].astype(float)
    df["psc17_internalizing_subscale"] = df["psc17_internalizing_subscale"].astype(float)
    df["psc17_attention_subscale"] = df["psc17_attention_subscale"].astype(float)
    df["psc17_total_score"] = df["psc17_total_score"].astype(float)

    return df


def filter_rare_cats(df, min=100):
    out_df = df.copy()
    dfg = df.sort_values(["study_id", 'visit_date']).groupby("study_id")

    for col in list(df.filter(regex='^diagnosis_code', axis=1).columns):
        if (dfg[col].sum() > 0).value_counts().get(True, 0) < min:
            out_df.drop([col], axis=1, inplace=True)

    return out_df


def add_passed_days_from_first_visit(df):
    df = df.sort_values(["study_id", 'visit_date'])
    first_time = df.groupby("study_id")['visit_date'].first().to_dict()

    df["delta_first2visit"] = ""
    for index, row in df.iterrows():
        df.at[index, "delta_first2visit"] = (row['visit_date'] - first_time[row['study_id']]).days
    df["delta_first2visit"] = df["delta_first2visit"].astype(int)
    return df


def filter_by_time_from_first_visit(in_df, min=0, max=30):
    df = in_df.copy()
    df = df[df["delta_first2visit"] >= min]
    df = df[df["delta_first2visit"] <= max]
    return df


def exclude_inappropriate_patients(df, ws, we, label, min_visit):
    patients = []
    dfg = df.sort_values(["study_id", 'visit_date']).groupby("study_id")
    if label == "bmi":
        for patient_id in df["study_id"].unique():
            count = 0
            previous_date = None
            for index, row in df[df["study_id"] == patient_id].iterrows():
                if ws <= row["delta_first2visit"] <= we and row["delta_first2visit"] != previous_date and not pd.isna(
                        row["bmi_percentile"]):
                    previous_date = row["delta_first2visit"]
                    count += 1
                    if count >= min_visit:
                        patients.append(patient_id)
                        break

    else:
        for key, value in dfg["delta_first2visit"].max().to_dict().items():
            if value >= we:
                patients.append(key)

    return df.loc[df["study_id"].isin(patients)]


def impute(df):
    df.replace('', np.nan)
    # column_list = df.columns.tolist()
    # imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    # imputer.fit(df)
    # df = imputer.transform(df)
    # df = pd.DataFrame(df, index=df[:, 0], columns=column_list)
    return df.fillna(0)


###############################################################################################

def add_features(np_data, time_df, patient_index, timebin_index, delta_bmi_percentile):
    counter = 0
    # demographic
    np_data[patient_index, timebin_index, counter] = time_df["sex"].mean()
    counter += 1
    for col in list(time_df.filter(regex='^race', axis=1).columns):
        np_data[patient_index, timebin_index, counter] = time_df[col].mean()
        counter += 1
    for col in list(time_df.filter(regex='^ethnic_group', axis=1).columns):
        np_data[patient_index, timebin_index, counter] = time_df[col].mean()
        counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["medicaid"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["private_insurance"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["buying_food"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["food_did_not_last"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["age_yrs"].mean()
    counter += 1
    split_index = counter
    # temporal - categorical
    np_data[patient_index, timebin_index, counter] = time_df["telehealth_visit"].sum()
    counter += 1

    np_data[patient_index, timebin_index, counter] = time_df["care_connect_visit"].sum()
    counter += 1

    for col in list(time_df.filter(regex='^visit_type', axis=1).columns):
        np_data[patient_index, timebin_index, counter] = time_df[col].sum()
        counter += 1

    for col in list(time_df.filter(regex='^diagnosis_code', axis=1).columns):
        np_data[patient_index, timebin_index, counter] = time_df[col].sum()
        counter += 1

    # temporal - continuous
    np_data[patient_index, timebin_index, counter] = time_df["delta_first2visit"].mean()
    counter += 1
    # np_data[patient_index, timebin_index, counter] = time_df["age_yrs"].mean()
    # counter += 1

    # np_data[patient_index, timebin_index, counter] = time_df["height"].mean()
    # counter += 1
    # np_data[patient_index, timebin_index, counter] = time_df["weight_in_kg"].mean()
    # counter += 1

    np_data[patient_index, timebin_index, counter] = time_df["lifestyle_sleep_score"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["lifestyle_nutrition_score"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["lifestyle_activity_score"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["lifestyle_behavior_score"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["lifestyle_total_score"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["psc17_externalizing_subscale"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["psc17_internalizing_subscale"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["psc17_attention_subscale"].mean()
    counter += 1
    np_data[patient_index, timebin_index, counter] = time_df["psc17_total_score"].mean()
    counter += 1
    # np_data[patient_index, timebin_index, counter] = time_df['bmi_ratio'].mean()
    # counter += 1
    np_data[patient_index, timebin_index, counter] = time_df['bmi_percentile'].mean()
    counter += 1

    return split_index, counter


def transform2seq(feature_df, label_df, whole_df, prediction_time, time_points):
    label_dfg = label_df.sort_values(["study_id", 'delta_first2visit']).groupby("study_id")
    whole_dfg = whole_df.sort_values(["study_id", 'delta_first2visit']).groupby("study_id")
    feature_dfg = feature_df.sort_values(["study_id", 'delta_first2visit']).groupby("study_id")

    delta_bmi_percentile = label_dfg['bmi_percentile'].last() - label_dfg['bmi_percentile'].first()
    delta_bmi_percentile_feature = feature_dfg['bmi_percentile'].last() - feature_dfg['bmi_percentile'].first()

    delta_first2visit = whole_dfg["delta_first2visit"].max()
    delta_first2visit_dict = whole_dfg["delta_first2visit"].apply(list).to_dict()

    patient_count = len(feature_df["study_id"].unique())
    timestep_count = len(time_points) - 1
    feature_count = 200

    np_x = np.zeros((patient_count, timestep_count, feature_count))
    np_y = np.zeros((patient_count, 3))

    patient_index = 0
    for study_id in feature_df["study_id"].unique():
        delta_bmi_percentile_feature_value = delta_bmi_percentile_feature[study_id]
        # print(patient_index)
        patient_df = feature_df[feature_df["study_id"] == study_id]
        for timebin_index in range(0, len(time_points) - 1):
            time_df = patient_df[patient_df["delta_first2visit"] < time_points[timebin_index + 1]]
            time_df = time_df[time_df["delta_first2visit"] >= time_points[timebin_index]]
            if time_df.shape[0] > 0:
                split_index, counter = add_features(np_x, time_df, patient_index, timebin_index,
                                                    delta_bmi_percentile_feature_value)

        np_y[patient_index, 0] = (delta_first2visit[study_id] < prediction_time)  # attrition
        np_y[patient_index, 1] = (delta_bmi_percentile[study_id] > 0)  # outcome
        np_y[patient_index, 2] = not any(
            element >= prediction_time - 15 and element <= prediction_time + 15 for element in
            delta_first2visit_dict[study_id])

        np_y[patient_index, 1] += np_y[patient_index, 2] * 2
        patient_index += 1

    np_y = np_y[:, 0:2]
    x_demo = np_x[:, 0, :split_index].reshape(np_x.shape[0], split_index)
    x_temp = np_x[:, :, split_index:counter]
    return np.nan_to_num(x_demo), np.nan_to_num(x_temp), np.nan_to_num(np_y)


###############################################################################################

visit_dict = {
    "WEIGHT MANAGEMENT FOLLOW UP": "Medical in-person",
    "WEIGHT MANAGEMENT NEW PATIENT": "Medical in-person",
    "FOLLOW UP WEIGHT MGMT": "Medical in-person",
    "WT MGMT FU < 5 YEARS": "Medical in-person",
    "WT MGMT NP < 5 YEARS": "Medical in-person",
    "COMPLEX NEW": "Medical in-person",
    "WEIGHT MGMT NEW DV": "Medical in-person",
    "COMPLEX FOLLOW UP": "Medical in-person",
    "NP IMPAIRED GLUCOSE TOLERANCE": "Medical in-person",
    "FOLLOW UP": "Medical in-person",

    "TELEMED FP PCP EXTERNAL": "Medical Telemedicine",
    "TELEMED FP REMOTE": "Medical Telemedicine",
    "TELEMED NP PCP EXTERNAL": "Medical Telemedicine",
    "TELEMED WGMT FP REMOTE": "Medical Telemedicine",
    "TELEMED WGT FP & NUT FP REMOTE": "Medical Telemedicine",
    "TELEMED NP REMOTE": "Medical Telemedicine",
    "TELEMED HOME FP": "Medical Telemedicine",
    "TELEMED PROVIDER FP REMOTE": "Medical Telemedicine",

    "CARECONNECT FP": "Medical Care Connect",
    "CARECONNECT NP": "Medical Care Connect",

    "NUTRITION WGMT FOLLOW UP": "Nutrition in-person",
    "NUTRITION WGMT NEW": "Nutrition in-person",
    "NUTRITION WEIGHT MANAGEMENT NEW PATIENT": "Nutrition in-person",
    "NUTRITION WEIGHT MANAGEMENT FOLLOW UP": "Nutrition in-person",
    "NUTRITION FOLLOW UP": "Nutrition in-person",
    "NUTRITION NEW PATIENT": "Nutrition in-person",

    "NUTRITION GROUP COOKING CLASS": "Nutrition cooking class",
    "NUTRITION WGMT GROUP": "Nutrition cooking class",

    "TELEMED NTR WGMT NP REMOTE": "Nutrition telemedicine",
    "TELEMED NTR WGMT FP REMOTE": "Nutrition telemedicine",

    "COUNSELING FOLLOW UP": "Psychology in-person",
    "COUNSELING NEW": "Psychology in-person",
    "COUNSELING": "Psychology in-person",

    "CARECONNECT COUNSELING FP": "Psychology Care Connect",
    "CARECONNECT COUNSELING NEW": "Psychology Care Connect",

    "BARIATRIC GROUP": "Psychology group",
    "GROUP VISIT": "Psychology group",

    "EXERCISE GROUP (AGES 9-12)": "Exercise group",
    "EXERCISE GROUP (AGES 13-18)": "Exercise group",
    "EXERCISE GROUP (AGES 5-8)": "Exercise group",

    "EXERCISE NEW": "Exercise counseling",
    "EXERCISE FOLLOW UP": "Exercise counseling",

    "WM PERSONAL TRAINING": "Exercise personal training",
    "PERSONAL TRAINING": "Exercise personal training",
    "TEEN PERSONAL TRAINING": "Exercise personal training",

    "DIABETES EDUCATION": "Diabetes",
    "FP IMPAIRED GLUCOSE TOLERANCE": "Diabetes",
    "DIABETES FOLLOW UP": "Diabetes",
    "DIABETES NEW PATIENT": "Diabetes",

    "ACTIVITY TRACKER": "ELIMINATE",
    "RESEARCH": "ELIMINATE",
}

race_dict = {

    'White or Caucasian': "White",
    'Some Other Race': "Other",
    'Black or African American': "Black",
    'Asian Indian': "Asian",
    'Hawaiian Native or Other Pacific Islander': "Asian",
    'Asian': "Asian",
    'Other Asian': "Asian",
    'Vietnamese': "Asian",
    'American Indian or Alaska Native': "Other",
    'Chinese': "Asian",
    'Information Not Available': "Unknown",
    'Refused': "Unknown",
    'Filipino': "Asian",
    'Guamanian or Chamorro': "Other",
    'Other Pacific Islander': "Other",
    'Japanese': "Other",
    'Native Hawaiian': "Other",
}

ethnicity_dict = {
    'Another Hispanic, Latino, or Spanish Origin': "Hispanic",
    'Mexican, Mexican American, Chicano/a': "Hispanic",
    'Non-Hispanic or Latino': "Non-Hispanic",
    'Puerto Rican': "Hispanic",
    'Refused': "Refused or NA",
    'Information Not Available': "Refused or NA",
}
