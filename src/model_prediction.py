import os, pickle, gzip
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# Primary Key
primary_key = "序號"

# 腫瘤類別
target_Patho_class = "Patho_class"

# Thymoma分期
target_Mosaoka_number = "Mosaoka_number"

# 是否為Thymoma
target_Thymoma_or_not = "Thymoma_or_not"

# 是否為 Operation0_Biospy1
target_operation0_biopsy1 = "Operation0_Biopsy1"
GT_target_operation0_biopsy1 = "GT_operation0_biopsy1"

# T-stage 級數
t_stage = "T_stage"
target_t_stage = "T_stage_target"

# 主要特徵
main_target = GT_target_operation0_biopsy1
classical_features = ["Sex", "MG", "Pleural_effusion", "LAP", "LD_Category"]

# 要移除的特徵
remove_features = ["Subtype", "Patho", "Hospital", "Stage", "Hospital", "Real_operation0_biopsy1", "Wrong_decision", "Patho_Analysis"]

def KNN_Imputation_and_generate_column(data: pd.DataFrame,
                                       PK_feature: str, 
                                       numerical_features: list,
                                       classical_features: list, 
                                      load_Imputation_Information: str = None):
    
    # 先把連續變數、類別變數與目標分成三份資料
    PK_data = data[PK_feature].reset_index(drop = True)
    
    # 針對連續型變數進行遺失值處理
    if numerical_features.__len__() > 0:
        numerical_features_data = data[numerical_features]
        
        # 求出有遺失值的特徵
        missing_continuous_features = ["LD", "AFP", "HCG"]        
    
        # 產生「是否遺失值」特徵
        numerical_features_data = pd.concat([
            numerical_features_data, 
            pd.concat(
                [numerical_features_data[one_column].apply(lambda x: 1 if np.isnan(x) else 0).rename(f"{one_column}_missing_or_not") 
                 for one_column in missing_continuous_features], 
                axis = 1
            )
        ], axis = 1)

        with gzip.GzipFile(os.path.join("FE_obj", "KNNImputation_for_Continuous_{}.gzip".format(load_Imputation_Information)), "rb") as f:
            KNNImputation = pickle.load(f)
        knn_features = KNNImputation.feature_names_in_
        print(knn_features.__len__())
        # for i in [*numerical_features, *[f"{one_column}_missin_or_not" for one_column in missing_continuous_features]]:
        #     if i not in knn_features:
        #         print(i)

        numerical_features_data = pd.DataFrame(
            KNNImputation.transform(numerical_features_data[knn_features]), 
            columns = knn_features
        )

    if classical_features.__len__() > 0:    
        classical_features_data = data[classical_features]
        missing_classical_features = list(data[classical_features].columns[data[classical_features].isna().sum() > 0])

        if missing_classical_features.__len__() > 0:
            # 產生「是否遺失值」特徵
            classical_features_data = pd.concat([
                classical_features_data, 
                pd.concat(
                    [classical_features_data[one_column].apply(lambda x: 1 if np.isnan(x) else 0).rename(f"{one_column}_missing_or_not") 
                     for one_column in missing_classical_features], 
                    axis = 1
                )
            ], axis = 1)

            if os.path.exists(os.path.join("FE_obj", "SimpleImputation_for_Categorical_{}.gzip".format(load_Imputation_Information))):
                with gzip.GzipFile(os.path.join("FE_obj", "SimpleImputation_for_Categorical_{}.gzip".format(load_Imputation_Information)), "rb") as f:
                    SimpleImputation = pickle.load(f)
                simpleimpution_features = SimpleImputation.feature_names_in_
                classical_features_data = pd.DataFrame(
                    SimpleImputation.transform(classical_features_data[simpleimpution_features]), 
                    columns = simpleimpution_features
                )
            else:
                SimpleImputation = SimpleImputer(strategy = "most_frequent")
                classical_features_data = pd.DataFrame(
                    SimpleImputation.fit_transform(classical_features_data), 
                    columns = [*classical_features, *[f"{one_column}_missin_or_not" for one_column in missing_classical_features]]
                )

    # 補值後資料合併
    if "numerical_features_data" in locals() and "classical_features_data" in locals():
        data = pd.concat([
            PK_data, numerical_features_data, classical_features_data
        ], axis = 1)
    elif "classical_features_data" not in locals():
        data = pd.concat([
            PK_data, numerical_features_data
        ], axis = 1)
    return data

def main_func(predData):
    global classical_features
    predData["Sex"] = predData["Sex"].apply(lambda x: 1 if x == "M" else 0)
    predData["LD_Category"] = predData["LD"].apply(lambda x: 1 if x >= 135 and x <= 255 else 0)
    numerical_features = [i for i in list(predData.columns) if i not in remove_features+classical_features+[primary_key, target_Patho_class, target_Mosaoka_number, target_Thymoma_or_not, target_operation0_biopsy1, GT_target_operation0_biopsy1, t_stage, target_t_stage]]
    classical_features = [i for i in classical_features if i in predData.columns]

    # 讀取特徵工程物件、遺失值物件，以及模型檔案
    with gzip.GzipFile(os.path.join("final_model", "Lasso-0_None-None-None-CatBoost.gzip"), "rb") as f:
        model = pickle.load(f)
    model_inputFeatures = model.feature_names_

    # 填補遺失值
    if predData.isna().sum().sum() > 0:
        predData = KNN_Imputation_and_generate_column(
            data = predData,
            PK_feature = primary_key,
            numerical_features = numerical_features,
            classical_features = classical_features,
            load_Imputation_Information = "Lasso-0"
        )

    # 結果預測
    yhat = model.predict(predData[model_inputFeatures])
    yhat_proba = model.predict_proba(predData[model_inputFeatures])

    # 將 PK、年齡、性別以及預測結果轉換成 Data Frame
    predResult = {
        primary_key: predData[primary_key].tolist(),
        "Age": predData["Age"].tolist(),
        "Sex": predData["Sex"].apply(lambda x: "M" if x == 1 else "F").tolist(),
        "Predict": np.where(yhat == 0, "Operation", "Biopsy"), 
        "Operation_Proba": np.round(yhat_proba[:, 0], 4),
        "Biopsy_Proba": np.round(yhat_proba[:, 1], 4)
    }
    return pd.DataFrame(predResult)