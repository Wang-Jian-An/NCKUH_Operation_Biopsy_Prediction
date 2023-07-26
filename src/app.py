import os, datetime, base64, io, time, pickle, gzip
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, html, Input, Output, State, dcc, callback, CeleryManager, DiskcacheManager, dash_table
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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)


app = Dash(__name__, external_stylesheets = external_stylesheets, background_callback_manager=background_callback_manager)
server = app.server
app.layout = html.Div([
    html.Center(html.H1("The prediction for operation or biopsy")), 
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            "將檔案拖移至此，或點擊此功能上傳檔案"
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    ), 
    html.Div([
        html.Button(children = "預測", id = "submit_button", n_clicks = 0, style = {"width": "50%"}), # 開始預測按鈕
        html.Button(children = "重置", id = "reset_button", n_clicks = 0,  style = {"width": "50%"}), # 重置按鈕
    ]), 
    html.Div(id = "contents", children = [""]),
    html.Div(id = "prediction_result"),
    html.Div(id = "rawData")
])

# 讀取檔案
@callback(
    Output('contents', 'children', allow_duplicate = True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"), 
    prevent_initial_call = 'initial_duplicate',
    background = True, 
    running = [
        (
            Output("contents", "children", allow_duplicate = True), 
            [html.Br(), html.Center(["資料上傳中，請稍後"])], 
            [html.Br(), html.Center([""])]
        )
    ]
)
def read_file(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                predData = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename or "xlsx" in filename:
                # Assume that the user uploaded an excel file
                predData = pd.read_excel(io.BytesIO(decoded))
            predData.to_excel("predData.xlsx", index = None)
            return [html.Br(), html.Center(["資料上傳成功"])]
        except:
            return [html.Br(), html.Center(["資料上傳失敗"])]

# 點選 Submit、Rest，分別顯示預測結果表格、空白
@callback(
    Output("contents", "children", allow_duplicate = True), 
    Output("prediction_result", "children", allow_duplicate = True), 
    Output("submit_button", "n_clicks"),
    Input("submit_button", "n_clicks"), 
    prevent_initial_call = True,
    background = True, 
    running = [
        (
            Output("contents", "children", allow_duplicate = True), 
            [html.Br(), html.Center("資料預測中，請稍後")], 
            [html.Br(), html.Center("")]
        )
    ]
)
def click_submit_text(submit_n_click):
    if submit_n_click == 1:
        try:
            predData = pd.read_excel("predData.xlsx")
            predResult = main_func(predData = predData)
            # return [html.Br(), html.Center("")], [
            #     html.Br(), 
            #     html.Div([
            #         dash_table.DataTable(predResult.to_dict("records"), [{"name": i, "id": i} for i in predResult.columns])
            #     ])
            # ], 0
            return [html.Br(), html.Center("")], [
                html.Br(), 
                html.Div([
                    "{}".format(predResult.to_dict("list"))
                ])
            ], 0
        except:
            return [
                html.Br(),
                html.Center("預測失敗，{}".format(os.getcwd()))
            ], [
                html.Br(), 
                html.Div([
                    dash_table.DataTable(predData.to_dict("records"), [{"name": i, "id": i} for i in predData.columns])
                ])
            ], 0

@callback(
    Output("contents", "children", allow_duplicate = True),
    Output("prediction_result", "children", allow_duplicate = True), 
    Output("reset_button", "n_clicks"),  
    Output("upload-data", "contents"), 
    Input("reset_button", "n_clicks"), 
    prevent_initial_call = True,
    running = [
        (Output("contents", "children"), "", "")
    ]
)
def click_submit_text(reset_n_click):
    global predData
    if reset_n_click == 1:
        predData = None
        return [], [], 0, None

if __name__ == "__main__":
    app.run_server(debug=True)