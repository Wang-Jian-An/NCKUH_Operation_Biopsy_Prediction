import os, datetime, base64, io, time
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, html, Input, Output, State, dcc, callback, CeleryManager, DiskcacheManager, dash_table
import model_prediction
import data

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# if 'REDIS_URL' in os.environ:
#     # Use Redis & Celery if REDIS_URL set as an env variable
#     from celery import Celery
#     celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
#     background_callback_manager = CeleryManager(celery_app)

# else:
#     # Diskcache for non-production apps when developing locally
#     import diskcache
#     cache = diskcache.Cache("./cache")
#     background_callback_manager = DiskcacheManager(cache)
# , background_callback_manager=background_callback_manager

app = Dash(__name__, external_stylesheets = external_stylesheets)
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
    # background = True, 
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
            predData = pd.read_excel(io.BytesIO(decoded))
            # if 'csv' in filename:
            #     # Assume that the user uploaded a CSV file
            #     data.predData = pd.read_csv(
            #         io.StringIO(decoded.decode('utf-8')))
            # elif 'xls' in filename or "xlsx" in filename:
            #     # Assume that the user uploaded an excel file
            #     data.predData = pd.read_excel(io.BytesIO(decoded))
                # predData.to_excel("predData.xlsx", index = None)
            return [html.Br(), html.Center(["{}".format(predData.to_dict("list"))])]
        except:
            return [html.Br(), html.Center(["資料上傳失敗"])]

# 點選 Submit、Rest，分別顯示預測結果表格、空白
@callback(
    Output("prediction_result", "children", allow_duplicate = True), 
    Output("submit_button", "n_clicks"),
    Input("submit_button", "n_clicks"), 
    prevent_initial_call = True,
    # background = True, 
    running = [
        (
            Output("contents", "children", allow_duplicate = True), 
            [html.Br(), html.Center("資料預測中，請稍後")], 
            [html.Br(), html.Center("資料預測完成，以下為預測結果")]
        )
    ]
)
def click_submit_text(submit_n_click):
    if submit_n_click == 1:
        print("開始預測")
        print(data.predData)
        predResult = model_prediction.main_func(predData = data.predData)
        print(predResult)
        return [
            html.Br(), 
            html.Div([
                dash_table.DataTable(predResult.to_dict("records"), [{"name": i, "id": i} for i in predResult.columns])
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