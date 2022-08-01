import numpy as np
import pandas as pd
import pickle
import torch
from flask import Flask, request, render_template
import io
import plotly.graph_objects as go
import plotly.utils as p
import json
from binance.client import Client


api_key = 'EgOzpBslsLvGa4g2iSF9TR4CJIevxLZdygNwYUakavv5aGP8eGCGvW7iGCcvvAZ8'

api_secret = 'Do5zqOduTJCxIwp9V0BfFwxOk0ta9XkF6XsbZPxBJSm8Y6zVvjKjqmtVijgToHYE'

client = Client(api_key, api_secret)


def std_rush_order_feature(df_buy, time_freq, rolling_freq):
    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results


def avg_rush_order_feature(df_buy, time_freq, rolling_freq):
    df_buy = df_buy.groupby(df_buy.index).count()
    df_buy[df_buy == 1] = 0
    df_buy[df_buy > 1] = 1
    buy_volume = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].sum()
    buy_count = df_buy.groupby(pd.Grouper(freq=time_freq))['btc_volume'].count()
    buy_volume.drop(buy_volume[buy_count == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    return results


def std_trades_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].count()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    #print(rolling_diff)
    results = rolling_diff.pct_change()
    return results


def std_volume_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['btc_volume'].sum()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results


def avg_volume_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['btc_volume'].sum()
    buy_volume.drop(buy_volume[buy_volume == 0].index, inplace=True)
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).mean()
    results = rolling_diff.pct_change()
    return results


def std_price_feature(df_buy_rolling, rolling_freq):
    buy_volume = df_buy_rolling['price'].mean()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=rolling_freq).std()
    results = rolling_diff.pct_change()
    return results


def avg_price_feature(df_buy_rolling):
    buy_volume = df_buy_rolling['price'].mean()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=10).mean()
    results = rolling_diff.pct_change()
    return results


def avg_price_max(df_buy_rolling):
    buy_volume = df_buy_rolling['price'].max()
    buy_volume.dropna(inplace=True)
    rolling_diff = buy_volume.rolling(window=10).mean()
    results = rolling_diff.pct_change()
    return results


def chunks_time(df_buy_rolling):
    # compute any kind of aggregation
    buy_volume = df_buy_rolling['price'].max()
    buy_volume.dropna(inplace=True)
    #the index contains time info
    return buy_volume.index


def build_features(coin, time_freq, index):
    
    records=[]
    agg_trades = client.aggregate_trade_iter(symbol=coin+'BTC', start_str='1 days ago UTC')###'ADXBTC'
    a= (list(agg_trades))
    if len(a)==0:
        return "NO TRADES"#"No Trades"
    
    for l in a:
        if l['m']==True:
            A='buy'
        else:
            A='sell'
        records.append({
        'symbol':str(coin),
        'timestamp':l['T'],
        'side':str(A),
        'price':float(l['p']),
        'amount':float(l['q']),
        'btc_volume':float(l['p'])*float(l['q'])
        })
    #print(records)
    df = pd.DataFrame.from_records(records)
    
    
    df["time"] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
    df = df.reset_index().set_index('time')

    df_buy = df[df['side'] == 'buy']
    print(len(df_buy))
    rolling_freq=len(df_buy)//10
    df_buy_grouped = df_buy.groupby(pd.Grouper(freq=time_freq))

    date = chunks_time(df_buy_grouped)


    
    b=np.array(std_rush_order_feature(df_buy, time_freq, rolling_freq).values)
    b=b.reshape(-1,1)
    c=np.array(avg_rush_order_feature(df_buy, time_freq, rolling_freq).values)
    c=c.reshape(-1,1)
    d=np.array(std_trades_feature(df_buy_grouped, rolling_freq).values)
    d=d.reshape(-1,1)
    e=np.array(std_volume_feature(df_buy_grouped, rolling_freq).values)
    e=e.reshape(-1,1)
    f=np.array(avg_volume_feature(df_buy_grouped, rolling_freq).values)
    f=f.reshape(-1,1)
    g=np.array(std_price_feature(df_buy_grouped, rolling_freq).values)
    g=g.reshape(-1,1)
    h=np.array(avg_price_feature(df_buy_grouped).values)
    h=h.reshape(-1,1)
    i=np.array(avg_price_max(df_buy_grouped).values)
    i=i.reshape(-1,1)
    j=np.array(np.sin(2 * np.pi * date.hour/23))
    j=j.reshape(-1,1)
    k=np.array(np.cos(2 * np.pi * date.hour/23))
    k=k.reshape(-1,1)
    l=np.array(np.sin(2 * np.pi * date.minute / 59))
    l=l.reshape(-1,1)
    m=np.array(np.cos(2 * np.pi * date.minute / 59))
    m=m.reshape(-1,1)
    DATE=date.to_frame(index = False)
    print(DATE['time'][0])
    PLACEHOLDER_TIMEDELTA = pd.Timedelta(minutes=0)
    n=(DATE - DATE.shift(1)).fillna(PLACEHOLDER_TIMEDELTA)  #delta_minutes
    n = np.array(n.apply(lambda x: x.dt.total_seconds() / 60))
    n=n.reshape(-1,1)
    input_=np.hstack((b,c,d,e,f,g,h,i,j,k,l,m,n))
    return(input_[-1])
    
def predict(coin):
    out=build_features(coin=coin,time_freq='15S',index=0)
    #print(out)
    file = open('model.pkl', 'rb')  #'AnomalyTransfomerBasic'#10 epochs
    model = pickle.load(file)
    file.close()
    if isinstance(out, str): 
        return(out)
    else:
        out=out.reshape(-1,1,13) 
        pred=model(torch.from_numpy(out).float())
        return(pred.detach().numpy()[0][0]) 



app = Flask(__name__)

@app.route('/')
def home():       
    ###gradient bar
    #values 1.76 to -1.8 and threshold -0.02      
    Thrsh=(-1)*-0.02
    #Pred=(-1)*(result)  
    fg_score = 0
    fg_thrsh = (Thrsh+1.76)*(100/3.56)                              

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",#+delta
        value = fg_score,#fg_thrsh
        delta = {"reference": 0,"increasing":{"color": "red"},"decreasing":{"color": "green"}},
        gauge = {"axis": {"range": [0,100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0,10], "color": "#15e610",},
                    {"range": [10,20], "color": "#56dc00",},
                    {"range": [20,30], "color": "#75d200",},
                    {"range": [30,40], "color": "#9ebc00",},
                    {"range": [40,50], "color": "#adb000",},
                    {"range": [50,60], "color": "#c79700",},
                    {"range": [60,70], "color": "#e06c00",},
                    {"range": [70,80], "color": "#e84a00",},
                    {"range": [80,90], "color": "#ea3500",},
                    {"range": [90,100], "color": "#ea1717",}
                ],
                'threshold' : {'line': {'color': "#D2042D", 'width': 5}, 'thickness': 0.75, 'value': fg_thrsh},
                }))

    fig.update_layout(paper_bgcolor = "#fff", font = {"color": "black", "family": "sans-serif",})
    gauge_json = json.dumps(fig, cls=p.PlotlyJSONEncoder)
    detail="Details about the most recent trade will appear here"
  
    return render_template('home1.html',prediction="",url_=gauge_json,description="Details about the prediction will appear here",description_plot=detail)

@app.route('/result', methods = ['POST'])       
def result():
    Thr = -0.02
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(str, to_predict_list))
        coin=str(to_predict_list[0])
        trades = client.get_recent_trades(symbol=coin+'BTC')
        if len(trades)>0:
            price=trades[-1]['price']
            quantity=trades[-1]['qty']
            detail=f'     {float(quantity):.2f} quantity of {coin} traded at {price} price'
        else:    
            detail=f"     No recent trades for {coin}"
        result = predict(coin)
        if isinstance(result, str): 
            prediction='NO TRADES'#'No Trades'
            safety="safe"
            message="Not any pumping event predicted"
            print(prediction)
        elif result > Thr:
            prediction='NON PUMP'#'Non Pump'
            safety="safe"
            message="Not any pumping event predicted"
        else:
            prediction='PUMP'#'Pump'                        
            safety="not safe"
            message="Model predicts a possible pump"                        
        
        if prediction == 'NO TRADES':#'No Trades':

            ###gradient bar
            #values 1.76 to -1.8 and threshold -0.02
            Thrsh=(-1)*-0.02
            #Pred=(-1)*(result)  
            fg_score = 0
            fg_thrsh = (Thrsh+1.76)*(100/3.56)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",#+delta
                value = fg_score,   
                delta = {"reference": 0,"increasing":{"color": "red"},"decreasing":{"color": "green"}},
                gauge = {"axis": {"range": [0,100]},
                        "bar": {"color": "black"},
                        "steps": [
                            {"range": [0,10], "color": "#15e610",},
                            {"range": [10,20], "color": "#56dc00",},
                            {"range": [20,30], "color": "#75d200",},
                            {"range": [30,40], "color": "#9ebc00",},
                            {"range": [40,50], "color": "#adb000",},
                            {"range": [50,60], "color": "#c79700",},
                            {"range": [60,70], "color": "#e06c00",},
                            {"range": [70,80], "color": "#e84a00",},
                            {"range": [80,90], "color": "#ea3500",},
                            {"range": [90,100], "color": "#ea1717",}
                        ],
                        'threshold' : {'line': {'color': "#D2042D", 'width': 5}, 'thickness': 0.75, 'value': fg_thrsh},
                        }))

            fig.update_layout(paper_bgcolor = "#fff", font = {"color": "black", "family": "sans-serif",})
            gauge_json = json.dumps(fig, cls=p.PlotlyJSONEncoder)

  
        else:    

            print(result)

            ###gradient bar
            #values 1.76 to -1.8 and threshold -0.02
            Thrsh=(-1)*-0.02
            Pred=(-1)*(result)  
            fg_score = (Pred+1.76)*(100/3.56)
            fg_thrsh = (Thrsh+1.76)*(100/3.56)

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = fg_score,
                delta = {"reference": fg_thrsh,"increasing":{"color": "red"},"decreasing":{"color": "green"}},
                gauge = {"axis": {"range": [0,100]},
                        "bar": {"color": "black"},
                        "steps": [
                            {"range": [0,10], "color": "#15e610",},
                            {"range": [10,20], "color": "#56dc00",},
                            {"range": [20,30], "color": "#75d200",},
                            {"range": [30,40], "color": "#9ebc00",},
                            {"range": [40,50], "color": "#adb000",},
                            {"range": [50,60], "color": "#c79700",},
                            {"range": [60,70], "color": "#e06c00",},
                            {"range": [70,80], "color": "#e84a00",},
                            {"range": [80,90], "color": "#ea3500",},
                            {"range": [90,100], "color": "#ea1717",}
                        ],  
                        'threshold' : {'line': {'color': "#D2042D", 'width': 5}, 'thickness': 0.75, 'value': fg_thrsh},
                        }))

            fig.update_layout(paper_bgcolor = "#fff", font = {"color": "black", "family": "sans-serif",})
            gauge_json = json.dumps(fig, cls=p.PlotlyJSONEncoder)

        klines = client.get_historical_klines(coin+'BTC', Client.KLINE_INTERVAL_1MINUTE, "5 days ago UTC")#5days           
        klines=np.array(klines)                       
        if len(klines)==0:
            return render_template('home1.html',prediction=result,url_=gauge_json,description=f"No trades of {coin} have happen",description_plot=f"No trades of {coin} have happen")#f"{coin} is {safety} to buy.{message}"
        df = pd.DataFrame(klines, columns = ['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Ignore'])
        df=df[['Open time','Open','High','Low','Close','Volume']]
        df['Open time'] = pd.to_datetime(df['Open time'].astype(np.int64), unit='ms')
        df = df.set_index('Open time')  
        #print(df)                                                     
        #df['Open time'] = df['Open time'].map(mpdates.date2num)
        if df['Open'].values[0].split('.')[0]=='0':
            df['Open']=df.Open.str.split('.').str[-1]
            df['Open'] = df['Open'].astype(np.int64)
        else:               
            df['Open'] = df['Open'].astype(np.float64) 

        if df['High'].values[0].split('.')[0]=='0':
            df['High']=df.High.str.split('.').str[-1]
            df['High'] = df['High'].astype(np.int64)
        else:
            df['High'] = df['High'].astype(np.float64)

        if df['Low'].values[0].split('.')[0]=='0':
            df['Low']=df.Low.str.split('.').str[-1]
            df['Low'] = df['Low'].astype(np.int64)
        else:
            df['Low'] = df['Low'].astype(np.float64) 

        if df['Close'].values[0].split('.')[0]=='0':
            df['Close']=df.Close.str.split('.').str[-1]
            df['Close'] = df['Close'].astype(np.int64)
        else:
            df['Close'] = df['Close'].astype(np.float64)

        if df['Volume'].values[0].split('.')[0]=='0':
            df['Volume']=df.Volume.str.split('.').str[-1]
            df['Volume'] = df['Volume'].astype(np.int64)
        else:
            df['Volume'] = df['Volume'].astype(np.float64)
        buf = io.BytesIO()

        """mc = mpl.make_marketcolors(
                            up='tab:green',down='tab:red',
                            edge='lime',
                            wick={'up':'blue','down':'red'},
                            volume='tab:green',
                           )

        s  = mpl.make_mpf_style(marketcolors=mc)
        mpl.plot(
            df,
            type="candle",  
            savefig=buf,                
            figratio=(12,8),    
            style='yahoo'           
            )"""

        
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    low=df['Low'],
                    high=df['High'],
                    close=df['Close'],
                    open=df['Open'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )
            ]

        )
        fig.update_layout(height=700,yaxis_title='Price (x1e-8)',xaxis_title='Timeline')    
        plot_json = json.dumps(fig, cls=p.PlotlyJSONEncoder)

        return render_template('home1.html',prediction=prediction,url=plot_json,url_=gauge_json,description=f"{coin} is {safety} to buy.{message}. Model predicts pump in {fg_score:.2f} percent"
        ,description_plot=detail)     


@app.route('/uploaded/<filename>', methods=['GET', 'POST'])
def uploaded_image(filename):
    #img_seg = main(filename)
    return render_template("home1.html", name=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)                          


