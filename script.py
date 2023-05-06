# 加载基础package
import datetime
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# 记录文件路径
residual_and_beta_folder_address = os.path.join(*[os.getcwd(), "data", "residual_and_beta"])
monitor_info_folder_address = os.path.join(*[os.getcwd(), "data", "monitor_info"])
extra_info_folder_address = os.path.join(*[os.getcwd(), "data", "extra_info"])
volume_feature_folder_address = os.path.join(*[os.getcwd(), "data", "volume_feature"])

# 取消warning输出
import warnings
warnings.filterwarnings("ignore")

#%%

# 准备数据
def prepare_data(nlags):

    # 读取作为y_to_predict的residual数据（需要之后手动shift下）
    residual_to_predict = pd.read_csv(os.path.join(residual_and_beta_folder_address, "one_factor_residual.csv"))
    residual_to_predict = residual_to_predict.set_index("date_time")
    
    # 加载停盘信息（注意，这里加载的停盘信息是全日停盘，不是涨停板停盘）
    monitor_info_df = pd.read_csv(os.path.join(monitor_info_folder_address, "stop_trading_df.csv"))
    monitor_info_df = monitor_info_df.set_index("date_time")
    
    # 对每一支股票进行处理
    all_y_and_X = {}
    all_available_stocks = os.listdir(volume_feature_folder_address)
    for each_stock in all_available_stocks:
        print("正在处理股票数据：" + each_stock)
        
        # ————————————————————————————————————————————————————————————————————————————————————
        
        # 读取volume_feature文件（在构建features的时候已经有过应有的shift，故这里无需shift）
        temp_df = pd.read_csv(os.path.join(volume_feature_folder_address, each_stock))
        temp_df = temp_df.set_index("date_time")
        
        ###
        # 这里有一个潜在的问题没有考虑到：（该问题在模拟盘中也尚未处理）
        # 由于股票的基本面数据每三个月更新一次，在更新的日期会有一些基本面feature发生非常大的变化
        # 比如该季度盈利由正转负，则pe会立即由正转负，随之而来的是该feature在一天内的突变
        # 这种情况很可能会极大地影响我们的策略表现（但方向并不明确），所以需要注意一下
        ###
        
        # =====================================================================================

        # 在数据表中添加y进去（注意这里的y需要进行shift）
        y_to_predict_series = residual_to_predict.shift(-1).loc[temp_df.index, each_stock.split(".")[0]]
        y_to_predict_series.name = "y_to_predict"
        temp_df = pd.concat([y_to_predict_series, temp_df], axis = 1)
        
        # 添加residual的lag_terms
        for lag_step in range(1, nlags + 1):
            temp_df["y_lag_" + str(lag_step)] = temp_df["y_to_predict"].shift(lag_step)
            
        # 构建过去12天的period_to_period_residual的均值作为新的features
        temp_df["rolling_mean_12"] = temp_df["y_to_predict"].rolling(12 + 1).apply(lambda x: x[:-1].mean())
        
        # fundamental数据中，如果turnoverRatio为0，则该日一定为停盘
        for each_column in temp_df.columns:
            if "turnoverRatio" in each_column:
                temp_df[each_column] = temp_df[each_column].replace(0, np.nan)  
        
        # =====================================================================================

        # 储存数据
        all_y_and_X[each_stock.split(".")[0]] = temp_df
    
    # 返回结果
    return all_y_and_X

#%%

# 进行rolling-window测试
def rolling_window_test(dataset_dict, date_index, train_length, test_length, predicted_return_threshold, holding_period):

    '''
    # 读取market_return数据文件
    # 在日期t的下午15:00:00才能够获得该数据，故open-to-open交易需要进行shift
    market_return_series = pd.read_csv(os.path.join(extra_info_folder_address, "stock_and_market_close_to_close_return.csv"))
    market_return_series = market_return_series.set_index("date_time")
    market_return_series = market_return_series["market_return"]
    market_return_series = market_return_series.shift().dropna()
    '''
    
    # 分割windows
    windows_list = []
    time_length = date_index.shape[0]
    test_end_list = list(range(time_length, 0, -test_length))
    test_end_list.sort()
    total_test_set_indexes = pd.Index([], name = "date")
    for each_end in test_end_list:
        if each_end >= train_length + test_length + holding_period - 1:
            full_date_index = date_index[each_end - (train_length + test_length + holding_period - 1):each_end]
            train_set_index = full_date_index[0:train_length]
            test_set_index = full_date_index[train_length + holding_period - 1:train_length + test_length + holding_period - 1]
            windows_list.append({"train_set_index": train_set_index, "test_set_index": test_set_index})
            total_test_set_indexes = total_test_set_indexes.append(test_set_index)
    
    # 设置一个window的train set dropout_rate，开始处理每一个window
    window_dropout_rate = 0.3
    decision_df = pd.DataFrame([], index = total_test_set_indexes, columns = dataset_dict.keys())
    return_df = pd.DataFrame([], index = total_test_set_indexes, columns = dataset_dict.keys())
    count = 0
    for each_window in windows_list:
        count += 1
        print("正在计算第" + str(count) + "/" + str(len(windows_list)) + "个window的结果！")
        # 使用Panel OLS进行测试
        
        # Parameter Sharing，使用全部股票的数据进行训练
        train_df = pd.DataFrame()
        for each_stock in dataset_dict:
            
            # 选出对应的时间段的数据，并添加market_return数据作为一列features
            # 注意，这里对于最后一天的y，需要使用point_to_period_y_to_predict
            # 其他时刻使用period_to_period_y_to_predict就好
            temp_df = dataset_dict[each_stock]
            temp_df = temp_df.loc[each_window["train_set_index"]]
            '''
            temp_df["market_return"] = market_return_series
            '''
            
            ###
            # 在这里如果想要更加真实地模拟放在模拟盘上的代码，
            # 需要手动移除数据集的最后两行数据，与起始的十二行数据
            temp_df = temp_df.iloc[12:]
            temp_df = temp_df.iloc[:-2]
            ###
            
            # 设计panel_model的输入数据的标准格式
            # The index on the time dimension must be either numeric or date-like
            temp_df = temp_df.dropna()
            temp_df["date"] = temp_df.index
            temp_df["date"] = temp_df["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
            temp_df["stock_code"] = each_stock
            temp_df = temp_df.set_index(["stock_code", "date"])
            train_df = train_df.append(temp_df)
            
        # 判断是否触及dropout条件，如果触发了该条件则可以直接跳过该window
        if train_df.shape[0] / (len(dataset_dict) * len(each_window["train_set_index"])) < 1 - window_dropout_rate:
            continue

        # 进行panel_model回归测试
        y_train = train_df["y_to_predict"]
        x_train = train_df.copy()
        del x_train["y_to_predict"]
        x_train = sm.add_constant(x_train, has_constant = "add")
        model = PanelOLS(y_train, x_train, entity_effects = True)
        model = model.fit()
        
        # 对于每一只股票进行交易判断
        for each_stock in dataset_dict:
            
            # 取出测试集并去除空值
            test_df = dataset_dict[each_stock].loc[each_window["test_set_index"]]
            test_df = test_df.dropna()
            
            # 判断是否有数据剩余，如果没有数据剩余则可以直接跳过该股票的该window
            if test_df.shape[0] == 0:
                continue
            
            # 添加market_return和constant，提取出true_return
            y_test = test_df["y_to_predict"]
            x_test = test_df.copy()
            del x_test["y_to_predict"]
            '''
            x_test["market_return"] = market_return_series
            '''
            x_test = sm.add_constant(x_test, has_constant = "add")

            # 设置index格式
            x_test["date"] = x_test.index
            x_test["date"] = x_test["date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
            x_test["stock_code"] = each_stock
            x_test = x_test.set_index(["stock_code", "date"])
            
            # 用训练好的模型进行交易判断
            predicted_return = model.predict(x_test)
            predicted_return = pd.Series(predicted_return.values[:, 0], index = y_test.index)
            decision = predicted_return.map(lambda x: 1 if x > predicted_return_threshold else 0)
            
            ###
            # 可以在这里调整decision的计算方法，从而对比两个阈值的表现
            # 比如将decision改为 x>0.004 & x<=0.005，从而对比0.004和0.005的表现
            # decision = predicted_return.map(lambda x: 1 if x > predicted_return_threshold - 0.001 and x <= predicted_return_threshold else 0)
            ###
            
            y_return = y_test.copy()
            y_return = y_return * decision
        
            # 记录结果
            decision_df.loc[decision.index, each_stock] = decision
            return_df.loc[y_return.index, each_stock] = y_return

    # 返回结果
    return decision_df, return_df

#%%

# 主函数
def main():
    
    # 设置本次回测的一些超参数
    train_length = 1000
    test_length = 30
    holding_period = 2
    features_lag = 5
    predicted_return_threshold = 0.005
    
    # 获取全部数据集
    print("开始加载数据集！")
    all_y_and_X = prepare_data(nlags = features_lag)
    
    # 调整每一支股票的y_to_predict
    print("正在对股票进行进一步处理......")
    for each_stock in all_y_and_X:
        temp_df = all_y_and_X[each_stock]
        
        # 在这里将y_to_predict转化成多天持仓的结果
        if holding_period > 1:
            y_to_predict_series = temp_df["y_to_predict"].copy()
            for i in range(1, holding_period):
                y_to_predict_series = (1 + y_to_predict_series) * (1 + temp_df["y_to_predict"].shift(-i)) - 1
            temp_df["y_to_predict"] = y_to_predict_series
        
        # 移除末尾和起始的无用的数据，并储存结果
        # temp_df = temp_df.iloc[:-holding_period]
        # temp_df = temp_df.iloc[12:]
        all_y_and_X[each_stock] = temp_df

    # 提取一个完整的日期，用于后续parameter_sharing时分割window
    complete_index = all_y_and_X[list(all_y_and_X.keys())[0]].index
    
    # 开始测试
    print("即将开始测试！")
    decision_df, return_df = rolling_window_test(all_y_and_X, complete_index, train_length, test_length, predicted_return_threshold, holding_period)

    # 整理结果
    decision_df = decision_df.fillna(0)
    return_df = return_df.fillna(0)
    
    # 输出一些中间结果到csv文件中
    decision_df.to_csv(special_name + "_decision_df.csv")
    return_df.to_csv(special_name + "_return_df.csv")
    
    '''
    # 读取csv文件
    decision_df = pd.read_csv(special_name + "_decision_df.csv")
    decision_df = decision_df.set_index("Unnamed: 0")
    decision_df.index.name = "date_time"
    return_df = pd.read_csv(special_name + "_return_df.csv")
    return_df = return_df.set_index("Unnamed: 0")
    return_df.index.name = "date_time"
    
    # 读取raw_return文件
    raw_return_df = pd.read_csv(os.path.join(extra_info_folder_address, "stock_and_market_close_to_close_return.csv"))
    raw_return_df = raw_return_df.set_index("date_time")
    # 由于在日期t早上判断是否交易，这里需要把raw_return进行shift操作，以便观察日期t-1的回报率情况进行后续分析
    raw_return_df = raw_return_df.shift()
    
    # 读取涨停板文件
    temp_stop_trading_df = pd.read_csv(os.path.join(monitor_info_folder_address, "temporary_stop_trading_df.csv"))
    temp_stop_trading_df = temp_stop_trading_df.set_index("date_time")
    
    # 分析回报率情况
    return_df_rule_out = return_df.copy()
    decision_df_rule_out = decision_df.copy()
    return_df_yest = return_df.copy() * 0
    decision_df_yest = decision_df.copy() * 0
    total_stock_traded_count = decision_df_rule_out.sum(axis = 1)

    # 这里先计算一下没有移除涨停板的回报率情况
    portfolio_return_df = return_df_rule_out.sum(axis = 1).div(total_stock_traded_count.apply(lambda x: max(x, 3) if x != 0 else 0), axis = "rows")
    portfolio_return_df = portfolio_return_df.fillna(0)
    portfolio_return_df = portfolio_return_df / holding_period
    print("daily mean return and num of stocks traded: ", np.mean(portfolio_return_df), np.mean(total_stock_traded_count))

    # 接下来移除涨停板的信息，并且添加一些其他约束
    all_stocks_sector = return_df_rule_out.columns.values
    time_list = list(return_df_rule_out.index)
    for each_stock in all_stocks_sector:
        for i in range(len(time_list)):
            time_temp = time_list[i]
            if decision_df_rule_out.loc[time_temp, each_stock] == 1:
                if temp_stop_trading_df.loc[time_temp, each_stock] == 1:
                    return_df_rule_out.loc[time_temp, each_stock] = 0
                    decision_df_rule_out.loc[time_temp, each_stock] = 0
                if (raw_return_df[raw_return_df.index == time_temp][each_stock][0] > 0.06 or raw_return_df[raw_return_df.index == time_temp][each_stock][0] < -0.05):
                    return_df_rule_out.loc[time_temp, each_stock] = 0
                    decision_df_rule_out.loc[time_temp, each_stock] = 0
    total_stock_traded_count = decision_df_rule_out.sum(axis = 1)

    # 计算移除涨停板的回报率情况
    portfolio_return_df = return_df_rule_out.sum(axis = 1).div(total_stock_traded_count.apply(lambda x: max(x, 3) if x != 0 else 0), axis = "rows")
    portfolio_return_df = portfolio_return_df.fillna(0)
    portfolio_return_df = portfolio_return_df / holding_period
    print("daily mean return and num of stocks traded: ", np.mean(portfolio_return_df), np.mean(total_stock_traded_count), np.sqrt(250) * np.mean(portfolio_return_df) / np.std(portfolio_return_df))
    
    # 画图
    p1 = portfolio_return_df.cumsum()
    p1.plot()
    
    # 输出每日交易决策及其相应回报率，以及cumulative_return
    result_df = pd.DataFrame([], index = decision_df_rule_out.index)
    result_df["cumu_return"] = portfolio_return_df.cumsum()
    result_df["portfolio"] = np.nan
    
    for each_row in decision_df_rule_out.iterrows():
        portfolio_dict = {}
        for each_stock in each_row[1].index:
            if each_row[1][each_stock] == 1:
                specific_return = return_df_rule_out.loc[each_row[0], each_stock]
                portfolio_dict[each_stock] = specific_return
        result_df.loc[each_row[0], "portfolio"] = str(portfolio_dict)
    
    '''
    
#%%
   
# 主程序
if __name__ == "__main__":
    special_name = "test_panelModel_open2open_0.005_simulation"
    # main()
    pass