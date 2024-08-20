from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    data = pd.read_json('./VFINX.json')
    data = data.reset_index(names='Date')
    df_bonds_1 = pd.read_csv('./DGS10.csv')
    df_bonds_2 = pd.concat([pd.read_csv('./Прошлые данные доходности облигаций США 10-летние 86-05.csv')
                         ,pd.read_csv('./Прошлые данные доходности облигаций США 10-летние 05-21.csv')])

    df_bonds_1 = df_bonds_1.rename(columns={'DATE': 'Date'})
    df_bonds_1['Date'] = pd.to_datetime(df_bonds_1['Date'], format='%Y-%m-%d')
    df_bonds_1['DGS10'] = df_bonds_1['DGS10'].apply(lambda x: None if x == '.' else x).astype(float)

    df_bonds_2 = df_bonds_2[['Дата', 'Цена']].rename(columns={'Дата': 'Date', 'Цена': 'Bonds_income'})

    df_bonds_2['Date'] = pd.to_datetime(df_bonds_2['Date'], format='%d.%m.%Y')
    df_bonds_2['Bonds_income'] = round(df_bonds_2['Bonds_income'].str.replace(',', '.').astype(float), 2)

    merge_df = pd.merge(df_bonds_1, df_bonds_2, on='Date', how='left')
    merge_df['DGS10'] = merge_df['DGS10'].fillna(merge_df['Bonds_income'])

    data = pd.merge(data, merge_df.drop('Bonds_income', axis=1), how='left', on='Date')
    data['DGS10'] = data['DGS10'].ffill()
    data['Returns'] = ((data['Close'] + data['Dividends'].shift(1)) / data['Close'].shift(1)) - 1

    data.index = data.Date
    data = data.drop(['Open', 'High', 'VIX_Close', 'Date', 'Low', 'Volume', 'Stock Splits'], axis=1)

    return data


data = load_data()

class TradeSystem():
    def __init__(self, investment, data, start_date, end_date, model=None, in_fund=True, cash=0, returns=0):
        self.investment = investment
        self.data = data
        self.start_date, self.end_date = start_date, end_date
        self.current_port_date, self.current_date = start_date, start_date

        self.portfolio = pd.DataFrame(index=[start_date]
                                      , data={'Close_Port': float(investment)
                                              # Для простоты расчёта инециируем стоимость портфеля с закрытия.
                , 'Returns': float(returns)
                , 'Sharpe_Ratio': float(0)
                , 'Cash': float(cash)})

        # self.portfolio_years = self.get_portfolio_years() -- необходимо написать геттеры и сеттеры
        # self.total_return = self.get_total_return()
        # self.cagr = self.get_cagr()

        self.no_trade_days = 8
        self.trade_result = None
        self.in_fund = in_fund
        self.model = model


    def start_trade(self):
        while self.current_date < self.end_date:
            date = self.current_date + timedelta(days=1)
            if date in self.data.index:  # Когда расчётный день - торговый
                self.calculate_daily_result(date)
                self.trade()

            self.current_date = date


    def calculate_daily_result(self, date):
        data_row = self.data.loc[date]
        self.current_port_date = date
        if self.no_trade_days > 0 and self.in_fund:  # Когда не было операций с портфелем на прошлый день
            new_port_row = self.portfolio.iloc[-1].copy()
            new_port_row['Close_Port'] = new_port_row['Close_Port'] * (1 + data_row['Returns']) + \
                                         self.data.shift(1)['Dividends'].loc[date]
            new_port_row['Returns'] = (new_port_row['Close_Port'] / self.portfolio.iloc[-1]['Close_Port']) - 1
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

        elif self.no_trade_days > 0 and not self.in_fund:  # Когда не было операций с портфелем на прошлый день
            new_port_row = self.portfolio.iloc[-1].copy()
            new_port_row.name = date

            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

        else:  # Когда были опирации с портфелем на прошлый день
            new_port_row = self.trade_result
            if not self.in_fund:
                new_port_row['Cash'] = new_port_row['Cash'] + self.data.shift(1)['Dividends'].loc[date]
                new_port_row['Returns'] = (new_port_row['Cash'] / self.portfolio.iloc[-1]['Close_Port']) - 1
                new_port_row.name = date

                self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

            elif self.in_fund:
                new_port_row['Close_Port'] = new_port_row['Close_Port'] + self.data.shift(1)['Dividends'].loc[date]
                new_port_row['Returns'] = (new_port_row['Close_Port'] / self.portfolio.iloc[-1]['Cash']) - 1
                new_port_row.name = date

                self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])


    def trade(self, percent_change=0.05):
        if self.no_trade_days > 7:
            dt = self.current_port_date
            current_close = self.data.loc[dt]['Close']
            # next_week_max_close = self.data.loc[dt:dt + timedelta(days=8)]['Close'].max()
            # next_week_min_close = self.data.loc[dt:dt + timedelta(days=8)]['Close'].min()
            next_week_mean_close = self.data.loc[dt:dt + timedelta(days=8)]['Close'].mean()

            if (next_week_mean_close / current_close) - 1 > percent_change and not self.in_fund:
                self.trade_buy()
                self.no_trade_days = 0

            elif (next_week_mean_close / current_close) - 1 < -percent_change and self.in_fund:
                self.trade_sell()
                self.no_trade_days = 0

            else:
                self.no_trade_days += 1
        else:
            self.no_trade_days += 1


    def trade_buy(self):
        trade_row = self.portfolio.iloc[-1].copy()
        trade_row['Close_Port'], trade_row['Cash'] = trade_row['Cash'], trade_row['Close_Port']
        self.trade_result, self.in_fund = trade_row, True


    def trade_sell(self):
        trade_row = self.portfolio.iloc[-1].copy()
        trade_row['Close_Port'], trade_row['Cash'] = trade_row['Cash'], trade_row['Close_Port']
        self.trade_result, self.in_fund = trade_row, False



    def calculate_sharpe_ratio(self):
        average_returns = self.get_average_returns()
        volatility = self.get_volatility()
        risk_free_rate = self.data.loc[self.portfolio.iloc[-1].name, 'DGS10']

        return (average_returns - risk_free_rate) / volatility


    def get_volatility(self):
        return np.std(self.portfolio['Returns']) * np.sqrt(252)


    def get_total_return(self):
        initial_value = self.portfolio['Close_Port'].iloc[0]
        final_value = self.portfolio['Close_Port'].iloc[-1]

        return (final_value / initial_value) - 1


    def get_cagr(self):
        portfolio_years = self.get_portfolio_years()
        return (1 + self.get_total_return()) ** (1 / portfolio_years) - 1


    def get_average_returns(self):
        portfolio_years = self.get_portfolio_years()
        return self.get_total_return() / portfolio_years


    def get_portfolio_years(self):
        return len(self.portfolio) / 252


    def plt(self, start_year=None, end_year=None):
        if start_year is None:
            start_year = self.portfolio.iloc[0].name
        else:
            start_year = datetime(start_year, 1, 1)
        if end_year is None:
            end_yaer = self.portfolio.iloc[-1].name
        else:
            end_year = datetime(end_year, 1, 1)

        plt.figure(figsize=(12, 6), dpi=200)
        sns.lineplot(data=self.portfolio, x=self.portfolio.index, y='Close_Port')
        plt.xlim((start_year, (end_year)))
        plt.xlabel('Date')
        plt.ylabel('Port Cost')


TS = TradeSystem(1000, data, datetime(1986, 1, 2), datetime(2000, 11, 21))  # Не принимает значение меньшее или равное первой дате в данных
TS.start_trade()
print(TS.portfolio)

print(TS.get_cagr())
print(TS.calculate_sharpe_ratio())
print(TS.portfolio['Cash'].mean())