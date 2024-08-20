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
    def __init__(self, investment, data, start_date, end_date, window=7, model=None, in_fund=True, returns=0):
        self.investment = investment
        self.data = data
        self.start_date, self.end_date = start_date, end_date
        self.current_date = start_date

        # Для простоты расчёта инициируем стоимость портфеля с закрытия
        self.portfolio = pd.DataFrame(index=[start_date]
                                      ,data={'Close_Port': float(investment)
                                            ,'Returns': float(returns)
                                            ,'Sharpe_Ratio': float(0)
                                            ,'Trade_ind': 0})

        self.no_trade_days, self.window = 8, window
        self.in_fund = in_fund
        self.model = model


    def start_trade(self):
        while self.current_date < self.end_date:
            self.current_date += timedelta(days=1)
            if self.current_date in self.data.index:        # Когда расчётный день - торговый
                self.calculate_daily_result(self.current_date)
                self.trade()



    def calculate_daily_result(self, date):
        data_row = self.data.loc[date]
        new_port_row = self.portfolio.iloc[-1].copy()
        dividends = self.data.shift(1)['Dividends'].loc[date]

        if self.no_trade_days > 0 and self.in_fund:     # Не было операций и деньги в фонде
            new_port_row['Close_Port'] = new_port_row['Close_Port'] * (1 + data_row['Returns']) + dividends
            new_port_row['Returns'] = (new_port_row['Close_Port'] / self.portfolio.iloc[-1]['Close_Port']) - 1
            new_port_row['Trade_ind'] = 0
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

        elif self.no_trade_days > 0 and not self.in_fund:   # Не было операций и деньги в кэше
            new_port_row['Trade_ind'] = 0
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

        elif not self.in_fund:      # Когда была продажа
            new_port_row['Close_Port'] = new_port_row['Close_Port'] + dividends
            new_port_row['Returns'] = (new_port_row['Close_Port'] / self.portfolio.iloc[-1]['Close_Port']) - 1
            new_port_row['Trade_ind'] = 1
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

        elif self.in_fund:      # Когда была покупка
            new_port_row['Trade_ind'] = 1
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])



    def trade(self, percent_change=0.05):
        if self.no_trade_days > 7:
            dt = self.current_date
            current_close = self.data.loc[dt]['Close']
            # next_week_max_close = self.data.loc[dt:dt + timedelta(days=8)]['Close'].max()
            # next_week_min_close = self.data.loc[dt:dt + timedelta(days=8)]['Close'].min()
            next_week_mean_close = self.data.loc[dt:dt + timedelta(days=self.window + 1)]['Close'].mean()

            if (next_week_mean_close / current_close) - 1 > percent_change and not self.in_fund:
                self.in_fund = True
                self.no_trade_days = 0

            elif (next_week_mean_close / current_close) - 1 < -percent_change and self.in_fund:
                self.in_fund = False
                self.no_trade_days = 0

            else:
                self.no_trade_days += 1
        else:
            self.no_trade_days += 1


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


TS = TradeSystem(1000, data, datetime(1986, 1, 2), datetime(2000, 11, 21), window=5)  # Не принимает значение меньшее или равное первой дате в данных
TS.start_trade()
print(TS.portfolio)

print('Window:', TS.window)
print('CAGR:', TS.get_cagr())
print('SHARP:', TS.calculate_sharpe_ratio())
print('Total returns:', TS.get_total_return())
print('Count trades:', TS.portfolio['Trade_ind'].sum())
print('Current state in fund:', TS.in_fund)
