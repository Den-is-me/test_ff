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
                               , pd.read_csv('./Прошлые данные доходности облигаций США 10-летние 05-21.csv')])

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
    data['DGS10'] = data['DGS10'] / 100
    data['Returns'] = ((data['Close'] + data['Dividends'].shift(1)) / data['Close'].shift(1)) - 1

    data.index = data.Date
    data = data.drop(['Open', 'High', 'Date', 'Low', 'Volume', 'Stock Splits'], axis=1)

    return data


class TradeSystem():
    def __init__(self
                 , investment  # Размер первоначального капитала
                 , data  # DF Фонда обязательные поля: (Close, Returns, VIX_Close, Dividends)
                 , start_date: datetime  # Дата начала торгов
                 , end_date: datetime  # Дата окончания торгов
                 , SMA_target='VIX_Close'
                 , SMA_short_window=20  # Размер окна для короткой скользящей
                 , SMA_long_window=100  # Размер окна для длинной скользящей
                 , coef_window=0  # Количество дней за которые расчитывать коэффиценты Sharpe, CAGR
                 , trade_needed=True  # Совершать операции или выбрать стратегию Buy and Hold
                 , model=None  # Модель для определения решения о покупке, продаже (можно улучшить)
                 , in_fund=True  # Индикатор того что капитал в Фонде
                 ):

        # use ceof_window=0 if you want to calculate coefficient by the whole DF
        self.investment = investment
        self.data = data
        self.start_date, self.end_date = start_date, end_date
        self.current_date = start_date

        # Для простоты расчёта инициируем стоимость портфеля с закрытия
        self.portfolio = pd.DataFrame(index=[start_date]
                                    ,data={'Close_Port': float(investment)
                                    ,'Returns': float(0)
                                    ,'Sharpe_Ratio': float(0)
                                    ,'Trade_ind': 0})

        # exit_coef - коэфициент выхода из фонда, увеличивается по мере приближения к end_date
        self.no_trade_days, self.exit_coef, self.trade_needed = 8, 1, trade_needed
        self.SMA_target = SMA_target
        self.data[f'Short_SMA_{SMA_target}'] = self.data[SMA_target].rolling(SMA_short_window).mean()
        self.data[f'Long_SMA_{SMA_target}'] = self.data[SMA_target].rolling(SMA_long_window).mean()
        self.data['Long_SMA_Close'] = self.data['Close'].rolling(SMA_long_window).mean()
        self.short_up = not in_fund
        self.in_fund = in_fund
        self.model = model
        self.coef_window = coef_window


    def start_trade(self):
        while self.current_date < self.end_date:
            self.current_date += timedelta(days=1)
            if self.current_date in self.data.index:  # Когда расчётный день - торговый
                self.calculate_daily_result(self.current_date)
                if self.trade_needed:
                    self.trade()


    def calculate_daily_result(self, date):
        data_row = self.data.loc[date]
        prev_data_row = self.data.shift(1)
        new_port_row = self.portfolio.iloc[-1].copy()
        dividends = new_port_row['Close_Port'] / (prev_data_row.loc[date]['Close'] if pd.notnull(prev_data_row['Close'].loc[date]) else 1) * \
                    (prev_data_row['Dividends'].loc[date] if pd.notnull(prev_data_row['Dividends'].loc[date]) else 0)

        if self.no_trade_days > 0 and self.in_fund:  # Не было операций и деньги в фонде
            new_port_row['Close_Port'] = new_port_row['Close_Port'] * (1 + data_row['Returns']) + dividends
            new_port_row['Returns'] = (new_port_row['Close_Port'] / self.portfolio.iloc[-1]['Close_Port']) - 1
            new_port_row['Trade_ind'] = 0
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

        elif self.no_trade_days > 0 and not self.in_fund:  # Не было операций и деньги в кэше
            new_port_row['Trade_ind'] = 0
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

        elif self.in_fund:  # Когда была покупка
            new_port_row['Trade_ind'] = 1
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])

        elif not self.in_fund:  # Когда была продажа
            new_port_row['Close_Port'] = new_port_row['Close_Port'] + dividends
            new_port_row['Returns'] = (new_port_row['Close_Port'] / self.portfolio.iloc[-1]['Close_Port']) - 1
            new_port_row['Trade_ind'] = 2
            new_port_row.name = date
            self.portfolio = pd.concat([self.portfolio, new_port_row.to_frame().T])


    def trade(self):
        # Для упрощения расчётов используем 100% распределение между фондом и кэшем.
        if self.no_trade_days > 7:
            dt = self.current_date
            if (self.end_date - dt).days < 30:  # Увеличиваем коэффициент выхода из фонда
                self.exit_coef += 0.05

            shrt_sma = self.data.at[dt, f'Short_SMA_{self.SMA_target}'] * self.exit_coef
            lng_sma = self.data.at[dt, f'Long_SMA_{self.SMA_target}']

            if not self.short_up and shrt_sma > lng_sma and self.in_fund:  # Sell
                self.in_fund = False
                self.no_trade_days = 0

            elif self.short_up and shrt_sma < lng_sma and not self.in_fund:  # Bye
                self.in_fund = True
                self.no_trade_days = 0

            else:
                self.no_trade_days += 1

            self.short_up = shrt_sma > lng_sma

        else:
            self.no_trade_days += 1


    def calculate_sharpe_ratio(self, portfolio=None):
        if portfolio is None:
            portfolio = self.portfolio
        wind = self.coef_window

        average_returns = portfolio.iloc[-wind:]['Returns'].mean() * 252
        std_dev = portfolio.iloc[-wind:]['Returns'].std() * np.sqrt(252)
        risk_free_rate = self.data.iloc[-wind:]['DGS10'].mean()  # Не правильно считает для окна с енд меньше чем в data

        return (average_returns - risk_free_rate) / std_dev


    def calculate_cagr(self, portfolio=None):
        if portfolio is None:
            portfolio = self.portfolio
        elif 'Close_Port' not in portfolio.columns:
            portfolio['Close_Port'] = portfolio['Close']
        wind = self.coef_window

        portfolio_years = len(portfolio[-wind:]) / 252

        return (1 + self.get_total_return(portfolio)) ** (1 / portfolio_years) - 1


    def get_total_return(self, portfolio=None):
        if portfolio is None:
            portfolio = self.portfolio
        elif 'Close_Port' not in portfolio.columns:
            portfolio['Close_Port'] = portfolio['Close']
        wind = self.coef_window

        initial_value = portfolio['Close_Port'].iloc[-wind]
        final_value = portfolio['Close_Port'].iloc[-1]

        return (final_value / initial_value) - 1


    def plt(self, start_year=None, end_year=None):
        if start_year is None:
            start_year = self.portfolio.iloc[0].name
        else:
            start_year = datetime(start_year, 1, 1)
        if end_year is None:
            end_year = self.portfolio.iloc[-1].name
        else:
            end_year = datetime(end_year, 1, 1)

        data = self.data.loc[start_year:end_year]
        print('CAGR:', self.calculate_cagr())
        print('SHARP:', self.calculate_sharpe_ratio())
        print('Total returns:', self.get_total_return())
        print('Count trades:', self.portfolio.loc[self.portfolio['Trade_ind'] != 0]['Trade_ind'].count())
        print('Current state in cash:', not self.in_fund)


        plt.figure(figsize=(12, 6), dpi=200)
        signal_bye_df = self.portfolio.loc[self.portfolio['Trade_ind'] == 1]
        signal_sell_df = self.portfolio.loc[self.portfolio['Trade_ind'] == 2]

        sns.lineplot(data=self.portfolio, x=self.portfolio.index, y='Close_Port', alpha=0.8, label='Portfolio Close')
        sns.lineplot(data=data, x=data.index, y=data['Close'], alpha=0.4, label='Fund Close')
        sns.scatterplot(data=signal_bye_df, x=signal_bye_df.index, y='Close_Port', color='green', label='Buy', s=60, marker='^')
        sns.scatterplot(data=signal_sell_df, x=signal_sell_df.index, y='Close_Port', color='red', label='Sell', s=60, marker='v')
        plt.xlabel('Date')
        plt.ylabel('Close Price')

        plt.xlim((start_year, (end_year)))
        # plt.show()


data = load_data()

# TS = TradeSystem(100, data, datetime(1986, 1, 2), datetime(2021, 1, 22))
# TS.start_trade()
# print(TS.portfolio)
# TS.plt()

# print(TS.calculate_sharpe_ratio())
#
#
train_df = data.copy().loc[datetime(1996, 1, 1):datetime(2016, 1, 1)]
test_df_flat = data.copy().loc[:datetime(1996, 1, 1)]
test_df_bull = data.copy().loc[datetime(2016, 1, 1):]


start_date = datetime(1996, 1, 1)
end_date = datetime(2022, 1, 22)

TS = TradeSystem(10, train_df, start_date, end_date)
TS.start_trade()

print(TS.portfolio)