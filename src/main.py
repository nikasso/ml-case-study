import pandas as pd
import datetime


def load_data(filename):
    df = pd.read_csv(filename,parse_dates=['last_trip_date','signup_date'])
    date_cutoff = datetime.date(2014, 6, 1)
    df['churn'] = (df.last_trip_date < date_cutoff).astype(int)
    return df


def data_processing(df):
    '''convert null'''
    df['avg_rating_of_driver_isnull'] = df.avg_rating_of_driver.isnull().astype(int)
    df['avg_rating_by_driver_isnull'] = df.avg_rating_by_driver.isnull().astype(int)

    df.avg_rating_of_driver = df.avg_rating_of_driver.fillna(value=0)
    df.avg_rating_by_driver = df.avg_rating_by_driver.fillna(value=0)

    '''dummify vaiables'''
    dic1 = {True: 1, False: 0}
    df["luxury_car_user"] = df["luxury_car_user"].map(dic1)
    df['phone'].fillna('no_phone', inplace=True)
    city_dummy = pd.get_dummies(df['city'],drop_first=True)
    phone_dummy = pd.get_dummies(df['phone'],drop_first=True)
    df_dummy = pd.concat([df, city_dummy, phone_dummy], axis=1)
    df_dummy.drop(['city','phone'], axis=1,inplace=True)
    return df_dummy

def drop_date(df):
    df = df.drop(['last_trip_date','signup_date'],axis=1)
    return df

if __name__ == '__main__':
    train_file = 'data/churn_train.csv'
    test_file = 'data/churn_test.csv'

    train_df = load_data(train_file)
    test_df = load_data(test_file)

    train_df = data_processing(train_df)
    test_df = data_processing(test_df)
