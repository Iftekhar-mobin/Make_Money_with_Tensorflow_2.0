from sklearn.preprocessing import MinMaxScaler
from numpy import array , hstack

    # split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def process_dataset(df_h4, steps_in, steps_out, split=True):
    # Check for missing values
    print(df_h4.isna().sum())

    # Drop rows with missing values
    df_h4 = df_h4.dropna()

    df_h4.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'} , inplace = True)
    df_h4['avg'] = df_h4.loc[:, ['Open', 'High', 'Low', 'Close']].mean(axis=1)

    dataset = df_h4

    x_1 = dataset['Close'].values
    x_2 = dataset['Open'].values
    x_3 = dataset['High'].values
    x_4 = dataset['Low'].values
    y_avg = dataset['avg'].values

    # convert to [rows, columns] structure
    x_1 = x_1.reshape((len(x_1), 1))
    x_2 = x_2.reshape((len(x_2), 1))
    x_3 = x_2.reshape((len(x_3), 1))
    x_4 = x_2.reshape((len(x_4), 1))
    y = y.reshape((len(y_avg), 1))

    # print ("x_1.shape" , x_1.shape)
    # print ("y.shape" , y.shape)

    # normalization features
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_1_scaled = scaler.fit_transform(x_1)
    x_2_scaled = scaler.fit_transform(x_2)
    x_3_scaled = scaler.fit_transform(x_3)
    x_4_scaled = scaler.fit_transform(x_4)
    y_scaled = scaler.fit_transform(y)

    # horizontally stack columns
    dataset_stacked = hstack((x_1_scaled, x_2_scaled, x_3_scaled, x_4_scaled, y_scaled))
    print ("dataset_stacked.shape" , dataset_stacked.shape)


    # covert into input/output
    X_, y_ = split_sequences(dataset_stacked, steps_in, steps_out)

    print ("After sequence Splitting X.shape" , X_.shape)
    print ("After sequence Splitting y.shape" , y_.shape)

    if split:
        split_range = int(len(X_)*0.95)
        train_X , train_y = X_[:split_range, :] , y_[:split_range, :]
        test_X , test_y = X_[split_range:, :] , y_[split_range:, :]
    else:
        train_X , train_y = X_, y_
        test_X , test_y = None, None

    n_features = train_X.shape[2]


    print ("train_X.shape" , train_X.shape)
    print ("train_y.shape" , train_y.shape)
    print ("n_features" , n_features)

    return n_features, train_X , train_y, test_X , test_y, y_avg, scaler
