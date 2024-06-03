from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,LSTM

def lstm_model(n_steps_in, n_steps_out, n_features):
    # define model
    model_arc = Sequential()
    model_arc.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model_arc.add(LSTM(50, activation='relu'))
    model_arc.add(Dense(n_steps_out))
    # model.add(Activation('linear'))
    model_arc.compile(loss='mse' , optimizer='adam' , metrics=['accuracy'])
    return model_arc


def train_or_reload(lstm_model, saved_model, train_X, train_y, test_X=None, test_y=None):
    history = ''
    try:
        lstm_model = load_model(saved_model)
        print('Model loaded successfully')
    except OSError as e:
        print("Model is not loaded, Training model now:", e)
        # Fit network
        history = lstm_model.fit(train_X, train_y , epochs=25,  
                            verbose=1,validation_data=(test_X, test_y), 
                            validation_split = 0.1, shuffle=False)
        lstm_model.save(saved_model)
    finally:    
	    return history, lstm_model

def predict_prices(model, test_X):
    predicted_price = model.predict(test_X)
    print('predicted_price: \n', predicted_price)
    return predict_prices