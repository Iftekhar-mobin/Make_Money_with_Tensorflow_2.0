import matplotlib.pyplot as plt

    # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
	
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

def plot_predictions(nan_plus_predicted_30, last_100):
    plt.figure(figsize=(15,8))
    plt.plot(nan_plus_predicted_30, label='Predicted Actual', marker='o')
    plt.plot(last_100, label='History Actual')
    plt.legend()