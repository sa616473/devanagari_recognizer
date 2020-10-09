import matplotlib.pyplot as plt

def visualization(char_data, char_labels, isBinary=False):
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        img = char_data[i]
        fig.add_subplot(rows, columns, i)
        plt.title(char_labels[i].split('_')[-1])
        if isBinary:
            plt.imshow(img, cmap=plt.cm.binary)
            plt.savefig('../reports/figures/char_plots_binary.png')
        else:
            plt.imshow(img)
            plt.savefig('../reports/figures/char_plots.png')
            

def training_visualize(history, title=''):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy {}'.format(title))

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss {}'.format(title))
    plt.savefig('../reports/performance/{}'.format(title))
    plt.show()