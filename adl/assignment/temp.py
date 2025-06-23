def plot_training_history(train_metrics, val_metrics, model_name):
    """Plot training and validation metrics history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot loss
    axes[0].plot(train_metrics["loss"], label="Train")
    axes[0].plot(val_metrics["loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Plot accuracy
    axes[1].plot(train_metrics["acc"], label="Train")
    axes[1].plot(val_metrics["acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Plot F1 score
    axes[2].plot(train_metrics["f1"], label="Train")
    axes[2].plot(val_metrics["f1"], label="Val")
    axes[2].set_title("F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.suptitle(f"Training History - {model_name}")
    plt.tight_layout()
    plt.show()


"""
Epoch 1/20: 100%
 1563/1563 [02:29<00:00, 10.29it/s, loss=0.6964]

Epoch 1:
Train Loss: 0.6919, Acc: 0.5240, F1: 0.5224
Val Loss: 0.6932, Acc: 0.5004, F1: 0.3362
Epoch 2/20: 100%
 1563/1563 [02:29<00:00, 16.83it/s, loss=0.6403]

Epoch 2:
Train Loss: 0.6049, Acc: 0.6591, F1: 0.6590
Val Loss: 0.4433, Acc: 0.7924, F1: 0.7922
Epoch 3/20: 100%
 1563/1563 [02:30<00:00, 16.71it/s, loss=0.1858]

Epoch 3:
Train Loss: 0.3335, Acc: 0.8630, F1: 0.8630
Val Loss: 0.3685, Acc: 0.8362, F1: 0.8358
Epoch 4/20: 100%
 1563/1563 [02:29<00:00, 16.85it/s, loss=0.5301]

Epoch 4:
Train Loss: 0.1946, Acc: 0.9272, F1: 0.9272
Val Loss: 0.4038, Acc: 0.8347, F1: 0.8343
Epoch 5/20: 100%
 1563/1563 [02:30<00:00, 14.28it/s, loss=0.3241]

Epoch 5:
Train Loss: 0.1025, Acc: 0.9673, F1: 0.9673
Val Loss: 0.5015, Acc: 0.8455, F1: 0.8454
Epoch 6/20: 100%
 1563/1563 [02:29<00:00, 16.71it/s, loss=0.0031]

Epoch 6:
Train Loss: 0.0524, Acc: 0.9845, F1: 0.9845
Val Loss: 0.6075, Acc: 0.8310, F1: 0.8306
Epoch 7/20: 100%
 1563/1563 [02:30<00:00, 16.82it/s, loss=0.0004]

Epoch 7:
Train Loss: 0.0264, Acc: 0.9925, F1: 0.9925
Val Loss: 0.8652, Acc: 0.8351, F1: 0.8348
Epoch 8/20: 100%
 1563/1563 [02:30<00:00, 16.81it/s, loss=0.0003]

Epoch 8:
Train Loss: 0.0225, Acc: 0.9934, F1: 0.9934
Val Loss: 1.0397, Acc: 0.8362, F1: 0.8360
Epoch 9/20: 100%
 1563/1563 [02:30<00:00, 16.67it/s, loss=0.0040]

Epoch 9:
Train Loss: 0.0147, Acc: 0.9954, F1: 0.9954
Val Loss: 1.1362, Acc: 0.8374, F1: 0.8374
Epoch 10/20: 100%
 1563/1563 [02:29<00:00, 16.97it/s, loss=0.0025]

Epoch 10:
Train Loss: 0.0117, Acc: 0.9971, F1: 0.9971
Val Loss: 1.2712, Acc: 0.8358, F1: 0.8358
Epoch 11/20: 100%
 1563/1563 [02:30<00:00, 16.98it/s, loss=0.0007]

Epoch 11:
Train Loss: 0.0112, Acc: 0.9969, F1: 0.9969
Val Loss: 1.1998, Acc: 0.8373, F1: 0.8373
Epoch 12/20: 100%
 1563/1563 [02:30<00:00, 16.95it/s, loss=0.0001]

Epoch 12:
Train Loss: 0.0099, Acc: 0.9975, F1: 0.9975
Val Loss: 1.3750, Acc: 0.8353, F1: 0.8353
Epoch 13/20: 100%
 1563/1563 [02:29<00:00, 16.91it/s, loss=0.0000]

Epoch 13:
Train Loss: 0.0044, Acc: 0.9988, F1: 0.9988
Val Loss: 1.4307, Acc: 0.8320, F1: 0.8315
Epoch 14/20: 100%
 1563/1563 [02:29<00:00, 16.93it/s, loss=0.0000]

Epoch 14:
Train Loss: 0.0086, Acc: 0.9977, F1: 0.9977
Val Loss: 1.7426, Acc: 0.8344, F1: 0.8343
Epoch 15/20: 100%
 1563/1563 [02:29<00:00, 16.95it/s, loss=0.0000]

Epoch 15:
Train Loss: 0.0052, Acc: 0.9988, F1: 0.9988
Val Loss: 1.4422, Acc: 0.8299, F1: 0.8297
Epoch 16/20: 100%
 1563/1563 [02:29<00:00, 16.95it/s, loss=0.0003]

Epoch 16:
Train Loss: 0.0064, Acc: 0.9982, F1: 0.9982
Val Loss: 1.5500, Acc: 0.8326, F1: 0.8326
Epoch 17/20: 100%
 1563/1563 [02:28<00:00, 16.92it/s, loss=0.0006]

Epoch 17:
Train Loss: 0.0069, Acc: 0.9982, F1: 0.9982
Val Loss: 1.5939, Acc: 0.8335, F1: 0.8332
Epoch 18/20: 100%
 1563/1563 [02:30<00:00, 16.86it/s, loss=0.0000]

Epoch 18:
Train Loss: 0.0036, Acc: 0.9991, F1: 0.9991
Val Loss: 2.1447, Acc: 0.8347, F1: 0.8346
Epoch 19/20: 100%
 1563/1563 [02:27<00:00, 16.89it/s, loss=0.0005]

Epoch 19:
Train Loss: 0.0005, Acc: 0.9998, F1: 0.9998
Val Loss: 3.3284, Acc: 0.8175, F1: 0.8157
Epoch 20/20: 100%
 1563/1563 [02:29<00:00, 16.87it/s, loss=0.0011]

Epoch 20:
Train Loss: 0.0066, Acc: 0.9986, F1: 0.9986
Val Loss: 1.4806, Acc: 0.8290, F1: 0.8288
"""
