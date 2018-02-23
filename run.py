from model.densenet import DenseNet
from keras.optimizers import SGD
import keras
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard, LearningRateScheduler
from model.model_saver import MultiGPUCheckpointCallback
from keras.losses import categorical_crossentropy
import argparse
from image_utils import read_data
import os

batch_size = 64
n_gpus = 2
n_epochs = 40
image_shape = (64, 64, 3)
n_classes = 80
initial_learning_rate = 0.1
reduce_lr_epoch_1 = 20
reduce_lr_epoch_2 = 30
image_dir = "C:\\Users\\li\\Desktop\\IMAGE\\data"


def test_model():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--test", action="store_true", help="测试模型")
    parser.add_argument('--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'], default='DenseNet',
                        help='What type of model to use')
    parser.add_argument('--growth_rate', '-k', type=int, choices=[12, 24, 40], default=12,
                        help='Grows rate for every layer,choices were restricted to used in paper')
    parser.add_argument('--depth', '-d', type=int, choices=[40, 100, 190, 250], default=40,
                        help='Depth of whole network, restricted to paper choices')
    parser.add_argument('--total_blocks', '-tb', type=int, default=3, metavar='',
                        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument('--keep_prob', '-kp', type=float, default=1.0, metavar='',
                        help="Keep probability for dropout.")
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, metavar='',
                        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument('--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
                        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument('--reduction', '-red', type=float, default=0.5, metavar='',
                        help='reduction Theta at transition layer for DenseNets-BC models')
    parser.add_argument('--logs', dest='should_save_logs', action='store_true',
                        help='Write tensorflow logs ')
    parser.add_argument('--no-logs', dest='should_save_logs', action='store_false',
                        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)
    parser.add_argument('--saves', dest='should_save_model', action='store_true',
                        help='Save model during training')
    parser.add_argument('--no-saves', dest='should_save_model', action='store_false',
                        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    args = parser.parse_args()
    if args.model_type == 'DenseNet':
        args.bc_mode = False
        args.reduction = 0.0
    elif args.model_type == 'DenseNet-BC':
        args.bc_mode = True

    if not args.train and not args.test:
        print("需要指定 --train 或 --test")
        exit()

    if keras.backend.backend() != "tensorflow":
        print("只可运行于基于TensorFlow后端的Keras下")

    model_identifier = "%s_k=%s_d=%s" % (args.model_type, args.growth_rate, args.depth)

    images, labels = read_data(image_dir, image_shape)
    labels = keras.utils.to_categorical(labels, n_classes)

    base_model = DenseNet(classes=n_classes, input_shape=image_shape, depth=args.depth,
                          growth_rate=args.growth_rate,
                          bottleneck=args.bc_mode, reduction=args.reduction, dropout_rate=1.0 - args.keep_prob,
                          weight_decay=args.weight_decay)

    if args.train:
        batch_size *= n_gpus

        if os.path.exists("saves/%s.weight" % model_identifier):
            print("Loading model...")
            base_model.load_weights("saves/%s.weight" % model_identifier, by_name=True)

        if n_gpus > 1:
            model = multi_gpu_model(base_model, n_gpus)
        else:
            model = base_model


        def loss_func(y_true, y_pred):
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_weights])
            return categorical_crossentropy(y_true, y_pred) + l2_loss * 1e-4


        optimizer = SGD(lr=initial_learning_rate, clipvalue=0.5, momentum=0.9, decay=1e-4, nesterov=True)
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])


        # Callback:LearningRateScheduler
        def lr_reduce(epoch):
            if epoch < reduce_lr_epoch_1:
                return initial_learning_rate
            elif epoch >= reduce_lr_epoch_2:
                return initial_learning_rate / 100
            else:
                return initial_learning_rate / 10


        # define callbacks
        learning_rate_scheduler = LearningRateScheduler(lr_reduce)
        checkpoints = MultiGPUCheckpointCallback("saves/%s.weight" % model_identifier, base_model=base_model,
                                                 save_weights_only=True, save_best_only=True)

        tensorboard = TensorBoard("logs/%s/" % (model_identifier), batch_size=batch_size, histogram_freq=10)

        callbacks = [learning_rate_scheduler]
        if args.should_save_model:
            callbacks.append(checkpoints)
        if args.should_save_logs:
            callbacks.append(tensorboard)

        model.fit(x=images, y=labels, validation_split=0.1, batch_size=batch_size, epochs=n_epochs,
                  callbacks=callbacks, shuffle=True)

    elif args.test:

        if os.path.exists("saves/%s.weight" % model_identifier):
            print("Loading model...")
            base_model.load_weights("saves/%s.weight" % model_identifier, by_name=True)
        else:
            print("saves/%s.weight file not exists" % model_identifier)
            exit()


        def loss_func(y_true, y_pred):
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in base_model.trainable_weights])
            return categorical_crossentropy(y_true, y_pred) + l2_loss * 1e-4


        optimizer = SGD(lr=initial_learning_rate, clipvalue=0.5, momentum=0.9, decay=1e-4, nesterov=True)
        base_model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

        score = base_model.evaluate(images, labels, batch_size=batch_size)
        print("Test loss: %.3f" % score[0])
        print("Test accuracy: %.3f" % score[1])
