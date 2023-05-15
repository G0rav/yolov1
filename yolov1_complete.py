
################# model_tiny.py 

from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    Flatten, Dense, Reshape, LeakyReLU, BatchNormalization
# from tensorflow.keras.layers.normalization import 
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class Yolo_Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super(Yolo_Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def call(self, inputs, **kwargs):
        S = [self.target_shape[0], self.target_shape[1]]
        C = 20
        B = 2
        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B
        # class prediction
        class_probs = K.reshape(
            inputs[:, :idx1], (K.shape(inputs)[0],) + tuple([S[0], S[1], C]))
        class_probs = K.softmax(class_probs)
        # confidence
        confs = K.reshape(
            inputs[:, idx1:idx2], (K.shape(inputs)[0],) + tuple([S[0], S[1], B]))
        confs = K.sigmoid(confs)
        # boxes
        boxes = K.reshape(
            inputs[:, idx2:], (K.shape(inputs)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)
        # return np.array([class_probs, confs, boxes])
        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs


def model_tiny_yolov1(inputs):
    x = Conv2D(16, (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(inputs)
    x = BatchNormalization(name='bnconvolutional_0', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same', name='convolutional_1', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_1', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same', name='convolutional_2', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_2', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same', name='convolutional_3', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_3', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_4', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_4', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(512, (3, 3), padding='same', name='convolutional_5', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_5', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(1024, (3, 3), padding='same', name='convolutional_6', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_6', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_7', use_bias=False,
               kernel_regularizer=l2(5e-4), trainable=False)(x)
    x = BatchNormalization(name='bnconvolutional_7', trainable=False)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    x = Dense(1470, activation='linear', name='connected_0')(x)
    # outputs = Reshape((7, 7, 30))(x)
    outputs = Yolo_Reshape((7, 7, 30))(x)

    return outputs




################# yolo.py 

import tensorflow.keras.backend as K


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh


def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :20]  # ? * 7 * 7 * 20
    label_box = y_true[..., 20:24]  # ? * 7 * 7 * 4
    response_mask = y_true[..., 24]  # ? * 7 * 7
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    predict_trust = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss


################ data_sequence.py

from tensorflow.keras.utils import Sequence
import math
import cv2 as cv
import numpy as np
import os


class SequenceData(Sequence):

    def __init__(self, model, dir, target_size, batch_size, shuffle=True):
        self.model = model
        self.datasets = []
        if self.model is 'train':
            with open(os.path.join(dir, '2007_train.txt'), 'r') as f:
                self.datasets = self.datasets + f.readlines()
        elif self.model is 'val':
            with open(os.path.join(dir, '2007_val.txt'), 'r') as f:
                self.datasets = self.datasets + f.readlines()
        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_imgs = len(self.datasets)
        return math.ceil(num_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch = [self.datasets[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch)
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read(self, dataset):
        dataset = dataset.strip().split()
        image_path = dataset[0]
        label = dataset[1:]

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，所以要转换
        image_h, image_w = image.shape[0:2]
        image = cv.resize(image, self.image_size)
        image = image / 255.

        label_matrix = np.zeros([7, 7, 25])
        for l in label:
            l = l.split(',')
            l = np.array(l, dtype=np.int)
            xmin = l[0]
            ymin = l[1]
            xmax = l[2]
            ymax = l[3]
            cls = l[4]
            x = (xmin + xmax) / 2 / image_w
            y = (ymin + ymax) / 2 / image_h
            w = (xmax - xmin) / image_w
            h = (ymax - ymin) / image_h
            loc = [7 * x, 7 * y]
            loc_i = int(loc[1])
            loc_j = int(loc[0])
            y = loc[1] - loc_i
            x = loc[0] - loc_j

            if label_matrix[loc_i, loc_j, 24] == 0:
                label_matrix[loc_i, loc_j, cls] = 1
                label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                label_matrix[loc_i, loc_j, 24] = 1  # response

        return image, label_matrix

    def data_generation(self, batch_datasets):
        images = []
        labels = []

        for dataset in batch_datasets:
            image, label = self.read(dataset)
            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        return X, y




#################### train.py

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
from models.model_tiny_yolov1 import model_tiny_yolov1
from data_sequence import SequenceData
from yolo.yolo import yolo_loss
from callback import callback
import tensorflow as tf


epochs = 10
batch_size = 32
datasets_path = ''


def _main():
    epochs = epochs
    batch_size = batch_size

    input_shape = (448, 448, 3)
    inputs = Input(input_shape)
    yolo_outputs = model_tiny_yolov1(inputs)

    model = Model(inputs=inputs, outputs=yolo_outputs)

    print(model.summary())
    
    tf.keras.utils.plot_model(model,
                              to_file='yolov1.png',
                              show_shapes=True,
                              show_layer_names=True)
    
    model.compile(loss=yolo_loss, optimizer='adam')

    save_dir = 'checkpoints'
    weights_path = os.path.join(save_dir, 'weights.hdf5')
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if os.path.exists('checkpoints/weights.hdf5'):
        model.load_weights('checkpoints/weights.hdf5', by_name=True)
    else:
        model.load_weights('tiny-yolov1.hdf5', by_name=True)
        print('no train history')


    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')


    datasets_path = os.path.expanduser(datasets_path)

    train_generator = SequenceData(
        'train', datasets_path, input_shape, batch_size)
    validation_generator = SequenceData(
        'val', datasets_path, input_shape, batch_size)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=30,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        # use_multiprocessing=True,
        workers=4,
        callbacks=[checkpoint, early_stopping]
    )
    model.save_weights('my-tiny-yolov1.hdf5')


if __name__ == '__main__':
    _main()
