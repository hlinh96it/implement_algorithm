import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
import cv2


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self,all_filenames,
                 labels,
                 batch_size,
                 index2class,
                 input_dim,
                 n_channels,
                 n_classes=2,
                 normalize=True,
                 zoom_range=[0.8, 1],
                 rotation=15,
                 brightness_range=[0.8, 1],
                 shuffle=True):
        '''
        all_filenames: list toàn bộ các filename
        labels: nhãn của toàn bộ các file
        batch_size: kích thước của 1 batch
        index2class: index của các class
        input_dim: (width, height) đầu vào của ảnh
        n_channels: số lượng channels của ảnh
        n_classes: số lượng các class 
        normalize: có chuẩn hóa ảnh hay không?
        zoom_range: khoảng scale zoom là một khoảng nằm trong [0, 1].
        rotation: độ xoay ảnh.
        brightness_range: Khoảng biến thiên cường độ sáng
        shuffle: có shuffle dữ liệu sau mỗi epoch hay không?
        '''
        self.all_filenames = all_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.index2class = index2class
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.normalize = normalize
        self.zoom_range = zoom_range
        self.rotation = rotation
        self.brightness_range = brightness_range
        self.on_epoch_end()

    def __len__(self):
        '''
        return:
          Trả về số lượng batch/1 epoch
        '''
        return int(np.floor(len(self.all_filenames) / self.batch_size))

    def __getitem__(self, index):
        '''
        params:
          index: index của batch
        return:
          X, y cho batch thứ index
        '''
        # Lấy ra indexes của batch thứ index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # List all_filenames trong một batch
        all_filenames_temp = [self.all_filenames[k] for k in indexes]

        # Khởi tạo data
        X, y = self.__data_generation(all_filenames_temp)

        return X, y

    def on_epoch_end(self):
        '''
        Shuffle dữ liệu khi epochs end hoặc start.
        '''
        self.indexes = np.arange(len(self.all_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, all_filenames_temp):
        '''
        params:
          all_filenames_temp: list các filenames trong 1 batch
        return:
          Trả về giá trị cho một batch.
        '''
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Khởi tạo dữ liệu
        for i, fn in enumerate(all_filenames_temp):
            # Đọc file từ folder name
            img = cv2.imread(fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_dim)
            img_reshape = img.reshape(-1, 3)

            if self.normalize:
                mean = np.mean(img_reshape, axis=0)
                std = np.std(img_reshape, axis=0)
                img = (img-mean)/std

            if self.zoom_range:
                zoom_scale = 1 / \
                    np.random.uniform(self.zoom_range[0], self.zoom_range[1])
                (h, w, c) = img.shape
                img = cv2.resize(
                    img, (int(h*zoom_scale), int(w*zoom_scale)), interpolation=cv2.INTER_LINEAR)
                (h_rz, w_rz, c) = img.shape
                start_w = np.random.randint(0, w_rz-w) if (w_rz-w) > 0 else 0
                start_h = np.random.randint(0, h_rz-h) if (h_rz-h) > 0 else 0
                # print(start_w, start_h)
                img = img[start_h:(start_h+h), start_w:(start_w+w), :].copy()

            if self.rotation:
                (h, w, c) = img.shape
                angle = np.random.uniform(-self.rotation, self.rotation)
                RotMat = cv2.getRotationMatrix2D(
                    center=(w, h), angle=angle, scale=1)
                img = cv2.warpAffine(img, RotMat, (w, h))

            if self.brightness_range:
                scale_bright = np.random.uniform(
                    self.brightness_range[0], self.brightness_range[1])
                img = img*scale_bright

            label = 'dog' if 'dog' in fn else 'cat'
            label = self.index2class[label]

            X[i, ] = img

            # Lưu class
            y[i] = label
        return X, y
