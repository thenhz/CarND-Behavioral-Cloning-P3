import cv2
import numpy as np
import sklearn

# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

IMG_HT, IMG_WIDTH, IMG_CH = 66, 200, 3

def preprocess(img):
    img = img[40:-20, :, :]
    img = cv2.resize(img, (IMG_WIDTH, IMG_HT), cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def augment_img(img, steering):
    def random_shift(img, steering, shift_range=20):
        ht, wd, ch = img.shape

        shift_x = shift_range * (np.random.rand() - 0.5)
        shift_y = shift_range * (np.random.rand() - 0.5)
        shift_m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, shift_m, (wd, ht))

        steering += shift_x * 0.002
        return img, steering

    def adjust_brightness(img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv_img[:, :, 2] = hsv_img[:, :, 2] * ratio
        return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    def flip(img, steering):
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            if (steering != 0):
                steering = -steering
        return img, steering

    img, steering = flip(img, steering)
    img, steering = random_shift(img, steering)
    img = adjust_brightness(img)
    return img, steering


def get_train_test_labels(df, left_correction=0.0, right_correction=0.0):
    def get_relative_path(img_path):
        relative_path = './' + img_path.split('/')[-3] + '/' + img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
        return relative_path

    images = []
    steering_angle_list = []
    for index, row in df.iterrows():
        left_img = get_relative_path(row['Left Image'])
        right_img = get_relative_path(row['Right Image'])
        center_img = get_relative_path(row['Center Image'])
        angle = float(row['Steering Angle'])

        # Adjust the left and right steering angle
        langle = angle + left_correction
        rangle = angle - right_correction

        # Read Image
        limg = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)  # ndimage.imread(left_img)
        cimg = cv2.cvtColor(cv2.imread(center_img), cv2.COLOR_BGR2RGB)  # ndimage.imread(center_img)
        rimg = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)  # ndimage.imread(right_img)

        # Augment the Image
        limg, langle = augment_img(limg, langle)
        cimg, angle = augment_img(cimg, angle)
        rimg, rangle = augment_img(rimg, rangle)

        # Preprocess the augmented images to feed to model
        limg = preprocess(limg)
        cimg = preprocess(cimg)
        rimg = preprocess(rimg)

        images.append(limg)
        steering_angle_list.append(langle)
        images.append(cimg)
        steering_angle_list.append(angle)
        images.append(rimg)
        steering_angle_list.append(rangle)

    return images, steering_angle_list


def generator(df, bs=32):
    total = len(df)
    while 1:
        sklearn.utils.shuffle(df)
        for offset in range(0, total, bs):
            batch = df[offset:offset + bs]
            images, angles = get_train_test_labels(batch, 0.2, 0.2)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
