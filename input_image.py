import numpy as np
from PIL import Image


def preprocess_imagenet(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype("float32")
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def yolo_preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype="float32")
    image_data /= 255.0
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


# def get(model):
#     np.random.seed(0)
#     return np.asarray(np.random.uniform(model.shape), dtype=np.float32)


def get(model, batch_size=8):
    img = Image.open("images/dog.jpg")
    img = img.resize((224, 224), Image.BICUBIC)
    input = np.asarray(img)
    input = np.asarray([input[:, :, 0], input[:, :, 1], input[:, :, 2]])

    input_wrapped = [input for i in range(batch_size)]
    input_wrapped = np.asarray(input_wrapped).astype(np.float32)
    return input_wrapped

