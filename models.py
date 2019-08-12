from os import path
import glob
import unittest

base_model_dir = path.join(path.expanduser("~"), "data", "carml", "dlperf")


def find_onnx_model(name):
    files = glob.glob(path.join(base_model_dir, name, "**", "*.onnx"), recursive=True)
    if len(files) == 0:
        msg = "unable to find model {}".format(name)
        raise Exception(msg)
    files = [f for f in files if not path.basename(f).startswith(".")]
    batch_files = [f for f in files if path.basename(f) == "model_batch.onnx"]
    if batch_files != []:
        return batch_files[0]
    if len(files) != 1:
        raise Exception("found more than one onnx model {}".format(name))
    return files[0]


class model_url_info:
    def __init__(self, name, url, shape=(1, 224, 224, 3)):
        self.name = name
        self.url = url
        try:
            self.path = find_onnx_model(name)
        except:
            self.path = None
        self.shape = shape

    def __repr__(self):
        return self.path

    def __str__(self):
        return self.name


models = [
    model_url_info(name, url)
    for name, url in [
        (
            "ArcFace",
            "https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz",
        ),
        (
            "BVLC_AlexNet",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz",
        ),
        (
            "BVLC_CaffeNet",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz",
        ),
        (
            "BVLC_GoogleNet",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz",
        ),
        (
            "BVLC_RCNN_ILSVRC13",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_rcnn_ilsvrc13.tar.gz",
        ),
        (
            "DenseNet-121",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/densenet121.tar.gz",
        ),
        (
            "DUC",
            "https://s3.amazonaws.com/onnx-model-zoo/duc/ResNet101_DUC_HDC.tar.gz"),
        (
            "Emotion-FerPlus",
            "https://onnxzoo.blob.core.windows.net/models/opset_8/emotion_ferplus/emotion_ferplus.tar.gz",
        ),
        (
            "Inception-v1",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v1.tar.gz",
        ),
        (
            "Inception-v2",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/inception_v2.tar.gz",
        ),
        (
            "MNIST",
            "https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz",
        ),
        (
            "MobileNet-v2",
            "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz",
        ),
                (
            "ResNet018-v1",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.tar.gz",
        ),
        (
            "ResNet018-v2",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v2/resnet18v2.tar.gz",
        ),
        (
            "ResNet034-v1",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v1/resnet34v1.tar.gz",
        ),
        (
            "ResNet034-v2",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet34v2/resnet34v2.tar.gz",
        ),
        (
            "ResNet050-v1",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.tar.gz",
        ),
        (
            "ResNet050-v2",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz",
        ),
        (
            "ResNet101-v1",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v1/resnet101v1.tar.gz",
        ),
        (
            "ResNet101-v2",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet101v2/resnet101v2.tar.gz",
        ),
        (
            "ResNet152-v1",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v1/resnet152v1.tar.gz",
        ),
        (
            "ResNet152-v2",
            "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet152v2/resnet152v2.tar.gz",
        ),
        (
            "Shufflenet",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/shufflenet.tar.gz",
        ),
        (
            "Squeezenet-v1.1",
            "https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz",
        ),
        (
            "Tiny_YOLO-v2",
            "https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz",
        ),
        (
            "VGG16-BN",
            "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.tar.gz",
        ),
        ("VGG16", "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz"),
        (
            "VGG19-BN",
            "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.tar.gz",
        ),
        ("VGG19", "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.tar.gz"),
        (
            "Zfnet512",
            "https://s3.amazonaws.com/download.onnx/models/opset_9/zfnet512.tar.gz",
        ),
    ]
]


class TestModelList(unittest.TestCase):
    def test_models_init(self):
        self.assertEqual(len(models), 30)


if __name__ == "__main__":
    unittest.main()
