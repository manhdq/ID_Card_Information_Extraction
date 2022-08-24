
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class OCRCommon:
    def __init__(self, config_name='vgg_transformer', weights=None, device='cuda:0'):
        config = Cfg.load_config_from_name(config_name)
        # print(config)
        # config['weights'] = '/home/manhdq/ID_Card_Information_Extraction/informationExtractor/vietocr/weights/transformerocr.pth'
        # config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'

        if weights is not None:
            config['weights'] = weights
            config['cnn']['pretrained'] = False

        config['device'] = device
        # config['predictor']['beamsearch'] = False

        self.detector = Predictor(config)

    def predict(self, img):
        img = Image.fromarray(img)

        return self.detector.predict(img)