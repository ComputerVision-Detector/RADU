"""Checkpoint recognition using exactly the configured architecture and dictionary."""
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
PADDLE_ROOT = ROOT / 'src/cv/ocr/PaddleOCR'


class RecognitionRuntime:
    def __init__(self, config_path, checkpoint, device='cpu'):
        sys.path.insert(0, str(PADDLE_ROOT))
        import paddle
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        from ppocr.data.imaug.rec_img_aug import resize_norm_img_chinese
        self.paddle = paddle
        self.resize = resize_norm_img_chinese
        paddle.set_device(device)
        self.config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
        self.decoder = build_post_process(self.config['PostProcess'], self.config['Global'])
        n = len(self.decoder.character)
        self.config['Architecture']['Head']['out_channels_list'] = {
            'CTCLabelDecode': n, 'SARLabelDecode': n+2, 'NRTRLabelDecode': n+3}
        self.model = build_model(self.config['Architecture'])
        path = Path(checkpoint)
        if path.suffix != '.pdparams':
            path = Path(str(path) + '.pdparams')
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint does not exist: {path}')
        weights = paddle.load(str(path))
        expected = self.model.state_dict()
        missing = set(expected) - set(weights)
        mismatched = [k for k in expected if k in weights and list(expected[k].shape) != list(weights[k].shape)]
        if missing or mismatched:
            raise ValueError(f'Checkpoint/config mismatch: missing={sorted(missing)}, shapes={mismatched}')
        self.model.set_state_dict(weights)
        self.model.eval()
        self.checkpoint = path.resolve()
        self.image_shape = next(op['RecResizeImg']['image_shape'] for op in self.config['Eval']['dataset']['transforms'] if 'RecResizeImg' in op)

    def predict(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError('Empty image')
        image, _ = self.resize(bgr_image, self.image_shape)
        with self.paddle.no_grad():
            text, score = self.decoder(self.model(self.paddle.to_tensor(image[np.newaxis])))[0]
        return text, float(score)
