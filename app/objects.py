from torchvision import transforms as T


# pylint: disable=too-few-public-methods
class BaseTransform(T.Compose):
    pass


class ImageTransform(BaseTransform):
    def __init__(self, transforms):
        _ = transforms
        super().__init__(
            [
                T.ToTensor(),
            ]
        )


class TextTransform(BaseTransform):
    pass
