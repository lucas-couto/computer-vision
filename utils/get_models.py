from models import (
    Cnn,
    ResNet152Model,
    ConvNeXtBaseModel,
    DensenetModel,
    EfficientNetB7Model,
    InceptionResNetV2Model,
    NASNetLargeModel,
    Resnet50Model, 
    XceptionModel, 
)

def get_models():
    return [
    Cnn,ResNet152Model,ConvNeXtBaseModel,  
    DensenetModel, EfficientNetB7Model,
    InceptionResNetV2Model, NASNetLargeModel,
    Resnet50Model, XceptionModel
]