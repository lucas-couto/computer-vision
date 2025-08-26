from models import (
    Cnn,
    Resnet152,
    Convnext,
    Densenet,
    Efficientnet,
    Inception,
    Nasnet,
    Resnet50, 
    XceptionModel, 
)

def get_models():
    return [
    Cnn,Resnet152,Convnext,  
    Densenet, Efficientnet,
    Inception, Nasnet,
    Resnet50, XceptionModel
]