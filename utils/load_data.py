from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(config):
    input_shape = tuple(config['model']['input_shape'])
    train_dir = config['paths']['train_dir']
    valid_dir = config['paths']['valid_dir']
    batch_size = config['training']['batch_size']

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    # Carregar os dados de treino e validação
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    validation_data = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Mantendo a ordem para avaliação
    )

    # Obter rótulos das classes para treino e validação
    train_labels = train_data.classes
    validation_labels = validation_data.classes

    return train_data, train_labels, validation_data, validation_labels