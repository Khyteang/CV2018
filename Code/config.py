class Configuration:
    SCALED_IMAGE_SIZE = 256

    SCALED_MASK_SIZE = 68

    IMAGES_PER_GPU = 8

    GPU_COUNT = 2

    BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT

    LEARNING_RATE = 0.1

    VALIDATION_STEPS = 50

    STEPS_PER_EPOCH = 1000 / BATCH_SIZE

    VARIABLE_STRATEGY = 'GPU'

    WEIGHT_DECAY = 2e-4

    MOMENTUM = 0.9

    MODEL_SAVE_DIR = '../Models/unet_dense'

    IMAGES_DIR = '../Dataset/stage1_train'

    DATASET_DIR = '../Dataset/stage1_train_processed'
    
    EVAL_IMAGES_DIR = '../Dataset/stage1_test'
    
    EVAL_DATASET_DIR = '../Dataset/stage1_test_processed'

    VALIDATION_SPLIT_RATIO = 0.2