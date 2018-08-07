import os


REMOVE_STOP_WORDS = False
# TRAINING PARAMS
BATCH_SIZE = 128
EPOCHS = 60
CLIP_VALUE = 2
LEARNING_RATE = 0.001

# For Bi-LSTM
EMBEDDING_DIMENSION = 1024
HIDDEN_DIMENSION = 512
VOCAB_SIZE = 7737 #6452
MAX_CAPTION_LEN = 82 #49

# DATA RELATED
VISUAL_FEATURE_DIMENSION = 2048
NO_OF_REGIONS_IN_IMAGE = 72

# MODEL
MARGIN = 0.2

# PATH
#TRAIN_IMAGES_DIR = '/mnt/ssd1/junweil/vision_language/resnet-152/'
TRAIN_IMAGES_DIR = '/data/extDisk2/vvaibhav/vision_language/poyao_bottomup_feats_72/'
CAPTION_INFO = '/data/extDisk2/vvaibhav/vision_language/results_20130124.token'
SPLIT_INFO = '/data/extDisk2/vvaibhav/vision_language/splits/'
IMAGES_DIR = '/data/extDisk2/vvaibhav/vision_language/flickr30k_images/'
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#Local path for testing
#CAPTION_INFO = 'C:\\Users\\myste\\Downloads\\results_20130124.token'
#SPLIT_INFO = 'C:\\Users\\myste\\Downloads\\split\\'
#IMAGES_DIR = 'C:\\Users\\myste\\Downloads\\flickr30k_images\\'
ENTITY_SENT = 'entity_sent.json'
ENTITY_BBOX = 'entity_bbox.json'
PROPOSED_BBOX = '/data/extDisk2/vvaibhav/vision_language/poyao_72_roi_matchEntities_k1_iou0.5.json'
#PROPOSED_BBOX = 'C:\\Users\\myste\\Downloads\\Flickr30kEntities\\poyao_72_roi_matchEntities_k1_iou0.5.json'
#CONCEPT_DIR = 'C:\\Users\\myste\\Downloads\\semantic_feat\\'