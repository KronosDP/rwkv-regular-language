MODEL_HYPERPARAMETERS = {
    "D_MODEL": 8,
    "N_LAYER": 4,
    "HEAD_SIZE": 8,
    "FFN_HIDDEN_MULTIPLIER": 4,
    "LORA_DIM_W": 32,
    "LORA_DIM_A": 32,
    "LORA_DIM_V": 16,
    "LORA_DIM_G": 32
}
MAX_LEN = 50
DATASET_FILE_CONFIG = 'regex_dataset.json'
MODEL_CHECKPOINT_PATH_CONFIG = 'rwkv7_fsm_experimental_model.pth'
PAD_TOKEN_CONFIG = '<pad>'