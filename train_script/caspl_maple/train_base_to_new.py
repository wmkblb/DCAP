import os

dataset_list = [
    ['caltech101','data_path'],
    ['oxford_pets', 'data_path'],
    ['stanford_cars', 'data_path'],
    ['oxford_flowers', 'data_path'],
    ['food101', 'data_path'],
    ['fgvc_aircraft', 'data_path'],
    ['sun397', 'data_path'],
    ['dtd', 'data_path'],
    ['eurosat', 'data_path'],
    ['ucf101', 'data_path'],
    ['imagenet', 'data_path'],
]
GPU_NUM=0
SAVE_PATH="save_path/dcap_base_to_novel"
PATH_TEACHER=""
CFG_S1_DICT = [["vit_b16_c2_ep20_batch4_8+8ctx_l12",8, 12]]# other datasets
#CFG_S1_DICT = [["vit_b16_c2_ep20_batch128_lr0025_8+8ctx_l12", 8]] #imagenet
CFG_S2_LIST = [["vit_b16_c8_ep5_batch4_8ctx_l12",8]]

for (DATASET, DATA) in dataset_list:
    print(DATASET, DATA)
    for (CFG_S1, KD_N_CTX_VISION_TEACHER, KD_PROMPT_DEPTH_VISION_TEACHER) in CFG_S1_DICT:
        OUT_PUT_TEACHER = ("home/PromptLearning/tool")
        DIR_NAEM = "teacher"

        for (CFG_S2, N_CTX_VISION_STUDENT) in CFG_S2_LIST:
            OUT_PUT = "dcap_s1_v" + str(KD_N_CTX_VISION_TEACHER) + "_t" + str(KD_N_CTX_VISION_TEACHER) + "_s2_v" + str(N_CTX_VISION_STUDENT) +"_t" + str(N_CTX_VISION_STUDENT)
            RESULT_PATH =  SAVE_PATH + "/output_" + OUT_PUT + "/"
            DIR_NAEM_STUDNET = "student"
            DIR_NAME_TEACHER = DIR_NAEM
            CFG_NAME = CFG_S1
            KD_N_CTX_VISION = KD_N_CTX_VISION_TEACHER
            KD_N_CTX_TEXT = KD_N_CTX_VISION_TEACHER
            PROMPT_DEPTH_VISION = KD_PROMPT_DEPTH_VISION_TEACHER
            PROMPT_DEPTH_TEXT = KD_PROMPT_DEPTH_VISION_TEACHER
            # traing the second step
            for COUNTER in range(1,51):
                cmd = f"CUDA_VISIBLE_DEVICES={GPU_NUM} bash scripts/dcap/dcap/base2new_train.sh \
                      {DATA} {DATASET} {COUNTER} {OUT_PUT} {DIR_NAEM_STUDNET} {DIR_NAME_TEACHER} \
                      {RESULT_PATH} {CFG_S2} {CFG_NAME} \
                      {KD_N_CTX_VISION} {KD_N_CTX_TEXT} \
                      {PROMPT_DEPTH_VISION} {PROMPT_DEPTH_TEXT} \
                      {SAVE_PATH} {OUT_PUT_TEACHER} {PATH_TEACHER}"
                os.system(cmd)
                cmd = f"CUDA_VISIBLE_DEVICES={GPU_NUM} bash scripts/dcap/dcap/base2new_test.sh \
                      {DATA} {DATASET} {COUNTER} {OUT_PUT} {DIR_NAEM_STUDNET}  \
                      {RESULT_PATH} {CFG_S2} \
                      {KD_N_CTX_VISION} {KD_N_CTX_TEXT} \
                      {PROMPT_DEPTH_VISION} {PROMPT_DEPTH_TEXT} \
                      {SAVE_PATH}"
                os.system(cmd)


























