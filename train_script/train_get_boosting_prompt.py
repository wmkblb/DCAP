import os

dataset_list = [
    #['fgvc_aircraft', 'data_path'],
    #['food101', 'data_path'],
    #['caltech101', 'data_path'],
    #['oxford_pets', 'data_path'],
    #['stanford_cars', 'data_path'],
    #['oxford_flowers', 'data_path'],
    #['sun397', 'data_path'],
    #['dtd', 'data_path'],
    #['eurosat', 'data_path'],
    #['ucf101', 'data_path'],
    #['imagenet', 'data_path'],
]
GPU_NUM=0
SAVE_PATH="save_path/dcap_teacher"
CFG_S1_DICT = [["vit_b16_c2_ep20_batch4_8+8ctx_l12",8]]# other datasets
#CFG_S1_DICT = [["vit_b16_c2_ep20_batch128_lr0025_8+8ctx_l12", 8]] #imagenet

for (DATASET, DATA) in dataset_list:
    print(DATASET, DATA)
    for (CFG_S1, KD_N_CTX_TEACHER) in CFG_S1_DICT:
        # traing the first step
        OUT_PUT_TEACHER = "dcap_s1_v" + str(KD_N_CTX_TEACHER) + "_t" + str(KD_N_CTX_TEACHER) + "_s2_v" + str(0) + "_t" + str(0)
        DIR_NAEM = "teacher"
        for COUNTER in range(1, 2):
            cmd = f"CUDA_VISIBLE_DEVICES={GPU_NUM} bash scripts/dcap/get_boosting_prompt_teacher/train_get_boosting_prompt_teacher.sh \
                  {DATA} {DATASET} {COUNTER} {OUT_PUT_TEACHER} {DIR_NAEM} {CFG_S1} {SAVE_PATH}"
            os.system(cmd)

















