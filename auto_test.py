def eval_pairs(version = 'rlcard,base'):
    import os

    # time_frame_dic1 = {}
    # v1 = version.split(',')[0]
    # model_path1 = os.path.join("douzero_checkpoints",v1)
    # files1 = os.listdir(model_path1)
    # files1 = [os.path.join(model_path1,item) for item in files1 if item.endswith('ckpt')]
    # ids = 0
    # frames = []
    # for file in files1:
    #     frame_id = file.split('_')[-1].split(".")[0]
    #     if frame_id not in frames:
    #         frames.append(frame_id)
    #         ids += 1
    #         time_frame_dic1[ids] = [file]
    #     else:
    #         time_frame_dic1[ids].append(file)

    time_frame_dic2 = {}
    v2 = version.split(',')[1]
    model_path2 = os.path.join("douzero_checkpoints",v2)
    files2 = os.listdir(model_path2)
    files2 = [os.path.join(model_path2,item) for item in files2 if item.endswith('ckpt')]
    ids = 0
    frames = []
    for file in files2:
        frame_id = file.split('_')[-1].split(".")[0]
        if frame_id not in frames:
            frames.append(frame_id)
            ids += 1
            time_frame_dic2[ids] = [file]
        else:
            time_frame_dic2[ids].append(file)
    # print(time_frame_dic1)
    print('rlcard')
    print(time_frame_dic2)

    # check_length = min(max(time_frame_dic1.keys()),max(time_frame_dic2.keys()))
    # import subprocess
    # for i in range(1,check_length+1):
    #     landlord = [item for item in time_frame_dic1[i] if 'landlord_weights' in item][0]
    #     landlord_up = [item for item in time_frame_dic2[i] if 'landlord_up_weights' in item][0]
    #     landlord_down = [item for item in time_frame_dic2[i] if 'landlord_down_weights' in item][0]
    #     linux_str = f"python3 evaluate.py --landlord {landlord} --landlord_up {landlord_up} --landlord_down {landlord_down}"
        
    #     print(f"time : {i}, landlord:{v1}, landlord_up:{v2}, landlord_down:{v2}")
    #     print(linux_str)

    #     command = [
    # 'python3', 'evaluate.py', 
    # '--landlord', landlord,
    # '--landlord_up', landlord_up,
    # '--landlord_down', landlord_down
    #     ]

    #     result = subprocess.run(command, capture_output=True, text=True)
    #     print(result.stdout)
        
    # for i in range(1,check_length+1):
    #     landlord = [item for item in time_frame_dic2[i] if 'landlord_weights' in item][0]
    #     landlord_up = [item for item in time_frame_dic1[i] if 'landlord_up_weights' in item][0]
    #     landlord_down = [item for item in time_frame_dic1[i] if 'landlord_down_weights' in item][0]

    #     print(f"time : {i}, landlord:{v2}, landlord_up:{v1}, landlord_down:{v1}")

    #     linux_str = f"python3 evaluate.py --landlord {landlord} --landlord_up {landlord_up} --landlord_down {landlord_down}"
    #     # print(linux_str)
    #     command = [
    # 'python3', 'evaluate.py', 
    # '--landlord', landlord,
    # '--landlord_up', landlord_up,
    # '--landlord_down', landlord_down
    #     ]

    #     result = subprocess.run(command, capture_output=True, text=True)
    #     print(result.stdout)
        

eval_pairs(version = 'rlcard,base')
# eval_pairs(version = 'res,big')
# eval_pairs(version = 'res,base')
# eval_pairs(version = 'small,big')
# eval_pairs(version = 'small,base')
# eval_pairs(version = 'big,base')