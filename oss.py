import subprocess
import re
import os

l = os.listdir('/root/douzero1/DouZero/douzero_checkpoints/res')


pattern_down = re.compile(r'^landlord_down_weights_(\d+)')
pattern_up = re.compile(r'^landlord_up_weights_(\d+)')
down_f = [f for f in l if pattern_down.match(f)]


def get_num(x):
    y = x.split('_')[-1]
    z = y.split('.')[0]
    return int(z)

down_f_num = [get_num(x) for x in down_f]
down_f_num.sort()

down_f = ['landlord_down_weights_'+str(x) + '.ckpt' for x in down_f_num]
up_f = ['landlord_up_weights_'+str(x) + '.ckpt' for x in down_f_num]

down_f = [os.path.join('/root/douzero1/DouZero/douzero_checkpoints/res', x) for x in down_f]
up_f = [os.path.join('/root/douzero1/DouZero/douzero_checkpoints/res', x) for x in up_f]

for i in range(len(down_f)):
    command = [
    'python3',
    'evaluate.py',
    '--landlord',
    '/root/douzero1/DouZero/baselines/douzero_WP/landlord.ckpt',
    '--landlord_up',
    up_f[i],
    '--landlord_down',
    down_f[i]
]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 打印标准输出
        print('{}/{}'.format(i+1, len(down_f)))
        print("STDOUT:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred while trying to run the command.")
        print("Return code:", e.returncode)
        print("Output:", e.output)