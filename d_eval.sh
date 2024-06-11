echo "small vs res"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/small/landlord_weights_15318400.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/res/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/res/down_23.ckpt
echo "small vs big"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/small/landlord_weights_15318400.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/big/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/big/down_23.ckpt
echo "small vs base"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/small/landlord_weights_15318400.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/base/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/base/down_23.ckpt

echo "res vs small"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/res/landlord_weights_12208000.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/small/up_24.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/small/down_23.ckpt
echo "res vs big"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/res/landlord_weights_12208000.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/big/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/big/down_23.ckpt
echo "res vs base"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/res/landlord_weights_12208000.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/base/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/base/down_23.ckpt

echo "big vs small"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/big/landlord_weights_14166400.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/small/up_24.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/small/down_23.ckpt
echo "big vs res"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/big/landlord_weights_14166400.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/res/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/res/down_23.ckpt
echo "big vs base"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/big/landlord_weights_14166400.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/base/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/base/down_23.ckpt

echo "base vs small"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/base/landlord_weights_14400000.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/small/up_24.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/small/down_23.ckpt
echo "base vs res"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/base/landlord_weights_14400000.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/res/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/res/down_23.ckpt
echo "base vs big"
python3 evaluate.py --landlord /root/douzero1/DouZero/douzero_checkpoints/base/landlord_weights_14400000.ckpt --landlord_up  /root/douzero1/DouZero/douzero_checkpoints/big/up_23.ckpt --landlord_down /root/douzero1/DouZero/douzero_checkpoints/big/down_23.ckpt