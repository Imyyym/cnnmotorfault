# import tensorflow as tf
# print("TF 版本:", tf.__version__)
# print(tf.sysconfig.get_build_info()["cuda_version"])
# print(tf.sysconfig.get_build_info()["cudnn_version"])
# print(tf.config.list_physical_devices())

#
# import subprocess
# print(subprocess.getoutput("nvidia-smi"))
#


import tensorflow as tf
import sys

print("=" * 50)
print("Python版本:", sys.version)
print("TensorFlow版本:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("🎉 成功！找到GPU:", gpus)

    # 测试GPU计算
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        print("GPU计算测试:", c.numpy())
else:
    print("❌ 未找到GPU")

print("=" * 50)