import tensorflow as tf


@tf.function
def boo(a, b):
    return tf.sets.union(a, b)


print(boo(tf.constant([[1, 2, 3]]), tf.constant([[7, 8, 9] ])))
