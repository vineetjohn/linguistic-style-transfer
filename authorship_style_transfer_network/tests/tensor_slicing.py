import tensorflow as tf

with tf.Session() as sess:
    foo = tf.constant(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ])
    print(foo)
    print(sess.run(tf.slice(foo, [0, 0, 0], [foo.shape[0], foo.shape[1], 1])))
