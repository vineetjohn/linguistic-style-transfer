import tensorflow as tf

with tf.Session() as sess:
    foo = tf.constant(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

    # bar = tf.constant(
    #     [
    #         [1, 2, 3],
    #         [4, 5, 6]
    #     ])
    #
    # bar_1 = tf.add([1, 2, 3], 1)
    # baz = tf.fill(dims=[bar.shape[0]], value=7)
    # baz = tf.tile(input=tf.constant([[7]]), multiples=[2, 1])

    # print(foo)
    # print(bar)
    # print(sess.run(bar))
    # print(sess.run(bar_1))
    # print(sess.run(baz))
    # print(sess.run(tf.concat([baz, bar], axis=1)))
    print(sess.run(tf.strided_slice(foo, [0, 0], [3, -1], [1, 1], name='slice_input')))
