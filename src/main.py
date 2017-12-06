import argparse
import numpy as np
import os
import tensorflow as tf
import vgg
import utils

assets_dir = "../assets"
output_dir = "../output"

def get_content_cost(C, G):
    return tf.reduce_sum(tf.square(G - C)) / 2

def get_style_cost(S, G):
    g = tf.reshape(G, [-1, int(G.shape[3])])
    s = tf.reshape(S, [-1, int(S.shape[3])])
    gram_S = tf.matmul(tf.transpose(s), s)
    gram_G = tf.matmul(tf.transpose(g), g)
    shape = int(np.prod(S.shape[1:3]))
    return tf.reduce_sum(tf.square(gram_G - gram_S)) * 1.0 / (4.0 * shape * shape)

def main(num_iter, use_optimizer, weights_path, content_image_path,
         style_image_path, verbose=False):
    content = np.float32(utils.normalize_image(np.reshape(utils.open_image(content_image_path),
                                                          [1, 224, 224, 3])))
    style = np.float32(utils.normalize_image(np.reshape(utils.open_image(style_image_path),
                                                        [1, 224, 224, 3])))
    gen = np.float32(utils.normalize_image(np.reshape(utils.create_noise_image(), [1, 224, 224, 3])))
    gen_image = tf.Variable(gen, trainable=True, dtype=tf.float32)
    content_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    style_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

    vgg_gen = vgg.VGG(weights_path)
    modelC = vgg_gen.create_model(content_placeholder, scope="content")
    modelS = vgg_gen.create_model(style_placeholder, scope="style")
    modelG = vgg_gen.create_model(gen_image, scope="gen")
    with tf.Session() as sess:
        alpha = tf.constant(1.0, dtype=tf.float32)
        beta = tf.constant(1.0, dtype=tf.float32)
        gamma = tf.constant(1.0, dtype=tf.float32)
        content_cost = get_content_cost(modelC["conv4_2"], modelG["conv4_2"])
        style_cost = get_style_cost(modelS["conv4_2"], modelG["conv4_2"])
        total_variation_cost = tf.reduce_sum(tf.image.total_variation(gen_image))
        cost = tf.add(tf.add(alpha * content_cost, beta * style_cost),
                      gamma * total_variation_cost)
        if use_optimizer == 'adam':
            learning_rate = 1.0
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_step = optimizer.minimize(cost)
            optimizer_slots = []
            if isinstance(optimizer, tf.train.AdamOptimizer):
                optimizer_slots.extend([optimizer._beta1_power,
                                        optimizer._beta2_power])

            init_optimizer = tf.variables_initializer(optimizer_slots)
            sess.run(init_optimizer)
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(num_iter):
                _, c = sess.run([train_step, cost],
                         feed_dict={content_placeholder: content,
                                    style_placeholder: style})
                print(c)

                if verbose and i % 50 == 0:
                    utils.show_image(utils.restore_image(sess.run(gen_image)[0]))
        else:
            def loss_callback(c, g, content_cost, style_cost, total_variation_cost):
                if verbose and loss_callback.iter % 50 == 0:
                    utils.show_image(utils.restore_image(g[0]))

                print(content_cost, style_cost, total_variation_cost)
                loss_callback.iter += 1

            loss_callback.iter = 0
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, method='L-BFGS-B', options={'maxiter': num_iter})
            init = tf.global_variables_initializer()
            sess.run(init)
            optimizer.minimize(sess, feed_dict={content_placeholder: content,
                                                style_placeholder: style},
                               fetches=[cost, gen_image, content_cost, style_cost, total_variation_cost],
                               loss_callback=loss_callback)

        output = utils.restore_image(sess.run(gen_image)[0])
        if verbose:
            utils.show_image(output)

        utils.save_image(os.path.join(output_dir, "output.jpg"), output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("-content_image", type=str)
    parser.add_argument("-style_image", type=str)
    parser.add_argument("-num_iter", type=int, default=500)
    parser.add_argument("-optimizer", type=str, default="adam")
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()
    weights_file = "model/imagenet-vgg-verydeep-19.mat"
    main(num_iter=args.num_iter,
         use_optimizer=args.optimizer,
         weights_path=os.path.join(assets_dir, weights_file),
         content_image_path=args.content_image,
         style_image_path=args.style_image,
         verbose=args.verbose)
