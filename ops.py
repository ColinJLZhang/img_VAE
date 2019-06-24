import tensorflow as tf

def conv2d(x, in_channels, output_channels, name, strides=[1,2,2,1], reuse = False):
	'''Convolutional Layer'''
	with tf.variable_scope(name,  reuse = reuse):
		w = tf.get_variable('w', [5, 5, in_channels, output_channels], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_channels], initializer = tf.constant_initializer(0.1))

		conv = tf.nn.conv2d(x, w, strides = strides, padding = 'SAME') + b
		return conv

def deconv2d(x, output_shape, name, strides = [1,2,2,1], reuse = False):
	'''Deconvolutional Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('w', [5, 5, output_shape[-1], int(x.get_shape()[-1])], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_shape[-1]], initializer = tf.constant_initializer(0.1))

		deconv = tf.nn.conv2d_transpose(x, w, output_shape = output_shape, strides = strides) + b
		return deconv

def dense(x, input_dim, output_dim, name, reuse = False):
	'''Fully-connected Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('w', [input_dim, output_dim], initializer = tf.truncated_normal_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer(0.1))

		return tf.matmul(x, w) + b


if __name__ == '__main__':
	'''test all ops'''
	x1 = tf.random_normal([1,9,9,3])
	c1 = conv2d(x1, 3, 5, "c1")
	dc1 = deconv2d(c1, [1,9,9,3], "decon1")
	x2 = tf.random_normal([1,9])
	d1 = dense(x2, 9, 5, "dense1")
	
	print("x1:", x1)
	print("c1:", c1)
	print("dc1:", dc1)
	print("x2:", x2)
	print("d1:", d1)