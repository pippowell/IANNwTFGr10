
def preprocess(dataset, batchsize, sequence_len):
    '''
    :param dataset: the dataset to be prepared for input into the network
    :param batchsize: the desired batchsize
    :return: 2 datasets, one each for each of the math problems defined (see below), created after the original database was preprocessed with the
    steps below
    '''

    # Step 1 - General Preprocessing

    # convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # flatten the images into vectors
    # dataset = dataset.map(lambda img, target: (tf.reshape(img, (-1,)), target))

    # input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))

    # data = tf.data.Dataset.zip((dataset.shuffle(2000), dataset.shuffle(2000)))

    # print(dataset.shape)
    # The output of that lambda function should be a tuple of two tensors of shapes (num_images, height, width, 1) and (num_images, 1) or (num_images,)

    # Step 2 - Pairing Data Tuples & Respective Parameterized Targets

    # create a dataset that contains 2000 samples from the overall dataset paired with 2000 other samples
    # data = tf.data.Dataset.zip((dataset.shuffle(2000), dataset.shuffle(2000), dataset.shuffle(2000), dataset.shuffle(2000)))

    # create the dataset for the first math problem (a + b >= 5) - remembering to cast to int versus boolean!
    # first = data.map(lambda x1, x2, x3, x4: (x1[0], x2[0], x3[0], x4[0], x1[1]))
    # second = data.map(lambda x1, x2, x3, x4: (x1[0], x2[0], x3[0], x4[0], x1[1] - x2[1]))
    # third = data.map(lambda x1, x2, x3, x4: (x1[0], x2[0], x3[0], x4[0], x1[1] - x2[1] + x3[1]))
    # fourth = data.map(lambda x1, x2, x3, x4: (x1[0], x2[0], x3[0], x4[0], x1[1] - x2[1] + x3[1] - x4[1]))

    # list = [first, second, third, fourth]

    # dataset = dataset.map(lambda img, target: img, new_target_fnc(target, sequence_len))

    # Step 3 - Batching & Prefetching
    # new = new_target_fnc(data, sequence_len)
    dataset = dataset.batch(sequence_len)

    # labels = np.concatenate([label for img, label in dataset], axis=0)
    labels = [labels for _, labels in dataset.unbatch()]
    new_labels = new_target_fnc(labels, sequence_len)

    # dataset = dataset.map(lambda img, target: img, new_labels)
    dataset = tf.data.Dataset.zip(dataset, new_labels)
    # dataset = dataset.map(lambda img, target1, target2: img, target1))
    # dataset = dataset.map(lambda img, target: img, tf.data.Dataset.from_tensor_slices(target))

    # create a dataset that contains 2000 samples from the overall dataset paired with 2000 other samples
    # dataset = tf.data.Dataset.zip((dataset.shuffle(2000), dataset.shuffle(2000)))

    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # The shape of your tensors should be (batch, sequence-length, features).
    return dataset