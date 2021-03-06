import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, c_out, verbose=False, build=True, batch_size=64, lr=0.001,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500,
                 patience=60, monitor_metric='val_mean_squared_error', save=False, callbacks=[]):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.00001)
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=patience, verbose=0, restore_best_weights=True)
        self.callbacks = [es, reduce_lr] + callbacks
        
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose
        self.save = save

        if build == True:
            self.model = self.build_model(input_shape, c_out)
            if (verbose == True):
                self.model.summary()
            if (self.save == True):
            	self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > self.bottleneck_size:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, c_out):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(c_out, activation='linear')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(self.lr),
                      metrics=['mean_squared_error'])

        file_path = self.output_directory + 'best_model.hdf5'
        
        if (self.save == True):
        	model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
        	self.callbacks = self.callbacks + [model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, nb_epochs, batch_size):
        if not tf.test.is_gpu_available:
            print('error no gpu')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()
        
        if x_val is None or y_val is None:
        	val_data = None
        	self.callbacks = self.callbacks[1:] # no early stopping
        else:
        	val_data = (x_val, y_val)

        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=val_data, callbacks=self.callbacks)

        duration = time.time() - start_time

        if (self.save == True):
        	self.model.save(self.output_directory + 'last_model.hdf5')

        # y_pred = self.predict(x_val, y_true, x_train, y_train, y_val, return_df_metrics=False)

        # save predictions
        #np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        # y_pred = np.argmax(y_pred, axis=1)

        keras.backend.clear_session()

        return hist.history

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        if (self.save == True):
        	model_path = self.output_directory + 'best_model.hdf5'
        	model = keras.models.load_model(model_path)
        else:
        	model = self.model
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        return y_pred
        
        
def inceptionnet_architecture(nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, c_out=1, 
				kernel_size=40, stride=1, activation='linear', bottleneck_size = 32):

    def architecture(x):
        
        def _shortcut_layer(input_tensor, out_tensor):
            shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                             padding='same', use_bias=False)(input_tensor)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            x = keras.layers.Add()([shortcut_y, out_tensor])
            x = keras.layers.Activation('relu')(x)
            return x

        def _inception_module(input_tensor, activation='linear'):

            if use_bottleneck and int(input_tensor.shape[-1]) > bottleneck_size:
                input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                      padding='same', activation=activation, use_bias=False)(input_tensor)
            else:
                input_inception = input_tensor

      
            kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

            conv_list = []

            for i in range(len(kernel_size_s)):
                conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                     strides=stride, padding='same', activation=activation, use_bias=False)(
                    input_inception))

            max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

            conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                         padding='same', activation=activation, use_bias=False)(max_pool_1)

            conv_list.append(conv_6)

            x = keras.layers.Concatenate(axis=2)(conv_list)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activation='relu')(x)
            return x

        def _shortcut_layer(input_tensor, out_tensor):
            shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                             padding='same', use_bias=False)(input_tensor)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            x = keras.layers.Add()([shortcut_y, out_tensor])
            x = keras.layers.Activation('relu')(x)
            return x

        input_res = x

        for d in range(depth):

            x = _inception_module(x)

            if use_residual and d % 3 == 2:
                x = _shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(c_out, activation='linear')(gap_layer)
            
        return(output_layer)


    return(architecture)


