base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalMaxPooling2D()(x)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
prediction_layer = tf.keras.layers.Dense(196, activation='softmax')(x)

"""

"""

model_2 = Model(inputs=base_model.input, outputs=prediction_layer)
model_2.summary()

model_2.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
hist = model_2.fit(cars_train_pp, epochs=epochs, validation_data=cars_val_pp, workers=n_workers)
model_2.save('model_2')
