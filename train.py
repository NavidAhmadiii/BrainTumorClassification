from tensorflow.keras.callbacks import EarlyStopping


def train_model(model, train_generator, val_generator, epochs=11):
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[early_stop]
    )
    return history


# white EarlyStopping

# def train_model(model, train_generator, val_generator, epochs=20):
#     # حذف EarlyStopping از callbacks
#     history = model.fit(
#         train_generator,
#         validation_data=val_generator,
#         epochs=epochs,
#         steps_per_epoch=train_generator.samples // train_generator.batch_size,
#         validation_steps=val_generator.samples // val_generator.batch_size,
#         callbacks=[]  # لیست callbacks خالی است
#     )
#     return history
