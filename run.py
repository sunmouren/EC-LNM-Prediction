import sys 

import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

model_index = int(sys.argv[1])

input_shape = (224, 224)

def load_img(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, input_shape)
    # image = tf.image.rgb_to_grayscale(image)
    return image


def map_ds_x0(paths, labels):
    image = load_img(paths[0])
    return image

def map_ds_x1(paths, labels):
    image = load_img(paths[1])
    return image

def map_ds_y(paths, labels):
    return labels

df = pd.read_csv("dataset.csv")

df_train = df[df['ds'] == 'train']
df_test = df[df['ds'] == 'test']

train_x, train_y = df_train[['path1', 'path2']].values, df_train[['label']]
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds_x0 = np.array(list(train_ds.map(map_ds_x0).as_numpy_iterator())) 
train_ds_x1 = np.array(list(train_ds.map(map_ds_x1).as_numpy_iterator())) 
train_ds_y = np.array(list(train_ds.map(map_ds_y).as_numpy_iterator())) 

test_x, test_y = df_test[['path1', 'path2']].values, df_test[['label']]
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_ds_x0 = np.array(list(test_ds.map(map_ds_x0).as_numpy_iterator())) 
test_ds_x1 = np.array(list(test_ds.map(map_ds_x1).as_numpy_iterator())) 
test_ds_y = np.array(list(test_ds.map(map_ds_y).as_numpy_iterator()))

print(train_ds_x0.shape, test_ds_x0.shape)


name_list = ['ResNet101V2', 'InceptionResNetV2', 'EfficientNetV2S', 'ConvNeXtTiny']

tf.random.set_seed(42)

run_times = 1

model_name = name_list[model_index]

print(f'\n\n{model_name=}\n\n')


tf.keras.backend.clear_session()

input_tensor = tf.keras.layers.Input(shape=(*input_shape, 3))

if model_name == 'ResNet101V2':
    
    base_model = tf.keras.applications.ResNet101V2(
        weights='imagenet', 
        include_top=False, 
        input_tensor=input_tensor,
        # input_shape=(*input_shape, 3),
        # include_preprocessing=True,
    )
    
if model_name == 'InceptionResNetV2':
    
    base_model = tf.keras.applications.InceptionResNetV2(
        weights='imagenet', 
        include_top=False, 
        input_tensor=input_tensor,
        # input_shape=(*input_shape, 3),
        # include_preprocessing=False,
    )
    
    
if model_name == 'ConvNeXtTiny':
    
    base_model = tf.keras.applications.ConvNeXtTiny(
        weights='imagenet', 
        include_top=False, 
        input_tensor=input_tensor,
        # input_shape=(*input_shape, 3),
        include_preprocessing=False,
    )
    
        
if model_name == 'EfficientNetV2S':
    
    base_model = tf.keras.applications.EfficientNetV2S(
        weights='imagenet', 
        include_top=False, 
        input_tensor=input_tensor,
        # input_shape=(*input_shape, 3),
        include_preprocessing=False,
    )
    
for layer in base_model.layers:
    layer.trainable = True

in0 = tf.keras.layers.Input(shape=(*input_shape, 3))

x = base_model(in0)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(.2)(x)

x = tf.keras.layers.Dense(1, activation=None, name='latent_out_1')(x)

x = tf.keras.models.Model(inputs=in0, outputs=x)


in1 = tf.keras.layers.Input(shape=(*input_shape, 3))

y = base_model(in1)

y = tf.keras.layers.GlobalAveragePooling2D()(y)

y = tf.keras.layers.Dense(1024, activation='relu')(y)
y = tf.keras.layers.Dropout(.2)(y)
y = tf.keras.layers.Dense(1, activation=None, name='latent_out_2')(y)

y = tf.keras.models.Model(inputs=in1, outputs=y)

# # # 计算注意力权重
# attention = tf.keras.layers.Dense(1, activation='sigmoid')(tf.concat([x.output, y.output], axis=-1))  # (batch_size, 1)
# combined = attention * x.output + * y.output

# combined = tf.keras.layers.Add()([x.output , y.output])

combined = tf.keras.layers.concatenate([x.output , y.output])

z = tf.keras.layers.Dense(128, activation='relu')(combined)
z = tf.keras.layers.Dense(1, activation='sigmoid')(z)

model = tf.keras.models.Model(inputs=[x.input, y.input], outputs=z)

model_name = f'{model_name}-{run_times}'

opt = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=opt,
                loss='mse',
                metrics=['acc'])

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{model_name}-best.h5',
        monitor='loss',
        save_best_only=True,
        save_weights_only=True,
    ),
]


model.fit(
    x=[train_ds_x0, train_ds_x1],
    y=train_ds_y,
    validation_data=([test_ds_x0, test_ds_x1], 
                        test_ds_y),
    callbacks=my_callbacks,
    epochs=50,
    batch_size=16,
)

df = pd.DataFrame(data=list(model.history.history.values()))
df_t = df.T
df_t.columns = list(model.history.history.keys())
df_t.to_csv(f'{model_name}-history.csv', index=False)
model.save_weights(f'{model_name}-last.h5')


def save_result(ds):
    if ds == 'train':
        inputs = [train_ds_x0, train_ds_x1]
        phs = [i[0].split('/')[-1].split('_seg_')[0] for i in train_x]
        lb = train_ds_y
        model.load_weights(f'{model_name}-best.h5')
    else:
        inputs = [test_ds_x0, test_ds_x1]
        phs = [i[0].split('/')[-1].split('_seg_')[0] for i in test_x]
        lb = test_ds_y
        model.load_weights(f'{model_name}-best.h5')

    in1, in2 = model.get_layer('input_2').input,  model.get_layer('input_3').input
    out1, out2 = model.get_layer('latent_out_1').output,  model.get_layer('latent_out_2').output
    
    ft_model = tf.keras.Model(inputs=[in1, in2], outputs=[out1, out2])
    preds = model.predict(inputs)
   
    preds = np.round(preds, 2).flatten()

    fts = ft_model.predict(inputs)
   
    fts1, fts2 = fts[0].flatten(), fts[1].flatten()
    df = pd.DataFrame([phs, lb, preds, fts1, fts2])
   
    df = df.T
    df.columns = ['file', 'lb', 'preds', 'fts1', 'fts2']
    
    df.to_csv(f'{model_name}-result-{ds}-best.csv', index=False)

save_result('train')
save_result('test')   

