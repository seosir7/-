import keras
keras.__version__

from keras.datasets import imdb # 케라스에 있는 영화 리뷰 라이브러리 가져옴 

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) 
# 트레이닝 테스트 data,label 분리 / 나온 단어 빈도수  탑 10000 안에 드는 것만 가져옴

print('data',train_data[0]) 
# [1, 14, 22, 16, 43, 530, 973, 1622,....] 리스트 숫자형태로 나옴 
# 각 단어가 인덱싱 되어 엔코딩 되어있는 것 디코딩하면 다시 문자 - 문장 형태로 나옴
print('label',train_labels[0]) 
# 0, 1중 하나 0이면 부정적리뷰 1이면 긍정적 리뷰

max([max(sequence) for sequence in train_data]) 
# 빈도수 10000개로 범위설정 했기 때문에 단어 인덱스가 10000을 넘기지 않고 / 9999가 출력됨

# word_index는 단어와 정수 인덱스를 매핑한 딕셔너리입니다
word_index = imdb.get_word_index() 
# print('word_index:',word_index) 
# word_index: {'fawn': 34701, 'tsukino': 52006, 'nunnery': 52007, 'sonja': 16816, 
# 정수 인덱스와 단어를 매핑하도록 뒤집습니다
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# print('reverse_word_index:',reverse_word_index) #{34701: 'fawn', 52006: 'tsukino', 52007: 'nunnery', 16816: 'sonja'

# 리뷰를 디코딩합니다. 
# 0, 1, 2는 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺍니다
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# join - 각 값마다 한칸 띄어쓰기 해주고,
# get - 딕셔너리의 키 값 넣어주면 value 값나옴 '?' 디폴트 값이 나오면 ?  
# 즉 train_data[0]에 있는 데이터를 i로 받아와서 reverse_word_index.get의 키에 i를 넣어주는데
#


print('리뷰',decoded_review)
#  ? this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert ? 

import numpy as np

# train_data를 이함수에 넣으면 0으로 채워진 행렬을 만드는데 (9999,10000) 크기로 만듬 
# 
def vectorize_sequences(sequences, dimension=10000):
    # 크기가 (len(sequences), dimension))이고 모든 원소가 0인 행렬을 만듭니다
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # results[i]에서 특정 인덱스의 위치를 1로 만듭니다
        # 가령 i= 1 sequece가 [1,4,6,8,9,10] 나왔다 치면
        # 행렬의 1행의 1,4,6,8,9, 10 열은 1로 값을 넣어주라는 뜻
        # 결국 result 행렬은 0과 1로 이루어진 2차원 배열이 됨


    return results
 
# 훈련 데이터를 벡터로 변환합니다
x_train = vectorize_sequences(train_data)
# 테스트 데이터를 벡터로 변환합니다
x_test = vectorize_sequences(test_data)
 
print('x_train[0]',x_train[0])
 
# 레이블을 벡터로 바꿉니다
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
 
from keras import models
from keras import layers
 
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
 
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
 
from keras import optimizers
 
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

 
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
 
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
 

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
 
history_dict = history.history
print(history_dict.keys())
 

import matplotlib.pyplot as plt
 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(1, len(acc) + 1)
 
# ‘bo’는 파란색 점을 의미합니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# ‘b’는 파란색 실선을 의미합니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
 
from keras import losses
from keras import metrics
 
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
 
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
