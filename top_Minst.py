import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # 테스트 파일 트레인 파일 분리 위해 
from keras.utils.np_utils import to_categorical # 이진법으로 만들기 위해
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = train[:1000]
test = test[:500]
print(train.describe())
train.info()

# PREPARE DATA FOR NEURAL NETWORK
Y_train = train["label"] # 라벨 컬럼을 Y_TRAIN으로 받음 
X_train = train.drop(labels = ["label"],axis = 1)
# print('엑스',X_train)
X_train = X_train / 255.0 # 왜 255로 나누는 것인가? 값이 0에서 부터 255까지 표시된 값을 정규화 해주는 것이지 
X_test = test / 255.0 # 



 # 여기서 shape 의 -1 의 수는 이 모델 function 으로 패스된 x 의 총 이미지 갯수 입니다. 
    # -1 대신 모델로 패스된 이미지의 수를 직접 넣어도 되지만 그럴 경우에는 
    # 매번 다른 수의 트레이닝 이미지를 사용할때 마다 이 수를 직접 변경해야 합니다.
    # -1 를 사용함으로써 일일히 직접 변경하지 않고 알아서 알맞은 수를 찾아 변경해 줍니다.

    # shape 의 맨 마지막 1 의 수는 이미지가 갖는 channel 의 수로 보통의 이미지의 경우 3가지의 channel 이 있습니다. (Red, Green, Blue)
    # MNIST 의 경우에는 channel 이 하나 이므로 1 로 저장합니다.




X_train = X_train.values.reshape(-1,28,28,1) # 784개의 행을 28*28 행렬로 바꾸는 작업 / 형태도 넘파이로 바뀜
print('엑스',X_train)
print(type(X_train))
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10) # 2진법으로 바꾼는 작업

# tf.keras.utils.to_categorical( y, num_classes=None ) 클래스 벡터 (정수)를 이진 클래스 행렬로 변환합니다 y: 행렬로 변환 될 클래스 벡터 (0에서 num_classes의 정수). num_classes: 총 클래스 수입니다.


import matplotlib.pyplot as plt
# PREVIEW IMAGES
plt.figure(figsize=(15,4.5)) # plt.figure 활용 1. / kaggle/basic/figurebals
for i in range(30):  
    plt.subplot(3, 10, i+1) # 3행 10열 30개 칸중 순서대로 
    plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary) # cmap : str 또는 Colormap선택 사항    Display an image, i.e. data on a 2D regular raster.
# Colormap 인스턴스 또는 등록 된 색상 맵 이름입니다. 색상 맵은 스칼라 데이터를 색상에 매핑합니다. RGB (A) 데이터에서는 무시됩니다. 기본값은 rcParams["image.cmap"]입니다
#화상(image) 데이터처럼 행과 열을 가진 행렬 형태의 2차원 데이터는 imshow 명령을 써서 2차원 자료의 크기를 색깔로 표시하는 것이다.
    plt.axis('off')  # 각 칸의 경계선 지움
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()

    

# keras에서는 이미지데이터 학습을 쉽게하도록 하기위해 다양한 패키지를 제공한다. 그 중 하나가 ImageDataGenerator 클래스이다.
# ImageDataGenerator 클래스를 통해 객체를 생성할 때 파라미터를 전달해주는 것을 통해 데이터의 전처리를 쉽게할 수 있고, 
#또 이 객체의 flow_from_directory 메소드를 활용하면 폴더 형태로된 데이터 구조를 바로 가져와서 사용할 수 있다. 이 과정은 매우 직관적이고 코드도 ImageDataGenerator를 사용하지 않는 방법에 비해 상당히 짧아진다. 


datagen = ImageDataGenerator(
        rotation_range=10,  # 무작위 회전 정도의 범위 / 10도 회전
        zoom_range = 0.10,  # 플로트 또는 [lower, upper]. 무작위 확대 / 축소 범위. , 부동 소수점의 경우
        width_shift_range=0.1,  #: Float, 1 차원 배열 형 또는 int - float : 1보다 작은 경우 총 너비의 부분 또는 1보다 큰 픽셀 - 1 차원 배열과 유사 : 배열의 임의 요소입니다
        height_shift_range=0.1) # Float, 1 차원 배열 형 또는 int - float : 1보다 작은 경우 전체 높이의 일부 또는 1보다 큰 픽셀 - 1 차원 배열과 유사 : 배열의 임의 요소입니다. - int : 간격의 정수 픽셀 수 
          

#generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.05, height_shift_range=0.05)
#로 이미지 생성기를 만들었는데 이름을 보시면 아시겠지만 이미지를 최대 20도 회전시킬 수 있고 좌우, 상하 이동은 최대 5% 비율로 하겠다는 말입니다.

# keras에서는 이미지데이터 학습을 쉽게하도록 하기위해 다양한 패키지를 제공한다. 그 중 하나가 ImageDataGenerator 클래스이다.
# ImageDataGenerator 클래스를 통해 객체를 생성할 때 파라미터를 전달해주는 것을 통해 데이터의 전처리를 쉽게할 수 있고, 
# 또 이 객체의 flow_from_directory 메소드를 활용하면 폴더 형태로된 데이터 구조를 바로 가져와서 사용할 수 있다.
#  이 과정은 매우 직관적이고 코드도 ImageDataGenerator를 사용하지 않는 방법에 비해 상당히 짧아진다. 환경은 keras tensorflow backend를 이용하였다.






X_train3 = X_train[9,].reshape((1,28,28,1)) #4d텐서 
# print(X_train3)
Y_train3 = Y_train[9,].reshape((1,10))
plt.figure(figsize=(15,4.5))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()
    plt.imshow(X_train2[0].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis('off')
    if i==9: X_train3 = X_train[11,].reshape((1,28,28,1)) # 10번쨰에 x의 11--(값9)번쨰 값을을 받겠다.
    if i==19: X_train3 = X_train[18,].reshape((1,28,28,1)) # 20번쨰 18번쨰 값 7을 받음
# plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()

# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
# 
#   input : [batch, in_height, in_width, in_channels] 형식. 28x28x1 형식의 손글씨 이미지.
#   filter : [filter_height, filter_width, in_channels, out_channels] 형식. 3, 3, 1, 32의 w.
#   strides : 크기 4인 1차원 리스트. [0], [3]은 반드시 1. 일반적으로 [1], [2]는 같은 값 사용.
#   padding : 'SAME' 또는 'VALID'. 패딩을 추가하는 공식의 차이. SAME은 출력 크기를 입력과 같게 유지.
# 
# 3x3x1 필터를 32개 만드는 것을 코드로 표현하면 [3, 3, 1, 32]가 된다. 순서대로 너비(3), 높이(3), 입력 채널(1), 출력 채널(32)을 뜻한다. 32개의 출력이 만들어진다.
# 출처: https://pythonkim.tistory.com/56 [파이쿵]

# 
# 필터로 특징을 뽑아주는 컨볼루션(Convolution) 레이어
# 케라스에서 제공되는 컨볼루션 레이어 종류에도 여러가지가 있으나 영상 처리에 주로 사용되는 Conv2D 레이어를 살펴보겠습니다. 레이어는 영상 인식에 주로 사용되며, 필터가 탑재되어 있습니다. 아래는 Conv2D 클래스 사용 예제입니다.

# Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')

# 첫번째 인자 : 컨볼루션 필터의 수 입니다. 
# 두번째 인자 : 컨볼루션 커널의 (행, 열) 입니다.

# 필터는 가중치를 의미합니다. 하나의 필터가 입력 이미지를 순회하면서 적용된 결과값을 모으면 출력 이미지가 생성됩니다. 여기에는 두 가지 특성이 있습니다.
# 하나의 필터로 입력 이미지를 순회하기 때문에 순회할 때 적용되는 가중치는 모두 동일합니다. 이를 파라미터 공유라고 부릅니다. 이는 학습해야할 가중치 수를 현저하게 줄여줍니다.
# 출력에 영향을 미치는 영역이 지역적으로 제한되어 있습니다. 즉 그림에서 y~0~에 영향을 미치는 입력은 x~0~, x~1~, x~3~, x~4~으로 한정되어 있습니다. 
# 이는 지역적인 특징을 잘 뽑아내게 되어 영상 인식에 적합합니다. 예를 들어 코를 볼 때는 코 주변만 보고, 눈을 볼 때는 눈 주변만 보면서 학습 및 인식하는 것입니다.

# 이 주위에 큰
nets = 15
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1))) # 필터 32개 / 커널사이즈 3*3픽셀/ 
    model[j].add(BatchNormalization())
    # https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/ 
    # 배치 노멀레이션 설명 
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4)) # 지나치게 많은 정보를 제거하기 위해 40% 제거 - 정확도 향상시킴

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten()) # 28*28을 다시 한줄로 표현 
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax')) # softmax - 로지스틱스 회기분석과 비슷한 성질 / 강화학습을 일으키는 요소  

    # COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Adam method의 강점 
#여기서 소개하는 Adam method는 Adagrad + RMSProp의 장점을 섞어 놓은 것으로 자세한 알고리즘은 추후에 다루기로 하겠다. 
# 저자가 말하는 Adam method의 의 주요 장점은 stepsize가 gradient의 rescaling에 영향 받지 않는다는 것이다. 
# gradient가 커져도 stepsize는 bound되어 있어서 어떠한 objective function을 사용한다 하더라도 안정적으로 최적화를 위한 하강이 가능하다. 게다가 stepsize를 과거의 gradient 크기를 참고하여 adapted시킬 수 있다.







annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
history = [0] * nets
epochs = 45
for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
    

results = np.zeros( (X_test.shape[0],10) ) 
for j in range(nets):
    results = results + model[j].predict(X_test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("MNIST-CNN-ENSEMBLE.csv",index=False)


plt.figure(figsize=(15,6))
for i in range(40):  
    plt.subplot(4, 10, i+1)
    plt.imshow(X_test[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("predict=%d" % results[i],y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.show()