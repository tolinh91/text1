Problems
Làm bài toán nhận diện chữ in hoa viết tay từ A-Z

Nguồn dữ liệu

Link 1: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format?resource=download

Link 2: https://www.nist.gov/srd/nist-special-database-19


Dàn bài:

1. Import các thư viện cần thiết

2. Đọc file data A.csv

Tạo list `rows`.Mỗi entry của list là một dòng lưu dữ liệu của 1 ảnh --->  1 dòng lưu dữ liệu 1 ảnh có phải là 1 tensor vector?

3. Tạo ra tập data set cho training

- Tạo training data ---> Xác định số lượng ảnh dùng cho training
- Tạo label cho training dât ---> Bao nhiêu label? 
- In ra các thông số trên
- Set up các thông số sau

* BATCH_SIZE = 32
* IMG_SIZE = 28
* N_CLASSES = 4
* LR = 0.001
* N_EPOCHS = 50

4. Tạo neural network
- Neural network gồm 6 convolutional layer + 2 fully-connected layer ---> Giải thích algorithm trong từng layer
- Fit model vào trianing data
- Model: tflearn.DNN()

5. Dự đoán với tập test data

6. Extra: thử lại với model khác




Question:
* Model nào?
* Kết quả?
* Cách training data
* Tool sử dụng
* sử dụng layer như nào? Bao nhiêu layer?
* Add API nếu có



Answer:

1. Model: tflearn.DNN()
2. Kết quả: 0.9964297306069458
3. Cách training Data: 
4. Tool sử dụng:
5. Cách sử dụng layer
Model gồm 6 lớp Convolutional layer và 2 lớp Fully Connected Layer nối tiếp nhau.

5.1. Convolutional Layer số 1

Convolutional layer số 1 gồm 32 filter (kernel), fileter size là 3x3, bước nhảy stride mặc định = 1.
``` py
network = conv_2d(network, 64, 3, activation='relu')
```



5.2. Maxpool layer
Maxpool layer có kernel size bằng 2 (lấy max trong vòng 2 ô).


``` py
network = max_pool_2d(network, 2)
```


5.3. Fully-connected layer

Số lượng neuron: 1024

Activation function: ReLu

```py
network = fully_connected(network, 1024, activation='relu') #4
```

5.4. Dropout 80%
```py
network = dropout(network, 0.8) #5
```

5.5. Fully-connected layer đại diện cho output

N_CLASSES: số output đầu ra
Activation function: softmax --> Tổng xác suất đầu ra bằng 1 -> Normalized

```py
network = fully_connected(network, N_CLASSES, activation='softmax')#6
```

6. Các API được sử dụng