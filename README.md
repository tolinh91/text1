# Bài toán nhận diện chữ viết tay
Xây dựng hệ thống nhận diện và phân biệt được 26 chữ cái in hoa (uppercase) trong tiếng Anh: A, B, C, etc. Bài toán được áp dụng trong việc chấm điểm bài thi trắc nghiệm dạng trả lời câu hỏi bằng chữ viết.

DATA bộ dữ liệu hình ảnh của 26 chữ cái tiếng Anh được lưu trong file `A.csv`.
## General Information

### Installation
``` py

pip install tensorflow

pip install matplotlib.pyplot
```

### Tools
* Python3
* Git

### Data set

Bộ dữ liệu Special Database 19 chứa NIST Handprinted Forms and Characters Database chứa dữ liệu training cho bài toán nhận diện tài liệu viết tay và nhận diện kí tự. 

Bộ dữ liệu này gồm các mẫu chữ viết tay từ hơn 3600 người, 810 000 ảnh chụp kí tự tách rời, chứng nhận bản quyền, tài liệu tham khảo và phần mềm quản lý hình ảnh.

Các đặc trưng của database này bao gồm như sau.

* Final accumulation of NIST's handprinted sample data
* Full page HSF forms from 3600 writers
* Separate digit, upper and lower case, and free text fields
* Over 800,000 images with hand checked classifications

Data Source: [Here](https://www.nist.gov/srd/nist-special-database-19)  .

**For more information please contact:**
Standard Reference Data Program

National Institute of Standards and Technology

100 Bureau Dr., Stop 6410

Gaithersburg, MD 20899-6410

(844) 374-0183 (Toll Free) 
