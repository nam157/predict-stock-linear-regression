import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math as mt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Google.csv')
df.Date = df['Date'].astype('datetime64[ns]')
df.set_index('Date', inplace=True)
num=[]
for  i in range(1,len(df)+1):
    num.append(i)
x = np.asarray(num)

# tạo một biến gọi là forecast_out, để lưu trữ số ngày để dự đoán trong tương lai
forecast_out = 65

df['Prediction'] = df[['Close']].shift(-1)

# Tạo X dữ liệu độc lập
X = np.array(df.drop(['Prediction'], axis=1))

# xoá 65 của forecast ra khỏi X
X = X[:-forecast_out]
print(f'giá trị x: {X}')
# tạo biến phục thuộc y
y = np.array(df['Prediction'])

# Nhận tất cả các giá trị y ngoại trừ hàng 65 cuối cùng
y = y[:-forecast_out]
print(f'giá trị y :{y}')

split_percentage = 0.8
split = int(split_percentage*len(df))

x_train = X[:split]
y_train = y[:split]

x_test = X[split:]
y_test = y[split:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# dùng modun có sẵn
model = LinearRegression()
# Train the model
model.fit(x_train, y_train)
lr_confidence = model.score(x_test, y_test)
print("tỉ lệ dự đoán: ", lr_confidence)
# lay 65 hang cuoi cung trong Close ra rồi chuyển thành mảng array
x_forecast = np.array(df.drop(['Prediction'], 1))
a = x_forecast[-forecast_out:]
# dùng hồi quy để dự đoán
lr_prediction = model.predict(a)
print("dự đoán trong 65: ", lr_prediction)
list=[i for i in range(len(df['Close']),len(df['Close'])+forecast_out,1)]

plt.plot(x,df['Close'],label='Close')
plt.plot(list,lr_prediction,label='Prediction')
plt.legend(loc=4)
plt.show()
