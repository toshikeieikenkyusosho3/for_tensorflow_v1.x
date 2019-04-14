# 値を予測するニューラルネット用のテンプレート
# 変更が必要なのは、下の入力する部分と書かれているところだけ。
%matplotlib inline
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# 入力する部分
num_hiddens = []    # 各隠れ層の次元数 例）num_hiddens = [64, 64] 
max_epochs =        # 訓練のステップの数 例） max_epochs = 2000
print_interval =    # 途中経過の印刷のインターバル 例）print_interval = 100
filename_data =     # 訓練用データのファイル名 例） filename_data = 'salary_model.csv' 
# 入力データのうち、訓練に使うデータの割合。
# 例）ratio_for_training = 0.8であれば、80%が訓練に、20％が確認に使われる。
ratio_for_training = 
# 説明変数の最初の列の列番号 例）1列目であれば explanatory_start_column = 1
explanatory_start_column = 
# 説明変数の最後の列の列番号 例）explanatory_end_column = 24列目であれば 24
explanatory_end_column =
# 目的変数（教師データ）の列番号 例） 25列目であれば outcome_column =25
outcome_column = 
# 入力はここまで

explanatory_variables = np.arange(explanatory_start_column-1, 
                                  explanatory_end_column)    
outcome_variables = [outcome_column-1]        

tf.set_random_seed(123)

def inference(x, num_in, num_hiddens, num_out):
    # 重み変数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    # バイアス変数
    def bias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    for i, num_hidden in enumerate(num_hiddens):
        if i == 0:     # 入力層から隠れ層
            input = x
            input_dim = num_in
        else:        # 隠れ層から隠れ層
            input = output
            input_dim = num_hiddens[i-1]

        W = weight_variable([input_dim, num_hidden])
        b = bias_variable([num_hidden])

        h = tf.nn.relu(tf.matmul(input, W) + b)
        output = h

    # 隠れ層から出力層
    W_out = weight_variable([num_hiddens[-1], num_out])
    b_out = bias_variable([num_out])
    y = tf.matmul(h, W_out) + b_out
    return y


def loss(y, t):
    # 値の予測の場合は以下の平均2乗誤差をロス関数を使う。
    mse = tf.reduce_mean(tf.square(y-t))
    loss = mse
    return loss

def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                       beta1=0.9,
                                       beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step

def mae(y, t):
    mae = tf.reduce_mean(tf.abs(y-t))   
    return mae

if __name__ == '__main__':

    # データの生成
    # read csv file as panda dataframe, then convert to list ane then to array
    df = pd.read_csv(filename_data, skiprows=[0], header=None)
    df1 = np.array(df.values.tolist())
    
    seed = 1
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    train_indices = np.random.choice(len(df1), 
                                     round(len(df)*ratio_for_training), 
                                     replace = False)
    test_indices = np.array(list(set(range(len(df1)))-set(train_indices)))
    
    train_set = df1[train_indices]
    test_set = df1[test_indices]
    
    # 列でXとYを分ける。列の番号を指定する必要がある。
    X_train = train_set[:, explanatory_variables]
    Y_train = train_set[:, outcome_variables]
    X_test = test_set[:, explanatory_variables]
    Y_test = test_set[:, outcome_variables]
    
    # 入力データを正規化
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train = (X_train - X_min)/(X_max - X_min)
    X_test = (X_test - X_min)/(X_max - X_min)

    # モデル設定
    N_train = len(X_train[0])
    num_in = len(X_train[0])
    num_out = len(Y_train[0])

    x = tf.placeholder(tf.float32, shape=[None, num_in])
    t = tf.placeholder(tf.float32, shape=[None, num_out])

    y = inference(x, num_in=num_in, num_hiddens=num_hiddens, num_out=num_out)
    loss = loss(y, t)
    train_step = training(loss)

    mae = mae(y, t)

    history = {
        'val_loss': [],
        'val_mae': [],
        'val_loss_test': [],
        'val_mae_test': []
    }

    # モデル学習   

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(max_epochs):
        sess.run(train_step, feed_dict={x: X_train, t: Y_train})    

        # データを用いた評価
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_train,
            t: Y_train,
        })
        val_mae = mae.eval(session=sess, feed_dict={
            x: X_train,
            t: Y_train,
        })
        val_loss_test = loss.eval(session=sess, feed_dict={
            x: X_test,
            t: Y_test,
        })
        val_mae_test = mae.eval(session=sess, feed_dict={
            x: X_test,
            t: Y_test,
        })
        
        # データに対する学習の進み具合を記録
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_loss_test'].append(val_loss_test)
        history['val_mae_test'].append(val_mae_test)
        
        if (epoch+1) % print_interval == 0:
            print('epoch+1:', epoch+1, ' train loss:', val_loss, ' train mae:', val_mae,
                 ' test loss:', val_loss_test, ' test mae:', val_mae_test)

    # 学習の進み具合を可視化
    # 計算ループごとのlossの動きをグラフ化。kは黒、rは赤、-は実線、-- は破線。
    plt.plot(history['val_loss'], 'k-', label='Train Loss')
    plt.plot(history['val_loss_test'], 'r--', label='Test Loss')
    plt.title('Loss per epoch')
    plt.legend(loc='upper right')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.show()
    
    # 計算ループごとの精度の動きをグラフ化
    plt.plot(history['val_mae'], 'k-', label='Train MAE')
    plt.plot(history['val_mae_test'], 'r--', label='Test MAE')
    plt.title('Mean abosolute error per epoch')
    plt.legend(loc='upper right')
    plt.xlabel('epochs')
    plt.ylabel('MAE')
    plt.show()
