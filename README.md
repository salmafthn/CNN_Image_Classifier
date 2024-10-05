Implementing CNN with LeNet5 architecture for MNIST image dataset

![image](https://github.com/user-attachments/assets/3736e4fe-83d1-48f6-96f2-9e3b365922c5)

Model CNN ini memiliki tiga lapisan convolusi (**conv layers**) dengan feature map dan ukuran kernel yang bervariasi. Lapisan **Conv1** memiliki 6 feature maps dengan kernel size 5x5 dan stride default 1, diikuti oleh fungsi aktivasi **Tanh** dan max pooling dengan ukuran kernel 2x2 dan stride 2. Lapisan **Conv2** memiliki 16 feature maps, kernel size 5x5, dan juga menggunakan **Tanh** serta max pooling 2x2. Lapisan **Conv3** menambahkan 120 feature maps dengan kernel size 4x4, diikuti oleh **Tanh** tanpa pooling. Setelah itu, fitur di-flatten untuk lapisan fully connected (**fc1**) dengan 120 ke 84 neuron, menggunakan **Tanh** sebagai aktivasi, diikuti oleh lapisan **fc2** yang mengeluarkan 10 kelas, dengan fungsi aktivasi **Softmax** pada output akhir.
