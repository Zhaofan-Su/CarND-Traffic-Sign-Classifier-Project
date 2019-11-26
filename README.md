## 图像分类: 道路交通标志识别

---

概述
---

交通标志识别能有效地帮助司机们了解道路的关键信息、获取当前的道路状况，从而有效地避免交通事故的发生。同时，交通标志识别技术对于自动驾驶来说也有着非常重要的作用。

使用传统的计算机视觉的方法来对交通标志进行识别并不是不可行的事情，但是需要耗费大量的事件来手工处理图像中的一些重要的特征。如果引入深度学习技术来识别交通标志，则可以大大地减少人工工作量，选择合适的网络结构以及所需的训练参数则可以将识别的准确率大幅度提高。

该项目构建了可以调参的LeNet网络，在德国交通标志数据集[GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)上训练进行分类识别。

项目结构
---


项目运行方式
---

请使用下列命令运行项目：
```sh
git clone https://github.com/Zhaofan-Su/CarND-Traffic-Sign-Classifier-Project.git
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
使用这种方式，你将会看到整个项目的数据预处理、网络搭建、数据训练以及模型测试的过程，你也可以根据需要修改网络参数，训练自己的模型。


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.


