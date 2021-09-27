### 1. TextSeg数据集处理

TextSeg数据集包括：

- image：原图
- annotation：注解，json文件提供了识别结果、词级和字符级边界框，mask为字符级别
- semantic_label：image中单词级别的mask
- semantic_label_v1：论文使用的较老的数据，用于论文复现和效果对比

需要处理成单词级别图片，运行extract_word.py

```bashrc
$ python extract_word.py \
--textseg_path XXX \
--textseg_extract_word_path XXX
```
产生TextSeg_Extract_word数据集：

- image：对TextSeg中image处理，根据单词四边形边界框构造最小外接矩形，从而取出文字区域并变换为标准矩形
- mask：对TextSeg中semantic_label处理，同上
- fg：根据image与mask取出image中前景文字
- json：记录图片名称和其对应文本

### 2. 合成StyleText数据

基于TextSeg_Extract_word数据集合成一组七张数据

```bashrc
$ python gen.py \
--textseg_extract_word_path '/home/yfx/datasets/TextSeg_Extract_Word' \
--bg_filepath '/data1/yfx/datasets/bg_no_text/labels.txt' \
--process_num 16 \
--data_capacity 256 \
--sample_num 100000 \
--min_h 20 \
--data_dir '/home/yfx/datasets/StyleText_RealData'
```
其中：
- textseg_extract_word_path：TextSeg_Extract_word数据集地址
- bg_filepath：背景图像地址文件，每一行记录一张背景图像的绝对地址
- process_num：使用进程数目，默认4
- data_capacity：多进程共同维护一个队列，队列容量，默认256
- sample_num：合成数据数量，默认10
- min_h：从TextSeg_Extract_word的image取的图像高度低于min_h，会被剔除，默认为20
- data_dir：合成数据保存地址


### 3. 代码结构
```
StyleText-RealDatagen
|-- extract_word.py    // 从TextSeg数据集提取word级别图像
|-- gen.py             // 产生StyleText所需数据，一组7张
|-- render_standard_text.py    // 生成背景为127，文字为黑色、标准字体的图像
|-- skeletonization.py         // 细化算法，生成骨架
|-- utils.py     
```

