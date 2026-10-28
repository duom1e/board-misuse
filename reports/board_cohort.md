# 案板模型统一分组诊断

30张已消费图，7视频来源，18正例12负例；同输入评5模型。按模型列训练来源暴露，不能混作独立测试准确率。

|模型|来源|训练暴露|匹配/正例|负例误帧/负例|额外框|
|---|---|---|---|---|---|
|default|BV15J411D718|False|3/3|0/0|0|
|default|BV15gZzYXE4N|False|1/2|0/0|0|
|default|BV1aRmLBmE7i|False|2/2|0/0|2|
|default|BV1dC7H6PEoG|False|0/0|2/6|2|
|default|BV1dHckzwEBA|False|0/8|1/2|3|
|default|BV1eB4y1J7B6|False|0/0|0/4|0|
|default|BV1hK411u73L|False|3/3|0/0|0|
|ordinary|BV15J411D718|False|1/3|0/0|1|
|ordinary|BV15gZzYXE4N|False|2/2|0/0|0|
|ordinary|BV1aRmLBmE7i|False|2/2|0/0|0|
|ordinary|BV1dC7H6PEoG|False|0/0|0/6|0|
|ordinary|BV1dHckzwEBA|True|5/8|0/2|0|
|ordinary|BV1eB4y1J7B6|False|0/0|3/4|3|
|ordinary|BV1hK411u73L|False|2/3|0/0|1|
|conservative|BV15J411D718|False|3/3|0/0|0|
|conservative|BV15gZzYXE4N|False|1/2|0/0|0|
|conservative|BV1aRmLBmE7i|False|2/2|0/0|1|
|conservative|BV1dC7H6PEoG|False|0/0|2/6|2|
|conservative|BV1dHckzwEBA|True|0/8|1/2|1|
|conservative|BV1eB4y1J7B6|False|0/0|2/4|2|
|conservative|BV1hK411u73L|False|2/3|0/0|1|
|multisource|BV15J411D718|False|3/3|0/0|2|
|multisource|BV15gZzYXE4N|True|2/2|0/0|0|
|multisource|BV1aRmLBmE7i|True|2/2|0/0|3|
|multisource|BV1dC7H6PEoG|False|0/0|2/6|2|
|multisource|BV1dHckzwEBA|True|5/8|2/2|5|
|multisource|BV1eB4y1J7B6|True|0/0|1/4|1|
|multisource|BV1hK411u73L|False|3/3|0/0|0|
|negative_augmented|BV15J411D718|False|2/3|0/0|3|
|negative_augmented|BV15gZzYXE4N|True|2/2|0/0|0|
|negative_augmented|BV1aRmLBmE7i|True|2/2|0/0|0|
|negative_augmented|BV1dC7H6PEoG|True|0/0|0/6|0|
|negative_augmented|BV1dHckzwEBA|True|6/8|1/2|3|
|negative_augmented|BV1eB4y1J7B6|True|0/0|2/4|2|
|negative_augmented|BV1hK411u73L|False|2/3|0/0|0|

阈值conf .25 / IoU .5，一GT最多匹配一框。额外框含低重叠框与重复框，不得只报告命中而忽略它们。来源暴露仅针对本项目新增视频，原默认预训练重叠未知。人工非专家标签，截断外接框宽松，类别/接触/错用未评。

下一批优先补未参与训练或调参的边缘截断板、手遮挡圆板、木纹桌面圆板，以及锅盘容器负例。每一类尽量覆盖不同拍摄来源；保留整来源，不随机拆相邻帧。现有30图固定开发回归，不再反复称新测试。
