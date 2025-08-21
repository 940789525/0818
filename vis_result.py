# from torch.utils.data import DataLoader
# from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
# from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
# # 初始化分词器
# tokenizer = ClipTokenizer()
#
# # 创建数据加载器实例
# msrvtt_testset = MSRVTT_DataLoader(
#     csv_path="E:/pyproject/CCVTR/data/MSR-VTT/msrvtt_data/MSRVTT_JSFUSION_test.csv",
#     features_path="E:/pyproject/CCVTR/data/MSR-VTT/MSRVTT/videos/all",
#     max_words=32,
#     feature_framerate=1,
#     tokenizer=tokenizer,
#     max_frames=12,
#     frame_order=0,
#     slice_framepos=0,
# )
#
# # 创建DataLoader
# train_dataloader = DataLoader(msrvtt_testset, batch_size=32, shuffle=True)
#
# # 遍历数据加载器
# for batch in train_dataloader:
#     pairs_text, pairs_mask, pairs_segment, video, res, mv, video_mask = batch
#     # 进行模型训练或评估
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取CSV数据
data = pd.read_csv("smallarray.csv")  # 替换为实际的文件路径
custom_y_labels = ['text8349','text9542','text9770','text9201','text9771',
                   'text9202','text9679','text9589','text9203','text7020',
                   'text9837','text8690','text9773','text9836','text8978',
                   'text9835','text9208','text7026','text9209','text8851']
# 2. 使用Seaborn绘制热力图
plt.figure(figsize=(8, 6))  # 调整热力图的大小
sns.heatmap(data, annot=False, cmap="YlGnBu", linewidths=0.5, yticklabels=custom_y_labels)

# 3. 显示图形
plt.title("Heatmap of CSV Data")  # 图形标题

# 4. 保存图形为 PNG 图片
plt.savefig("heatmap.png", dpi=600)  # 保存图形为 PNG，设置分辨率为300

plt.show()