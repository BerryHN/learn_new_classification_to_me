# -*- coding:UTF-8 -*-
import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")

import jieba
import os

dir = {'baby': 129,'car': 410,'food': 409,'health': 406,'legend': 396,'life': 409,'love': 158,'news': 409,'science': 409,'sexual': 38}
# 设置词典，分别是类别名称和该类别下一共包含的文本数量
data_file_number = 0
# 当前处理文件索引数

a=[ line.strip() for line in open("../data/stopword.txt",encoding= 'UTF-8') ]


a=[line.strip()  for line in open("../data/stopword.txt",encoding= 'UTF-8')]


for world_data_name,world_data_number in dir.items():
	# 将词典中的数据分别复制到world_data_name,world_data_number中
	while (data_file_number < world_data_number):
		# 打印文件索引信息
		file = open('../data/raw_data/'+world_data_name+'/'+str(data_file_number)+'.txt','r',encoding= 'UTF-8')
		file_name='../data/train_data/'+world_data_name+'/'+str(data_file_number)+'.txt'
		file_dir='../data/train_data/' + world_data_name

		if not os.path.isdir(file_dir):
			os.makedirs(file_dir)
		if not os.path.exists(file_name):
			os.system(r'touch %s'%file_name)


		file_w = open('../data/train_data/'+world_data_name+'/'+str(data_file_number)+'.txt','w',encoding= 'UTF-8')
		for line in file:
			stoplist = {}.fromkeys([ line.strip() for line in open("../data/stopword.txt",encoding= 'UTF-8') ])
			# 读取停用词在列表中
			seg_list = jieba.lcut(line,cut_all=False)
			# jieba分词精确模式
			seg_list = [word for word in list(seg_list) if word not in stoplist]  
			# 去除停用词
			for i in range(len(seg_list)):
				file_w.write(str(seg_list[i])+'\n')
			# 分完词分行输入到文本中
			# file_w.write(str(seg_list))
			# print(line, end='')
		file_w.close()
		file.close()
		data_file_number = data_file_number + 1
	data_file_number = 0
		
	


