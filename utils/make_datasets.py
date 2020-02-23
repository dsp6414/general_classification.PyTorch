# coding : utf-8

import os
import sys

images_path = ''
output_path = ''

dir_dict = {'0': 0, '90': 1, '180': 2, '270': 3}

file_writer = open(os.path.join(output_path, 'labels.txt'), 'wb')
dirs_list = os.listdir(images_path)
for dir_name in dirs_list:
	path_name = os.path.join(images_path, dir_name)
	img_list = os.listdir(path_name)
	for img_name in img_list:
		file_name = os.path.join(path_name, img_name)
		if not os.path.exists(file_name) or not os.path.isfile(file_name):
			continue
		line = '{}\t{}\n'.format(file_name, dir_dict[dir_name])
		file_writer.write(line.encode('utf-8'))

file_writer.close()