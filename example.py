#!/usr/bin/python
#coding:utf-8

from segmenter import NaiveSegmenter

seg=NaiveSegmenter()
with open('./data/pku_test.utf8','r') as pku_file:
	pku_content=pku_file.read()

with open('./data/pku_test.utf8','r') as msr_file:
	msr_content=msr_file.read()

print seg.cut('8月8日，甘州区法院处置一起哄闹法院，辱骂、撕扯和殴打法官事件，闹事女子张某某被控制后，被公安机关民警带走')

