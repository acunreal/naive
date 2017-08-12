#!/usr/bin/python
#coding:utf-8

from segmenter import NaiveSegmenter

seg=NaiveSegmenter()

print seg.cut('8月8日，甘州区法院处置一起哄闹法院，辱骂、撕扯和殴打法官事件，闹事女子张某某被控制后，被公安机关民警带走')

