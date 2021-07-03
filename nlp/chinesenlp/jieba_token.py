import jieba
import jieba.posseg as psg
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

content = "现如今，机器学习和深度学习带动人工智能飞速的发展，并在图片处理、语音识别领域取得巨大成功。"

#精确分词
segs_1 = jieba.cut(content, cut_all=False)
logging.info("/".join(segs_1))

#全模式
segs_2 = jieba.cut(content, cut_all=True)
logging.info("/".join(segs_2))

#搜索引擎模式
seg_3 = jieba.cut_for_search(content)
logging.info("/".join(seg_3))

#用lcut生成cut
seg_4 = jieba.lcut(content, cut_all=False)
logging.info(seg_4)
seg_5 = jieba.lcut_for_search(content)
logging.info(seg_5)

#获取词性
logging.info([(x.word, x.flag) for x in psg.lcut(content)])


