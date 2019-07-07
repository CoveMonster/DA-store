from xml.dom import minidom
import codecs
import fnmatch
import os
from unit import Connective, ArgsUnit, Sense, DiscourseUnit
import re
from zhon.hanzi import punctuation
import discourseConList as dis_con

class DataReader:
    # 标签属性值
    def get_attr_value(self, node, attrname):
        return node.getAttribute(attrname) if node else ''

    # 节点值
    def get_node_value(self, node, filepath, index=0):
        try:
            return node.childNodes[index].nodeValue if node else ''
        except IndexError as e:
            print(e)
            print(filepath)
            #return 0

    # 返回节点
    def get_xml_node(self, node, name):
        return node.getElementsByTagName(name) if node else []

    def get_xml_data(self, fileName = 'cbs_0003#.xml'):
        doc = minidom.parse(fileName)
        root = doc.documentElement
        sense_node = self.get_xml_node(root, 'Sense')
        senselist = []
        for node in sense_node:
            # 获取属性 sense
            sense_type = self.get_attr_value(node, 'type')
            sense_rel_no = self.get_attr_value(node, 'RelNO')
            sense_content = self.get_attr_value(node, 'content')
            # sense structure
            sense_unit = Sense()
            #sense_unit = Sense(sense_type, sense_rel_no, sense_content)
            sense_unit.init(sense_type, sense_rel_no, sense_content)

            #print(sense_type, sense_rel_no, sense_content)
            # 获取source
            node_source = self.get_xml_node(node, 'Source')
            source_con = self.get_node_value(node_source[0], fileName)

            #print(source_con)
            # 获取连词
            node_connect = self.get_xml_node(node, 'Connectives')
            # con unit
            #con_unit
            con_unit = Connective()
            for node_con in node_connect:
                node_con_span = self.get_xml_node(node_con, 'Span')
                connective_span = self.get_node_value(node_con_span[0], fileName)
                node_con_content = self.get_xml_node(node_con, 'Content')
                connective_contend = self.get_node_value(node_con_content[0], fileName)
                #print(connective_span, connective_contend)
                con_unit.init(connective_span, connective_contend)


            # 获取args1
            node_agr1 = self.get_xml_node(node, 'Arg1')
            arg1_unit = ArgsUnit()
            for node_agr in node_agr1:
                node_span = self.get_xml_node(node_agr, 'Span')
                arg1_span = self.get_node_value(node_span[0], fileName)
                node_content = self.get_xml_node(node_agr, 'Content')
                arg1_contend = self.get_node_value(node_content[0], fileName)
                #print(arg1_span, arg1_contend)
                arg1_unit.init(arg1_span, arg1_contend)

            #获取args2
            node_agr2 = self.get_xml_node(node, 'Arg2')
            arg2_unit = ArgsUnit()
            for node_agr in node_agr2:
                node_span = self.get_xml_node(node_agr, 'Span')
                arg2_span = self.get_node_value(node_span[0], fileName)
                node_content = self.get_xml_node(node_agr, 'Content')
                arg2_contend = self.get_node_value(node_content[0], fileName)
                #print(arg2_span, arg2_contend)
                arg2_unit.init(arg2_span, arg2_contend)

            # 获取 注解
            node_anno = self.get_xml_node(node, 'Annotation')
            annotation = self.get_node_value(node_anno[0], fileName)
            #print(annotation)
            discourse_unit = DiscourseUnit()
            discourse_unit.init(fileName, sense_unit, source_con, con_unit, arg1_unit, arg2_unit, annotation)
            senselist.append(discourse_unit)
        return senselist


# 改变原有文件的编码格式
def changeFormat(fileName='cbs_0003#.p1'):
    xml_file_text = codecs.open(fileName, 'r')  #b, 'gb2312'
    text = xml_file_text.read().encode('utf-8')
    newtext = str(text, encoding="utf-8").replace('gb2312', 'utf-8')
    xml_file_text.close()
    return newtext

def write_text(text, fileName):
    try:
        f = open(fileName+'.xml', 'w', encoding='utf-8')
        f.write(text)
        f.close()
    except IOError as e:
        print(e)
        print('异常文件：',fileName)
    except UnicodeDecodeError as e:
        print(e)
        print('异常文件：',fileName)
    #print (text.decode('utf-8').encode('gb2312'))


def is_file_match(filename, patterns):
        for pattern in patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False


def find_specific_files(root, patterns=['*.p*'], exclude_dirs=[]):
        for root, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if is_file_match(filename, patterns):
                    yield os.path.join(root, filename)

            for d in exclude_dirs:
                if d in dirnames:
                    dirnames.remove(d)


def change_all_file_format():
    # d = input('输入文件目录:')
    d = os.path.abspath('xml')
    timelist = []
    newfilelist = []
    for item in find_specific_files(d, patterns=['*.p[0-9]']):
        timelist.append(item)
        # print(item)
    # print(len(timelist))
    for filepath in timelist:
        filepath = filepath.replace('xml', 'Source')
        newfilelist.append(filepath)
    for i in range(len(timelist)):
        text = changeFormat(timelist[i])
        write_text(text, newfilelist[i])


def check():
    d = os.path.abspath('xml')
    new_d = os.path.abspath('source')
    newfilelist = []
    timelist = []
    another = []
    for item in find_specific_files(d, patterns=['*.p[0-9]']):
        timelist.append(item)
    for item in find_specific_files(d, patterns=['*.txt']):
        another.append(item)
    for item in find_specific_files(new_d, patterns=['*.xml']):
        newfilelist.append(item)
    print(len(timelist), len(another), len(newfilelist))

    # for i in timelist:
    #     print(i)
    # print('--------------------------------'*10)
    # for i in another:
    #     print(i)
    # print('--------------------------------'*10)
    # for i in newfilelist:
    #     print(i)


# 数据读取到内存中
def reader_data():
    #print(1)
    root = os.path.abspath('source')
    filepath_list = []
    for item in find_specific_files(root, patterns=['*.xml']):
        filepath_list.append(item)
    print(len(filepath_list))
    datareader = DataReader()
    data_list = []
    for file in filepath_list:
        now_list = datareader.get_xml_data(file)
        [data_list.append(data) for data in now_list]
        #print(file)
    #print(len(data_list))
    return data_list

def test_bad_sample_reader():
    reader = DataReader()
    # text = changeFormat('cbs_0003#.p1')
    # write_text(text, 'cbs_0003#.p1')
    filepath = 'C:/Users/chenyi/Documents/GitHub/ChapterAnalysis/hitcdtb/data/source/bn/cnr/00/cnr_0089#.p2.xml'
    reader.get_xml_data(filepath)

# 预处理成词向量训练的格式
def pre_text(data_list):
    #print(data_list)
    sentences = ''
    for data in data_list:
        sen = data.args1.get_contend()
        # print(type(sen))
        # print('char  == ', sen) if type(sen) == 'NoneType' else ''

        #sen = re.sub(r"[%s]+" % punctuation, "", sen)
        try:
            sen = re.sub(r"[%s]+" % punctuation, "", sen)
            sentences += sen
        except TypeError as e:
            print(e)
        # sen = re.sub(u"[%s]+" %punctuation, "", sen.decode("utf-8"))
        sen = data.args2.get_contend()
        try:
            sen = re.sub(r"[%s]+" % punctuation, "", sen)
            sentences += sen
        except TypeError as e:
            print(e)
        # sen = re.sub(r"[%s]+" % punctuation, "", sen)
        # sen = re.sub(u"[%s]+" % punctuation, "", sen.decode("utf-8"))

    filepath = 'corpus.txt'
    f = open(filepath, mode='w', encoding='utf-8')
    f.write(sentences)
    f.close()

def display_unit(discourse_unit_list):
    text = ''
    for unit in discourse_unit_list:
        str = ''
        str += '*'*20 + '\n'
        str += '文件路径：' + unit.filepath + '\n'
        str += '显隐式：' + unit.sense.type + ' | ' + '类型编号：' + unit.sense.relNo + ' | ' + '类型名：' + unit.sense.contend + '\n'
        str += 'source：' + unit.source + '\n'
        str += '连词 span：' + unit.connective.span + ' | ' + 'contend ：' + unit.connective.contend + '\n'

        # agr1
        str += 'arg1：' + '\n\t span :'
        try:
            str += unit.args1.span
        except TypeError as e:
            str += 'NUll'
        str += '\n\t contend ：'
        try:
            str += unit.args1.contend + '\n'
        except TypeError as e:
            str += 'NUll' + '\n'
        # agr2
        str += 'arg2：' + '\n\t span :'
        try:
            str += unit.args2.span
        except TypeError as e:
            str += 'NUll'
        str += '\n\t contend ：'
        try:
            str += unit.args2.contend + '\n'
        except TypeError as e:
            str += 'NUll' + '\n'
        #str += 'arg2：' + '\n\t span :' + unit.args2.span + ' | ' + '\n\t contend ：' + unit.args2.contend + '\n'
        str += 'annotion ：' + unit.annotation + '\n'
        str += '*' * 20 + '\n'
        text += str
    #print(str)
    file = open('reader_result.txt', 'w', encoding='utf-8')
    file.write(text)
    file.close()


def display_con(con_list):
    text = ''
    for con in con_list:
        str += '*'*20
        str += 'connective : ' + con.name
        #for

if __name__ == '__main__':
    # 篇章读取 连词 生成连词列表
    discourse_unit_list = reader_data()
    #写入 reader_result.txt
    #display_unit(discourse_unit_list)

    print(len(discourse_unit_list))
    conn_list = dis_con.count_conn(discourse_unit_list)
    print(len(conn_list))



    # 检查
    #check()

    # 数据读取 进行预处理  生成词向量训练的格式
    #pre_text(reader_data())

    # change_all_file_format()

    # result = sorted(timelist, key=lambda list: list[1], reverse=True)[:5]
    # for a in result:
    #     print(a[0], time.ctime(a[1]))
