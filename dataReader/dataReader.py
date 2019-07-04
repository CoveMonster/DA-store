from xml.dom import minidom
import codecs
import fnmatch
import os

class DataReader:
    # 标签属性值
    def get_attr_value(self, node, attrname):
        return node.getAttribute(attrname) if node else ''

    # 节点值
    def get_node_value(self, node, index=0):
        return node.childNodes[index].nodeValue if node else ''

    # 返回节点
    def get_xml_node(self, node, name):
        return node.getElementsByTagName(name) if node else []

    def get_xml_data(self, fileName = 'cbs_0003#.xml'):
        doc = minidom.parse(fileName)
        root = doc.documentElement
        sense_node = self.get_xml_node(root, 'Sense')
        # senselist = []
        for node in sense_node:
            # 获取属性
            sense_type = self.get_attr_value(node, 'type')
            sense_rel_no = self.get_attr_value(node, 'RelNO')
            sense_content = self.get_attr_value(node, 'content')
            print(sense_type, sense_rel_no, sense_content)
            # 获取source
            node_source = self.get_xml_node(node, 'Source')
            source_con = self.get_node_value(node_source[0])
            print(source_con)
            # 获取连词
            node_connect = self.get_xml_node(node, 'Connectives')
            for node_con in node_connect:
                node_con_span = self.get_xml_node(node_con, 'Span')
                connective_span = self.get_node_value(node_con_span[0])
                node_con_content = self.get_xml_node(node_con, 'Content')
                connective_contend = self.get_node_value(node_con_content[0])
                # source_con = self.get_node_value(node_source[0])
                print(connective_span, connective_contend)
            # 获取args1
            node_agr1 = self.get_xml_node(node, 'Arg1')
            for node_agr in node_agr1:
                node_span = self.get_xml_node(node_agr, 'Span')
                arg1_span = self.get_node_value(node_span[0])
                node_content = self.get_xml_node(node_agr, 'Content')
                arg1_contend = self.get_node_value(node_content[0])
                print(arg1_span, arg1_contend)
            #获取args2
            node_agr2 = self.get_xml_node(node, 'Arg2')
            for node_agr in node_agr2:
                node_span = self.get_xml_node(node_agr, 'Span')
                arg2_span = self.get_node_value(node_span[0])
                node_content = self.get_xml_node(node_agr, 'Content')
                arg2_contend = self.get_node_value(node_content[0])
                print(arg2_span, arg2_contend)
            # 获取 注解
            node_anno = self.get_xml_node(node, 'Annotation')
            annotation = self.get_node_value(node_anno[0])
            print(annotation)




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
    for item in find_specific_files(d, patterns=['*(.p)[0-9]']):
        timelist.append(item)
    for item in find_specific_files(d, patterns=['*(.p)(\d)']):
        another.append(item)
    for item in find_specific_files(new_d, patterns=['*.xml']):
        newfilelist.append(item)
    print(len(timelist),len(newfilelist),len(another))

    for i in timelist:
        print(i)
    print('--------------------------------'*10)
    for i in another:
        print(i)
    print('--------------------------------'*10)
    for i in newfilelist:
        print(i)

if __name__ == '__main__':
    # check()
    change_all_file_format()

    # result = sorted(timelist, key=lambda list: list[1], reverse=True)[:5]
    # for a in result:
    #     print(a[0], time.ctime(a[1]))


    # reader = DataReader()
    # text = changeFormat('cbs_0003#.p1')
    # write_text(text, 'cbs_0003#.p1')
    # reader.get_xml_data()