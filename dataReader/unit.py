# encoding = utf-8


# 连接词信息 ： 连接词位置和连接词内容
class Connective:
    def init(self, span, contend):
        self.span = span
        self.contend = contend


# 论元结构体：论元位置信息 和 论元内容
class ArgsUnit:
    def init(self, span, contend):
        self.span = span
        self.contend = contend

    def get_contend(self):
        return self.contend

# 句间关系的详细类型：类型、关系编号和关系名称
class Sense:
    def init(self, type, relNo, contend):
        self.type = type
        self.relNo = relNo
        self.contend = contend


# 定义篇章单元的结构
class DiscourseUnit:
    def init(self, filepath, sense, source, connective, args1, args2, annotation):
        self.filepath = filepath
        self.sense = sense
        self.source = source
        self.connective = connective
        self.args1 = args1
        self.args2 = args2
        self.annotation = annotation


# 篇章词表构建
# 从语料库中提取一部分；
# 查阅提取一部分

class ConnectivesCate:
    # cate = ''
    # num = 0

    def init(self, cate):
        self.cate = cate
        self.num = 0

    def add(self):
        self.num += 1


class DiscourseCon:


    # 对新建的连词单元进行 初始化
    def init(self, con, category_list):
        self.con = con
        self.category_list = category_list

    # 把连词的种类添加进来
    def insertCategory(self, category):
        flag = True
        # 检索该连词所表示的类别 是否在类别列表中
        for var in self.category_list:
            if category == var.cate:
                # 如果在 数量加一
                var.add()
                flag = False # 该类别已经检索到了
                break
        # 该类别不在列表中，新建一个连词类别，添加进列表
        if flag:
            con = ConnectivesCate()
            con.init(category)
            con.add()
            self.category_list.append(con)

