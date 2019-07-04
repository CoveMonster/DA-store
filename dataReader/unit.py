# encoding = utf-8


# 连接词信息 ： 连接词位置和连接词内容
class Connective:
    def __int__(self, span, contend):
        self.span = span
        self.contend = contend


# 论元结构体：论元位置信息 和 论元内容
class ArgsUnit:
    def __int__(self, span, contend):
        self.span = span
        self.contend = contend


# 句间关系的详细类型：类型、关系编号和关系名称
class Sense:
    def __int__(self, type, relNo, contend):
        self.type = type
        self.relNo = relNo
        self.contend = contend

    def __get__(self):
        return self.relNo, self.contend


# 定义篇章单元的结构
class DiscourseUnit:
    def __int__(self, sense, source, connective, args1, args2, annotation):
        self.sense = sense
        self.source = source
        self.connective = connective
        self.args1 = args1
        self.args2 = args2
        self.annotation = annotation

