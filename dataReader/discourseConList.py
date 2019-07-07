from unit import DiscourseCon, ConnectivesCate

def display(conn_list):
    for conn in conn_list:
        print('*'*20)
        print('连词：%s'%conn.con)
        print('-'*10)
        for var_cate in conn.category_list:
            print('\t', '种类编号： ', var_cate.cate, '\t', ' ; 出现的次数；', var_cate.num)
        print('-' * 10)
        print('*'*20)


def count_conn(disUnit_list):
    #print(1)
    conn_list = []
    # 遍历整个篇章数据
    for unit in disUnit_list:
        #print(2)
        # 得到 连词的名称
        con = unit.connective.contend
        # get 连词的种类
        relNo = unit.sense.relNo
        #print(type(relNo))
        flag = True
        # 在连列表里检索  是否已经在列表中
        for item in conn_list:
            if con == item.con:
                # 如果在，检索类别
                item.insertCategory(relNo)
                flag = False  # 表示已经已经添加了
                break
        # 该连词不在列表中，我们就新建一个篇章连词unit，添加至列表中
        if flag:
            dis_con = DiscourseCon()
            dis_con.init(con, [])
            dis_con.insertCategory(relNo)
            conn_list.append(dis_con)
    print(len(conn_list))
    display(conn_list)
    return conn_list


if __name__ == '__main__':
    print('')


