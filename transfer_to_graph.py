import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
def insert_multiplication(expression):
    # 定义一个新的字符串用于存储结果
    sp_t = ['π']
    result = []
    n = len(expression)
    
    # 遍历每一个字符
    for i in range(n):
        result.append(expression[i])
        
        if expression[i] == 'c' or expression[i] == 'C':
            if i - 1 >= 0 and i + 1 < n and expression[i + 1].isdigit() and expression[i - 1].isdigit():
                result.append('@')
        # 如果当前字符是字母
        elif expression[i].isalpha() or expression[i] in sp_t:
            # 检查后一个字符是否是数字或左括号
            if i + 1 < n and (expression[i + 1].isdigit() or expression[i + 1] == '('):
                result.append('*')
        
        # 如果当前字符是数字
        elif expression[i].isdigit():
            # 检查后一个字符是否是字母或左括号
            if i + 1 < n and (expression[i + 1].isalpha() or expression[i + 1] == '(' or expression[i + 1] in sp_t):
                result.append('*')
        
        # 如果当前字符是右括号
        elif expression[i] == ')':
            # 检查后一个字符是否是字母或数字
            if i + 1 < n and (expression[i + 1].isalpha() or expression[i + 1].isdigit() or expression[i + 1] in sp_t):
                result.append('*')
        elif expression[i] == '.':
            # 检查前一个字符是否是数字
            if i - 1 >= 0 and not expression[i - 1].isdigit():
                temp = result[-1]
                result[-1] = '0'
                result.append('.')
            elif i == 0:
                result[-1] = '0'
                result.append('.')

    # if result[0] == '-':
    #     result.insert(0, '0')
    return ''.join(result)

# 将表达式转换为后缀表达式（逆波兰表达式）
def infix_to_postfixwithout(expression):
    expression = insert_multiplication(expression)
    expression = expression.replace(' ', '')
    expression = expression.replace(',', '')
    expression = expression.replace('×', '*')
    expression = expression.replace('–', '-')
    expression = expression.replace('−', '-')
    expression = expression.replace('[', '(')
    expression = expression.replace(']', ')')
    expression = expression.replace('{', '(')
    expression = expression.replace('}', ')')
    
    precedence = {'+': 1, '-': 1, '*': 2,'&': 2,":":2,"/":2,"%":2, '^': 3, '@':3}
    sp_t = ['π']
    output = []
    n_flag = False
    dot_flag = False
    dl_flag = False
    fs_flag = False
    l_fs_flag = False
    operators = deque()
    # 修改正则表达式以识别字母
    tokens = re.findall(r'\d+|π||.||[a-zA-Z]+|\+|\-|\*|\/|\^|\(|\)|π|.|%|!|:|@|&', expression)
    tokens = [token for token in tokens if len(token) != 0 and ' ' not in token]
    for idx, token in enumerate(tokens):
        if token.isdigit() or token.isalpha() or token in sp_t:
            if idx > 0 and tokens[idx - 1].isalpha() and token.isalpha():
                output[-1] += token
            elif n_flag:
                output.append('-'+token)
                n_flag = False
            elif dot_flag:
                output[-1] += '.' + token
                dot_flag = False
            elif fs_flag:
                output[-1] += '/' + token
                fs_flag = False
                l_fs_flag = True
            elif dl_flag:
                output[-1] += '_' + token
                dl_flag = False
            else :
                output.append(token)
            if dot_flag:
                output[-1] += '.' + token
        elif token == '/' and (idx >= len(tokens) - 1 or (not tokens[idx + 1] == '(')):
            fs_flag = True
            if l_fs_flag :
                fs_flag = False
        elif token == '.':
            dot_flag = True
        elif token == '_':
            dl_flag = True
        elif token == '%' and  (idx >= len(tokens) - 1 or not (tokens[idx + 1].isdigit() or tokens[idx + 1].isalpha())):
            
            output[-1] += '%'
        elif token == '!':
            output[-1] += '!'
        elif token == '-' and (idx == 0 or not (tokens[idx - 1].isdigit() or tokens[idx - 1].isalpha() or tokens[idx - 1] in sp_t or tokens[idx - 1] == ')')):
            n_flag = True
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()  # remove '('
        else:
            while (operators and operators[-1] != '(' and
                   precedence[token] <= precedence[operators[-1]]):
                output.append(operators.pop())
            operators.append(token)
            l_fs_flag = False
    
    while operators:
        output.append(operators.pop())
    return output

# 将表达式转换为后缀表达式（逆波兰表达式）
def infix_to_postfix(expression):
    expression = insert_multiplication(expression)
    expression = expression.replace(' ', '')
    expression = expression.replace(',', '')
    expression = expression.replace('×', '*')
    expression = expression.replace('–', '-')
    expression = expression.replace('−', '-')
    expression = expression.replace('[', '(')
    expression = expression.replace(']', ')')
    expression = expression.replace('{', '(')
    expression = expression.replace('}', ')')
    
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2,'&': 2,":":2,"%":2, '^': 3, '@':3}
    sp_t = ['π']
    output = []
    n_flag = False
    dot_flag = False
    dl_flag = False
    operators = deque()
    # 修改正则表达式以识别字母
    tokens = re.findall(r'\d+|π||.||[a-zA-Z]+|\+|\-|\*|\/|\^|\(|\)|π|.|%|!|:|@|&', expression)
    tokens = [token for token in tokens if len(token) != 0 and ' ' not in token]
    for idx, token in enumerate(tokens):
        if token.isdigit() or token.isalpha() or token in sp_t:
            if idx > 0 and tokens[idx - 1].isalpha() and token.isalpha():
                output[-1] += token
            elif n_flag:
                output.append('-'+token)
                n_flag = False
            elif dot_flag:
                output[-1] += '.' + token
                dot_flag = False
            elif dl_flag:
                output[-1] += '_' + token
                dl_flag = False
            else :
                output.append(token)
            if dot_flag:
                output[-1] += '.' + token
        elif token == '.':
            dot_flag = True
        elif token == '_':
            dl_flag = True
        elif token == '%' and  (idx >= len(tokens) - 1 or not (tokens[idx + 1].isdigit() or tokens[idx + 1].isalpha())):
            
            output[-1] += '%'
        elif token == '!':
            output[-1] += '!'
        elif token == '-' and (idx == 0 or not (tokens[idx - 1].isdigit() or tokens[idx - 1].isalpha() or tokens[idx - 1] in sp_t or tokens[idx - 1] == ')')):
            n_flag = True
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()  # remove '('
        else:
            while (operators and operators[-1] != '(' and
                   precedence[token] <= precedence[operators[-1]]):
                output.append(operators.pop())
            operators.append(token)
    
    while operators:
        output.append(operators.pop())
    return output

# 从后缀表达式提取计算数和操作符
def extract_numbers_and_operations(postfix):
    sp_t = ['π']
    numbers = []
    operations = []
    stack = []
    f_flag = 1
    idx = 0
    for token in postfix:
        if token.isdigit() or token.isalpha() or token in sp_t or len(token) > 1:
            stack.append(token)
        else:
            # 每次遇到操作符，弹出两个操作数
            num2 = stack.pop()
            num1 = stack.pop()
            numbers.append(num1)
            numbers.append(num2)
            operations.append(token)
            stack.append('idx' + str(idx))
            idx += 1
    if len(operations) == 0:
        return postfix, operations
    return numbers, operations

def simplify(x):
    x=x.strip()
    try:
        x1=float(x)
    except:
        return x
    x2=int(x1)
    if x1==x2:
        return str(x2)
    else:
        return str(x1)
    
def split_elementssp(string):
    postfix = infix_to_postfixwithout(string)
    # 提取计算数和操作符
    numbers, operations = extract_numbers_and_operations(postfix)
    operators=['+','-','*','/','^']
    return [simplify(string) for string in numbers if string.strip() not in operators],operations

def split_elements(string):
    postfix = infix_to_postfix(string)
    # 提取计算数和操作符
    numbers, operations = extract_numbers_and_operations(postfix)
    operators=['+','-','*','/','^']
    return [simplify(string) for string in numbers if string.strip() not in operators],operations
    
def copy_add(G,val,node2num,num2node,nodename):
    # do copy the same val node
    nodename+=1
    savenode=num2node[val]
    num2node.pop(val)
    num2node[val+'_']=savenode
    node2num[savenode]=val+'_'
    G.add_node(nodename,value=val,op=[])
    num2node[val]=nodename
    node2num[nodename]=val
    # print(f'add node in copy: val:{val},nodename:{nodename}')
    # print(f'past node in copy: val:{val},nodename:{savenode}')
    G.nodes[savenode]['value']=val+'_'
    return G,node2num,num2node,nodename,savenode

def addnode(G,val,op,node2num,num2node,nodename):
    if val in node2num.values():
        G.nodes[num2node[val]]['op']=op
        return G,node2num,num2node,nodename
    nodename+=1
    G.add_node(nodename,value=val,op=op)
    # print(f'add node: val:{val},op:{op},nodename:{nodename}')
    node2num[nodename]=val
    num2node[val]=nodename
    return G,node2num,num2node,nodename

def addedge(G,lval,rval,node2num,num2node,nodename,copy=False, copy_r = False):
    if not copy:
        Gtemp=G.copy()
        if lval not in node2num.values():
            Gtemp,node2num,num2node,nodename=addnode(Gtemp,lval,[],node2num,num2node,nodename)
        # print(f'add edge: lval:{lval},rval:{rval}')
        Gtemp.add_edge(num2node[lval],num2node[rval])
        if nx.is_directed_acyclic_graph(Gtemp):
            G=Gtemp
            return G,node2num,num2node,nodename
        else:
            # plot_graph(Gtemp)
            copy=True
    if copy:
        # print(1)
        G,node2num,num2node,nodename,savenode=copy_add(G,lval,node2num,num2node,nodename)
        if copy_r :
            G.add_edge(nodename,num2node[rval+'_'])
        else:
            G.add_edge(nodename,num2node[rval])
        return G,node2num,num2node,nodename
    
def add_from_list(G,llist,rval,node2num,num2node,nodename,l_ns):
    save_add=[]
    rnode=num2node[rval]
    r_f = False
    for lval in llist:
        if lval==rval:
            r_f = True 
            G,node2num,num2node,nodename=addedge(G,lval,rval,node2num,num2node,nodename,True, r_f)
        # elif lval in l_ns :
        #     G,node2num,num2node,nodename=addedge(G,lval,rval,node2num,num2node,nodename,True,r_f)
        elif lval in save_add:
            # copy node and add edge
            G,node2num,num2node,nodename=addedge(G,lval,rval,node2num,num2node,nodename,True,r_f)
        else:
            # add node and add edge
            G,node2num,num2node,nodename=addedge(G,lval,rval,node2num,num2node,nodename)
        
        save_add.append(lval)
    return G,node2num,num2node,nodename
            

def build_graph(equation_str,sp_flag = False):
    
    eqs=equation_str.split('\n')
    eqs.reverse()
    g_flag = False
    G=nx.DiGraph()
    LCM_flag = False
    nodename=0
    node2num={}
    num2node={}
    l_ns = []
    ans_node = "@@answer@@"
    eq_id = 0
    for eqidx, eq in enumerate(eqs):
        eq_id+=1
        eq = eq.replace('°','')
        eq = eq.replace('$','')
        if ('<=') in eq:
            lval,rval=eq.split('<=')
            f_op = '<='
        elif '>=' in eq:
            lval,rval=eq.split('>=')
            f_op = '>='
        elif '<' in eq:
            lval, rval = eq.split('<')
            f_op = '<'
        elif '>' in eq:
            lval, rval = eq.split('>')
            f_op = '>'
        elif '≥' in eq:
            lval,rval=eq.split('≥')
            f_op = '≥'
        elif '≤' in eq:
            lval,rval=eq.split('≤')
            f_op = '≤'
        elif '=' in eq:
            lval, rval = eq.split('=')
            f_op = '='
        rval=simplify(rval)
        if 'GCD' in lval:
            lval = re.findall(r'\d+', lval)
            op = ['GCD']
        elif 'LCM' in lval:
            
            lval = re.findall(r'\d+', lval)
            op = ['LCM']
        elif 'HCF' in lval:
            
            lval = re.findall(r'\d+', lval)
            op = ['HCF']
        else:
            if(sp_flag):
                lval,op=split_elementssp(lval)
            else:
                lval,op=split_elements(lval)
        rval_o=rval
        if(sp_flag):
            rval,rop=split_elementssp(rval)
        else:
            rval,rop=split_elements(rval)
        # rval,rop=split_elements(rval)
        for i in range(len(lval)):
            if 'idx' in lval[i]:
                lval[i] = str(eq_id) + lval[i]
        for i in range(len(rval)):
            if 'idx' in rval[i]:
                rval[i] = str(eq_id) + 'r'+rval[i]
        for i in range(len(op)):
            if op[i] == '&':
                op[i] = '%'

        if eqidx == 0 and len(lval) == len(rval) and len(lval) == 1:
            if bool(re.search(r'[a-zA-Z]', lval[0])):
                ans_node = lval[0]
                continue
        if eqidx == 0:
            for vals in (lval + rval):
                if bool(re.search(r'[a-zA-Z]', vals)) and 'idx' not in vals:
                    if ans_node == '@@answer@@':
                        ans_node = vals
                    elif ans_node != '@@answer@@':
                        if ans_node == vals:
                            continue
                        ans_node = '@@answer@@'
                        break

            if ans_node == '@@answer@@':
                ans_node = rval[0]

                    
        
        if len(rval) == 1 and f_op == '=' or (len(rval) > 1 and rop == ['/']):
            if len(op) > 1 and 'LCM' not in op:
                for idx, n_op in enumerate(op):
                    if idx == len(op)-1 :
                        rv = rval_o
                        lv = [lval[2*idx], lval[2*idx+1]]
                        if 'nullx' in lval[2*idx] or 'nullx' in lval[2*idx+1]:
                            lv = [lvv for lvv in [lval[2*idx], lval[2*idx+1]] if 'nullx' not in lvv]
                        G,node2num,num2node,nodename=addnode(G,rv,n_op,node2num,num2node,nodename)
                        G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                        l_ns += lv
                    else:
                        rv = str(eq_id)+'idx' + str(idx)
                        lv = [lval[2*idx], lval[2*idx+1]]
                        if 'nullx' in lval[2*idx] or 'nullx' in lval[2*idx+1]:
                            lv = [lvv for lvv in [lval[2*idx], lval[2*idx+1]] if 'nullx' not in lvv]
                        G,node2num,num2node,nodename=addnode(G,rv,n_op,node2num,num2node,nodename)
                        G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                        l_ns += lv
            else:
                G,node2num,num2node,nodename=addnode(G,rval_o,op,node2num,num2node,nodename)
                G,node2num,num2node,nodename=add_from_list(G,lval,rval_o,node2num,num2node,nodename,l_ns)
                l_ns += lval
        elif len(rval) > 1 and f_op == '=':
            if len(op) > 1 and 'LCM' not in op:
                for idx, n_op in enumerate(op):
                    if idx == 0:
                        rv = str(eq_id)+'idx0'
                        lv = [lval[idx], lval[idx+1]]
                        if 'nullx' in lval[2*idx] or 'nullx' in lval[2*idx+1]:
                            lv = [lvv for lvv in [lval[2*idx], lval[2*idx+1]] if 'nullx' not in lvv]
                        G,node2num,num2node,nodename=addnode(G,rv,n_op,node2num,num2node,nodename)
                        G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                        l_ns += lv
                    else:
                        rv = str(eq_id)+'idx' + str(idx)
                        lv = [lval[2*idx], lval[2*idx+1]]
                        if 'nullx' in lval[2*idx] or 'nullx' in lval[2*idx+1]:
                            lv = [lvv for lvv in [lval[2*idx], lval[2*idx+1]] if 'nullx' not in lvv]
                        G,node2num,num2node,nodename=addnode(G,rv,n_op,node2num,num2node,nodename)
                        G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                        l_ns += lv    
            if len(rop) > 1:
                for idx, n_op in enumerate(rop):
                    if idx == 0:
                        rv = str(eq_id)+'ridx0'
                        lv = [rval[idx], rval[idx+1]]
                        G,node2num,num2node,nodename=addnode(G,rv,n_op,node2num,num2node,nodename)
                        G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                        l_ns += lv
                    else:
                        rv = str(eq_id)+'ridx' + str(idx)
                        lv = [rval[2*idx], rval[2*idx+1]]
                        G,node2num,num2node,nodename=addnode(G,rv,n_op,node2num,num2node,nodename)
                        G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                        l_ns += lv    
            else:
                rv = str(eq_id)+'ridx0'
                lv = rval
                G,node2num,num2node,nodename=addnode(G,rv,rop,node2num,num2node,nodename)
                G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                l_ns += lval
            rv = str(eq_id)+'final_ans'
            if len(lval) == 1:
                v1 = lval[0]
            else:
                v1 = str(eq_id)+'idx' + str(len(op)-1)
            if len(rval) == 1:
                v2 = rval_o
            else:
                v2 = str(eq_id)+'ridx' + str(len(rop)-1)
            
            lv = [v1,v2]
            G,node2num,num2node,nodename=addnode(G,rv,f_op,node2num,num2node,nodename)
            G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
        else:
            for idx, n_op in enumerate(op):
                rv = str(eq_id)+'idx' + str(idx)
                lv = [lval[2*idx], lval[2*idx+1]]
                if 'nullx' in lval[2*idx] or 'nullx' in lval[2*idx+1]:
                    lv = [lvv for lvv in [lval[2*idx], lval[2*idx+1]] if 'nullx' not in lvv]
                G,node2num,num2node,nodename=addnode(G,rv,n_op,node2num,num2node,nodename)
                G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                l_ns += lv    
            
            for idx, n_op in enumerate(rop):
                rv = str(eq_id)+'ridx' + str(idx)
                lv = [rval[2*idx], rval[2*idx+1]]
                if 'nullx' in rval[2*idx] or 'nullx' in rval[2*idx+1]:
                    lv = [lvv for lvv in [rval[2*idx], rval[2*idx+1]] if 'nullx' not in lvv]
                G,node2num,num2node,nodename=addnode(G,rv,n_op,node2num,num2node,nodename)
                G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
                l_ns += lv    
            rv = str(eq_id)+'final_ans'
            if len(lval) == 1:
                v1 = lval[0]
            else:
                v1 = str(eq_id)+'idx' + str(len(op)-1)
            if len(rval) == 1:
                v2 = rval_o
            else:
                v2 = str(eq_id)+'ridx' + str(len(rop)-1)
            
            lv = [v1,v2]
            G,node2num,num2node,nodename=addnode(G,rv,f_op,node2num,num2node,nodename)
            G,node2num,num2node,nodename=add_from_list(G,lv,rv,node2num,num2node,nodename,l_ns)
        # plot_graph(G)

        # plot_graph(G)
    # print(node2num)
    # while True:
    #     to_find = ans_node + '_'
    #     for ans_nnnn in node2num.values():
    #         if ans_nnnn == to_find:
    #             ans_node += '_'
    #     if ans_node != to_find:
    #         break
    # G,node2num,num2node,nodename=addnode(G,'@@answer@@',[],node2num,num2node,nodename)
    # G,node2num,num2node,nodename=add_from_list(G,[ans_node],'@@answer@@',node2num,num2node,nodename,l_ns)   
    return G
        
def plot_graph(G):
    pos = nx.spring_layout(G)
    values = {node:attr['value'] for node, attr in G.nodes(data=True)}
    ops = {node:attr['op'] for node, attr in G.nodes(data=True)}
    nx.draw(G, pos,with_labels=False)
    for node, (x, y) in pos.items():
        plt.text(x, y, values[node], ha='center', va='bottom')
        plt.text(x,y,ops[node],ha='center',va='top')

    plt.show()



G=build_graph('( ( 33.0 + 4.0 ) + 14.0 )=51',False)
print(nx.is_directed_acyclic_graph(G))
plot_graph(G)

import json
path = 'llama3-8b_complex_cot2.json'
with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)

from tqdm import  tqdm
error_graphs=[]
error_build=[]
for qid,itemsss in tqdm(enumerate(data)):
    equation=data[qid]['formula']
    eqs = equation.replace(' ', '').split('\n')
    lps = []
    rps = []
    F=False
    try:
        for eq in eqs:
            eq = eq.replace('°','')
            eq = eq.replace('$','')
            if ('<=') in eq:
                lval,rval=eq.split('<=')
                f_op = '<='
            elif '>=' in eq:
                lval,rval=eq.split('>=')
                f_op = '>='
            elif '<' in eq:
                lval, rval = eq.split('<')
                f_op = '<'
            elif '>' in eq:
                lval, rval = eq.split('>')
                f_op = '>'
            elif '≥' in eq:
                lval,rval=eq.split('≥')
                f_op = '≥'
            elif '≤' in eq:
                lval,rval=eq.split('≤')
                f_op = '≤'
            elif '=' in eq:
                lval, rval = eq.split('=')
                f_op = '='
            lps.append(lval)
            rps.append(rval)
    except:
        error_build.append(qid)
        continue
    sp_item = []
    for rp in rps:
        try:
            rval,rop=split_elementssp(rp)
        except:
            print('error' + equation)
        for rv in rval:
            if '/' in rv:
                sp_item.append(rv)

    for sp in sp_item:
        for lp in lps:
            if sp in lp:
                F = True
                break 
    if len(equation) == 0:
        with open(f'./save_graph/llama3_8b/{qid}.json','w') as f:
             json.dump(None,f)
    else:
        try:
            G=build_graph(equation,F)
        except:
            # print(qid)
            error_build.append(qid)
            continue
        json_graph=nx.json_graph.node_link_data(G)
        
        if nx.is_directed_acyclic_graph(G):
            with open(f'./save_graph/llama3_8b/{qid}.json','w') as f:
                json.dump(json_graph,f)
        else:
            error_graphs.append(qid)