import base64
import datetime
import json
import os
import random
import string
import sys
import time


# layui-upload-img
from flask import render_template, request, redirect, flash, url_for, session
from sqlalchemy import and_

sys.path.insert(0, os.path.abspath('..'))
from models import *
from 机器学习.predict import img_predict
from 机器学习.识农API import 小麦病害识别


@app.route('/')
def MainPage():
    return render_template('main.html', 当前时间=time.time())


@app.route('/backmain', methods=['GET', 'POST'])
def BackmainPage():
    nickname = request.values.get("nickname")
    print(nickname)
    entry = user.query.filter(user.nickname == nickname).first()
    print(entry.nickname)
    return render_template('backmain.html', entry=entry)


@app.route('/upload', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        用户ID = session["用户ID"]
        f = request.files['file']
        ori_path = os.path.dirname(os.path.realpath(__file__))
        img_path = ori_path + '/static/imagePredict.jpg'
        f.save(img_path)
        print(img_path)
        if 用户ID == "lkz":
            # 识农API
            host = "http://ldttwebserver.chinanorth2.cloudapp.chinacloudapi.cn"
            url = host + "/static/imagePredict.jpg"
            结果 = 小麦病害识别(url)
            print(结果)
            ImgIdentResult = 结果["识别结果"]
            PredictAccuracy = 结果["准确率"]
        else:
            # 本机机器学习API
            ImgIdentResult, PredictAccuracy = img_predict(img_path)

        # 为查询记录增加新数据
        当前使用时间 = datetime.datetime.now().strftime('%Y-%m-%d')
        本次准确率 = str(PredictAccuracy) + '%'
        本次推测病名 = ImgIdentResult
        # 生成随机字符串作为查询ID
        str_list = [random.choice(string.digits + string.ascii_letters) for i in range(5)]
        print(f.tell())
        f.seek(0)  # 回到文件起始位置
        new_img = 'data:image/jpeg;base64,' + str(base64.b64encode(f.read()), 'utf-8')
        entry = history(本次推测病名, 当前使用时间, 用户ID, 本次准确率, new_img)
        db.session.add(entry)
        db.session.commit()
        print(entry)
        百科条目 = wiki.query.filter(wiki.病名.like('%{0}%'.format(本次推测病名))).first()
        if 百科条目:
            return json.dumps(
                {'code': 0, 'result': ImgIdentResult, 'accuracy': PredictAccuracy, '病名': 百科条目.病名, '图片': 百科条目.图片})
        # 改为返回ajax
        return json.dumps({'code': 0, 'result': ImgIdentResult, 'accuracy': PredictAccuracy, '百科条目': 百科条目})


# 百科的页面Controller逻辑
@app.route('/wiki/', methods=['GET', 'POST'])
@app.route('/wiki/<param>', methods=['GET', 'POST'])
def WikiPage(param=None):
    print("百科传入参数：", param)
    身份 = session["status"]
    if param is None:
        return render_template('百科/diseaseWiki.html', wiki=wiki.query.all(), 身份=身份, 当前时间=time.time())
    elif param == "search":
        if request.method == 'POST':
            return render_template('功能页/error.html', error="Wiki Search 不能是POST")
        查询病名 = request.values.get("查询病名")
        return render_template('百科/diseaseWiki.html', wiki=wiki.query.filter(wiki.病名.like('%{0}%'.format(查询病名))).all(),
                               身份=身份, 当前时间=time.time())

    elif param == "new":
        if request.method == 'GET':
            return render_template('百科/diseaseDetails.html', 身份=身份, 当前时间=time.time())
        entry = wiki(request.form['病名'], request.form['危害'], request.form['症状'],
                     request.form['防治方式'], request.form['图片'],
                     request.form['其他说明'], request.form['访问次数'])
        db.session.add(entry)
        db.session.commit()
        return redirect("/wiki")

    elif param == "details":
        if request.method == 'POST':
            病名 = request.form['原病名']
            editable = True
        else:
            病名 = request.values.get("病名")
            editable = None

        entry = wiki.query.filter(wiki.病名 == 病名).first()
        return render_template('百科/diseaseDetails.html', entry=entry, editable=editable, 身份=身份, 当前时间=time.time())

    elif param == "delete":
        if request.method == 'GET':
            return render_template('功能页/error.html', error="Wiki Delete 不能是GET！")

        # POST
        entry = wiki.query.filter(wiki.病名 == request.form['原病名']).first()
        db.session.delete(entry)
        db.session.commit()
        return redirect("/wiki")
    elif param == "update":
        if request.method == 'GET':
            return render_template('功能页/error.html', error="Wiki Update 不能是GET！")
        # POST
        oldEntry = wiki.query.filter(wiki.病名 == request.form['原病名']).first()
        entry = wiki(request.form['病名'], request.form['危害'], request.form['症状'],
                     request.form['防治方式'], request.form['图片'],
                     request.form['其他说明'], request.form['访问次数'])
        oldEntry.update(entry)
        db.session.commit()
        return render_template('百科/diseaseDetails.html', entry=entry, 身份=身份, 当前时间=time.time())
    else:
        return render_template('功能页/error.html', error="Wiki 没有“{0}”参数！".format(param))


# 登录
@app.route('/login', methods=['GET', 'POST'])
def LoginPage():
    if request.method == 'POST':
        nickname = request.values.get('nickname')
        password = request.values.get('password')
        print('nickname', nickname, 'password', password)

        entry = user.query.filter(user.nickname == nickname).first()
        if entry.password == password:
            session["status"] = entry.status
            session["用户ID"] = entry.nickname
            print("LKZ: Session使用示例：", entry.status)
            return render_template('index.html', entry=entry, 当前时间=time.time())
        else:
            return render_template('功能页/error.html', error='用户名或密码错误！')
    return render_template('用户/login.html')


# 注册
@app.route('/res', methods=['GET', 'POST'])
def ResPage():
    if request.method == 'POST':
        nickname = request.form['nickname']
        entry = user.query.filter(user.nickname == nickname).first()
        if (entry != None):
            return render_template('功能页/error.html', error='该用户名已经被注册过！')

        # POST
        entry = user(request.form['name'], request.form['nickname'], request.form['sex'],
                     request.form['tel'], request.form['location'], request.form['password'], '普通用户')
        db.session.add(entry)
        db.session.commit()
        return redirect('/login')
    else:
        return (render_template('用户/res.html'))


# 个人信息
@app.route('/details', methods=['GET', 'POST'])
def DetailsPage():
    if request.method == 'GET':
        nickname = request.values.get("nickname")
        entry = user.query.filter(user.nickname == nickname).first()
        return render_template('用户/details.html', entry=entry)
    if request.method == 'POST':
        nickname = request.form['nickname']
        oldentry = user.query.filter(user.nickname == nickname).first()
        entry = user(request.form['name'], request.form['nickname'], request.form['sex'], request.form['tel'],
                     request.form['location'], request.form['password'], '普通用户')
        print('new：', entry)
        oldentry.update(entry)

        db.session.commit()
        return render_template('用户/details.html', entry=entry)


# 用户查看
@app.route('/admin/getuser', methods=['GET', 'POST'])
def AdminUserGet():
    return render_template("用户/UserGet.html", time=time.time())


# 用户管理
@app.route('/admin/user', methods=['GET', 'POST'])
def AdminUser():
    return render_template("用户/AdminUser.html", time=time.time())


# 查看用户表_Url
@app.route('/getuser', methods=['GET'])
def userget():
    print('-----------------userget-----------')
    users = user.query.filter(user.status == '普通用户')
    data = []
    for row in users:
        dic = {'name': row.name, 'nickname': row.nickname, 'sex': row.sex,
               'tel': row.tel, 'location': row.location, 'password': row.password}

        data.append(dic)
    print(data)
    count = users.count()
    print(count)
    num = []
    context = {
        "code": 0,
        "msg": "1",
        "count": count,
        "data": data
    }
    print(context)
    return context


# 搜索用户
@app.route('/user/search', methods=['GET', 'POST'])
def Usersearch():
    print('---------------search--------------------')
    dic = request.form.to_dict()
    nickname = dic['data[nickname]']
    print(nickname)
    row = user.query.filter(user.nickname == nickname).first()
    data = []
    data1 = {'name': row.name, 'nickname': row.nickname, 'sex': row.sex,
             'tel': row.tel, 'location': row.location, 'password': row.password}
    data.append(data1)
    datalist = {"code": 0, "msg": "1", "count": 1, "data": data}
    print(datalist)
    return datalist


# 添加用户
@app.route("/user/add", methods=["POST", "GET"])
def Useradd():
    print('-----------------add-------------------')
    edit_data = request.form.to_dict()
    user_info = edit_data['data']

    user_info = json.loads(user_info)
    nickname = user_info["nickname"]
    name = user_info["name"]
    location = user_info["location"]
    sex = user_info["sex"]
    tel = user_info["tel"]
    password = user_info['password']
    entry = user(name, nickname, sex, tel, location, password, '普通用户')
    db.session.add(entry)
    db.session.commit()
    datalist = {'code': 0, 'data': [user_info]}
    print(datalist)
    return datalist


# 编辑用户
@app.route("/user/edit", methods=["POST", "GET"])
def Useredit():
    edit_data = request.form.to_dict()
    user_info = edit_data['newparams']
    print('-----------------edit--------------------')
    user_info = json.loads(user_info)
    entry = user(user_info['name'], user_info['nickname'], user_info['sex'], user_info['tel'],
                 user_info['location'], user_info['password'], '普通用户')
    nickname = user_info['nickname']
    oldentry = user.query.filter(user.nickname == nickname).first()
    print(oldentry.location)
    oldentry.update(entry)
    print(oldentry.location)
    db.session.commit()
    return {'code': 0, 'msg': '1', 'data': [user_info]}


# 删除用户
@app.route('/user/del', methods=["POST", "GET"])
def Userdel():
    print("------------delete-------------")
    del_data = request.form.to_dict()
    user_info = del_data['data']
    print(user_info)
    user_info = json.loads(user_info)
    user.query.filter(user.nickname == user_info['nickname']).delete()
    db.session.commit()
    return {'code': 0}


def is_valid_date(str):
    '''判断是否是一个有效的日期字符串'''
    try:
        time.strptime(str, "%Y-%m-%d")
        return True
    except:
        return False


@app.route('/search')
@app.route('/search/<param>', methods=['GET', 'POST'])
def SearchPage(param=None):
    print("查询传入参数：", param)

    if param == "普通用户":
        return render_template('历史记录/search.html', history=history.query.all())
    elif param == "管理员":
        return render_template('历史记录/search-admin.html', history=history.query.all())
    elif param == "PredictResult":  # 按病虫害种类名查询
        if request.method == 'POST':
            return render_template('功能页/error.html', error="不能是POST")
        name = request.values.get("病名")
        print("结果%s" % name)
        return render_template('历史记录/search.html',
                               history=history.query.filter(history.病名.like('%{0}%'.format(name))).all())
    elif param == "PredictResult-admin":  # 按病虫害种类名查询
        if request.method == 'POST':
            return render_template('功能页/error.html', error="不能是POST")
        name = request.values.get("病名")
        print("结果%s" % name)
        return render_template('历史记录/search-admin.html', history=history.query.filter(history.病名 == name).all())
    elif param == "Timesearch":  # 按时间查询
        if request.method == 'POST':
            return render_template('功能页/error.html', error="不能是POST")
        开始时间 = request.values.get("开始时间")
        结束时间 = request.values.get("结束时间")
        print(type(开始时间))
        print("开始时间%s" % 开始时间)
        print("结束时间%s" % 结束时间)
        return render_template('历史记录/search.html',
                               history=history.query.filter(and_(history.使用时间 >= 开始时间, history.使用时间 <= 结束时间)).all())
    elif param == "Timesearch-admin":  # 按时间查询
        if request.method == 'POST':
            return render_template('功能页/error.html', error="不能是POST")
        开始时间 = request.values.get("开始时间")
        结束时间 = request.values.get("结束时间")
        print(type(开始时间))
        print("开始时间%s" % 开始时间)
        print("结束时间%s" % 结束时间)
        return render_template('历史记录/search-admin.html',
                               history=history.query.filter(and_(history.使用时间 >= 开始时间, history.使用时间 <= 结束时间)).all())
    elif param == "details":  # 跳转详情页，不需要用href="/search/details"来跳转，这个只是借由路由
        if request.method == 'POST':
            return render_template('功能页/error.html', error="不能是POST")
        check_id = request.values.get("详情的ID")
        print("详情的ID%s" % check_id)
        entry = history.query.filter(history.查询ID == check_id).one()

        # src='data:image/png;base64,'+str(base64.b64encode(entry.用户上传图像),'utf-8')
        src = entry.用户上传图像
        # 在后台把image转换成base64格式，并且转换成utf-8编码的字符串才行，然后字符流添加#data:image/*;base64,
        # print(src)
        # 身份方面先考虑做一个值
        return render_template('历史记录/details.html', entry=history.query.filter(history.查询ID == check_id).one(),
                               src=src)  # ,obj,image

    elif param == "details-admin":  # 跳转详情页，不需要用href="/search/details"来跳转，这个只是借由路由
        if request.method == 'POST':
            return render_template('功能页/error.html', error="不能是POST")
        check_id = request.values.get("详情的ID")
        print("详情的ID%s" % check_id)
        entry = history.query.filter(history.查询ID == check_id).one()
        src = entry.用户上传图像
        return render_template('历史记录/details-admin.html', entry=history.query.filter(history.查询ID == check_id).one(),
                               src=src)  # ,obj,image

    elif param == "delete":  # 详情页编辑
        if request.method == 'POST':
            return render_template('功能页/error.html', error="不能是POST")
        # GET
        id = request.values.get("删除的查询ID")
        if not id:
            flash('Please enter all the fields', 'error')
            return
        print(id)
        entry = history.query.filter(history.查询ID == id).first()
        print("查到的记录：", entry)
        db.session.delete(entry)
        db.session.commit()
        # flash('成功删除一条记录')
        return redirect("/search/管理员")

    elif param == "edit":  # 详情页编辑
        if request.method == 'GET':
            id = request.values.get("查询ID")
            if not id:
                flash('Please enter all the fields', 'error')
                return
            print(id)
            entry = history.query.filter(history.查询ID == id).first()
            print("查到的记录：\n")
            entry.print()
            return render_template('历史记录/details-admin.html', entry=entry)

        # request.form['病名']

        # request.form['准确率']
        # POST
        print(request.form)
        详情的ID = request.form['查询ID']
        oldEntry = history.query.filter(history.查询ID == request.form['查询ID']).first()

        # 时间格式规范编辑
        if is_valid_date(request.form['使用时间']):
            使用时间 = request.form['使用时间']
            flash('成功修改使用时间')
        else:
            flash('请按照2021-03-19格式重新输入"使用时间"', 'error')
            使用时间 = oldEntry.使用时间
        # 不可为空
        if len(request.form['病名']) != 0 and len(request.form['准确率']) != 0:
            病名 = request.form['病名']
            准确率 = request.form['准确率']
            flash('成功修改病名或准确率')
        else:
            flash('病名或准确率不可为空', 'error')
            病名 = oldEntry.病名
            准确率 = oldEntry.准确率

        entry = history(病名, 使用时间, request.form['用户ID'], 准确率, oldEntry.用户上传图像)
        oldEntry.update(entry)
        db.session.commit()

        return redirect("/search/details-admin?详情的ID=" + 详情的ID)
    else:
        pass


@app.route('/messageadd', methods=['POST'])
def add_entry():
    entry = message_board(datetime.datetime.now().strftime('%Y/%m/%d'), request.form['desc'], ' ',
                          request.form['title'], '否')
    db.session.add(entry)
    db.session.commit()

    # flash('New entry was successfully posted')
    return redirect(url_for('consult'))


@app.route('/add_reply', methods=['POST'])
def add_reply():
    old_entry = message_board.query.filter(message_board.id == request.form['replyid']).one()
    entry = message_board(old_entry.time, old_entry.content, request.form['reply_content'], old_entry.theme, '是')
    old_entry.update(entry)
    db.session.commit()
    return redirect(url_for('reply'))


@app.route('/messagesearch/<pagename>', methods=['POST'])
def messagesearch(pagename):
    entries = message_board.query.filter().all()
    session['IsSearchSuccess'] = True
    if request.form['opt'] == '1':
        entries = message_board.query.filter(message_board.id == request.form['search_content']).all()
    elif request.form['opt'] == '2':
        string_tmp = '%' + request.form['search_content'] + '%'
        entries = message_board.query.filter(message_board.time.like(string_tmp)).all()
    elif request.form['opt'] == '3':
        string_tmp = '%' + request.form['search_content'] + '%'
        entries = message_board.query.filter(message_board.theme.like(string_tmp)).all()
    elif request.form['opt'] == '4':
        string_tmp = '%' + request.form['search_content'] + '%'
        entries = message_board.query.filter(message_board.content.like(string_tmp)).all()
    if (pagename == 'reply'):
        return render_template('留言板/reply.html', searchresult=entries)
    else:
        return render_template('留言板/admin.html', searchresult=entries)


@app.route('/messagedelete/<messageid>', methods=['POST'])
def messagedelete(messageid):
    entry = message_board.query.filter(message_board.id == messageid).one()
    db.session.delete(entry)
    db.session.commit()

    return redirect(url_for('messageadmin'))


if __name__ == '__main__':
    # 如果你用的是pycharm,将以下内容取消注释并注释其它行可以免装gevent
    app.run(debug=True)

    # from gevent import pywsgi
    # server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    # server.serve_forever()
