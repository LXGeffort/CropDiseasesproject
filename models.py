from flask import Flask
from flask_sqlalchemy import SQLAlchemy
# import pymysql
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///database/wheatDiseasesProject.sqlite3'
# app.config["SQLALCHEMY_DATABASE_URI"] = 'mysql+pymysql://root:root@localhost:3306/liudao'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "random string"
#print('配置后',app.config)
db = SQLAlchemy(app)


class wiki(db.Model):
    __tablename__ = "百科"
    # 核心数据，数据一共7列
    病名 = db.Column(db.String(200), primary_key=True)
    危害 = db.Column(db.Text)
    症状 = db.Column(db.Text)
    防治方式 = db.Column(db.Text)
    # 图片 = db.Column(db.BLOB, nullable=False)这里就不用blob了，直接保存其在前端显示的方式，免得解码
    图片 = db.Column(db.Text)

    其他说明 = db.Column(db.Text)

    访问次数 = db.Column(db.Integer)

    def __init__(self, 病名, 危害, 症状, 防治方式, 图片, 其他说明, 访问次数):
        self.病名 = 病名
        self.危害 = 危害
        self.症状 = 症状
        self.防治方式 = 防治方式
        self.图片 = 图片
        self.其他说明 = 其他说明
        self.访问次数 = 访问次数

    def update(self, entry):
        self.病名 = entry.病名
        self.危害 = entry.危害
        self.症状 = entry.症状
        self.防治方式 = entry.防治方式
        self.图片 = entry.图片
        self.其他说明 = entry.其他说明
        self.访问次数 = entry.访问次数

    def print(self):
        print("\n病名:{0}, 危害:{1}, 症状:{2},防治方式:{3},图片:{4},\
              其他说明:{5},访问次数:{6}\n".format(self.病名, self.危害, self.症状,
                                          self.防治方式, self.图片, self.其他说明, self.访问次数))


# 用户表
class user(db.Model):
    __tablename__ = "用户"
    # 核心数据，数据一共7列
    name = db.Column(db.Text)
    nickname = db.Column(db.String(200), primary_key=True)
    sex = db.Column(db.Text)
    tel = db.Column(db.Text)
    location = db.Column(db.Text)
    # ID自增
    ID = db.Column(db.INTEGER, autoincrement=True)
    password = db.Column(db.String(20))
    status = db.Column(db.Text)

    def __init__(self, name, nickname, sex, tel, location, password, status):
        self.name = name
        self.nickname = nickname
        self.sex = sex
        self.tel = tel
        self.location = location
        self.password = password
        self.status = status

    def update(self, entry):
        self.name = entry.name
        self.nickname = entry.nickname
        self.sex = entry.sex
        self.tel = entry.tel
        self.location = entry.location
        self.ID = entry.ID
        self.password = entry.password
        self.status = entry.status

    def print(self):
        print("\nname:{0}, nickname:{1}, sex:{2},tel:{3},:location{4},\
              ID:{5},password:{6},status:{7}\n".format(self.name, self.nickname, self.sex,
                                                       self.tel, self.location, self.ID, self.password, self.status))


class history(db.Model):
    __tablename__ = "历史记录"
    查询ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    病名 = db.Column(db.Text)
    使用时间 = db.Column(db.Text)
    用户ID = db.Column(db.Text)
    准确率 = db.Column(db.Text)
    用户上传图像 = db.Column(db.Text)

    def __init__(self, 病名, 使用时间, 用户ID, 准确率, 用户上传图像):
        #self.查询ID = 查询ID
        self.病名 = 病名
        self.使用时间 = 使用时间
        self.用户ID = 用户ID
        self.准确率 = 准确率
        self.用户上传图像 = 用户上传图像

    def update(self, entry):
        #self.查询ID = entry.查询ID
        self.病名 = entry.病名
        self.使用时间 = entry.使用时间
        self.用户ID = entry.用户ID
        self.准确率 = entry.准确率
        self.用户上传图像 = entry.用户上传图像

    def print(self):
        print("查询ID:{0}, 病名:{1}, 使用时间:{2}, 用户ID:{3},准确率:{4},\
              用户上传图像:{5}".format(self.查询ID, self.病名, self.使用时间, self.用户ID, self.准确率, self.用户上传图像))


class message_board(db.Model):
    __tablename__ = "留言板"
    #id自增
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    time = db.Column(db.Text)
    content = db.Column(db.Text)
    reply = db.Column(db.Text)
    theme = db.Column(db.Text)
    isreply = db.Column(db.Text)

    def __init__(self, time, content, reply, theme, isreply):
        self.time = time
        self.content = content
        self.reply = reply
        self.theme = theme
        self.isreply = isreply

    def update(self, entry):
        self.time = entry.time
        self.content = entry.content
        self.reply = entry.reply
        self.theme = entry.theme
        self.isreply = entry.isreply

    def print(self):
        print(
            "查询ID:{0},查询时间:{1},留言内容:{2},客服回复:{3},留言主题:{4},是否回复:{5}".format(self.id, self.time, self.content, self.reply,
                                                                           self.theme, self.isreply))
if __name__ == '__main__':
  db.create_all()
