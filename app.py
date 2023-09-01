import werkzeug
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy

werkzeug.cached_property = werkzeug.utils.cached_property
from flask_restplus import Api, Resource
from datetime import datetime
from joblib import load
from bs4 import BeautifulSoup
import requests
import json
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

api = Api()
app = Flask(__name__)
#CORS(app, resources = {r"/api/*":{}})
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data_base.db'
app.config['SECRET_KEY'] = "secret key"
db = SQLAlchemy(app)
api.__init__(app)
CORS(app, resources={r'/*': {'origins': '*'}})

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String(350), nullable=False)
    news_id = db.Column(db.Integer, nullable=False)
    user_name = db.Column(db.String(80), nullable=False)
    toxic_num = db.Column(db.Integer, nullable=False)
    pub_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __init__(self, comment, news_id, user_name, toxic_num=0):
        self.user_name = user_name
        self.comment = comment
        self.news_id = news_id
        self.toxic_num = toxic_num

    @property
    def serialize(self):
        return {'user_name': self.user_name, 'comment': self.comment, 'news_id': self.news_id,
                'toxic_num': self.toxic_num, 'date': self.pub_date}


class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    descript = db.Column(db.String(350), nullable=False)
    title = db.Column(db.String(80), nullable=False)
    news_link = db.Column(db.String(80), nullable=False)
    date = db.Column(db.String(80), nullable=False)
    descript = db.Column(db.String(80), nullable=False)
    img_link = db.Column(db.String(80), nullable=False)

    def __init__(self, title, news_link, date, descript, img_link):
        self.descript = descript
        self.title = title
        self.date = date
        self.img_link = img_link
        self.news_link = news_link

    @property
    def serialize(self):
        return {
            'id': self.id,
            'descript': self.text,
            'title': self.title,
            'date': self.date,
            'img_link': self.img_link,
            'news_link': self.news_link,
        }


db.create_all()

#################################################
class Bs:

    def __init__(self):
        self.url = "https://www.npr.org/sections/world/"

    def crawl(self, db):
        html = requests.get(self.url)
        bsobj = BeautifulSoup(html.text, 'lxml')
        news = []
        titles = []
        news_links = []
        dates = []
        descripts = []
        img_links = []

        for i in bsobj.find_all('div', class_="item-info-wrap"):
            news.append(list(i.children))

        for n in news:
            titles.append(n[1].find('h2', class_="title").get_text())
            news_links.append(list(n[1].find('h2', class_="title").children)[0]['href'])
            dates.append(n[1].find('span', class_="date").get_text().replace("\x95", " : "))
            descripts.append(n[1].find('p', class_="teaser").get_text().replace("\x95", " : "))

        for i in bsobj.find_all('div', class_="item-image"):
            img_links.append(list(i.children)[1].find('img')['data-original'])

        for title, news_link, date, descript, img_link in zip(titles, news_links, dates, descripts, img_links):
            
            old_news = News.query.filter(News.title.contains(title)).all()
            if not old_news:
                new_news = News(title=title, news_link=news_link, date=date, descript=descript, img_link=img_link)
                db.session.add(new_news)
                db.session.commit()

#################################################
stemmer = PorterStemmer()
stopwords = set(stopwords.words("english"))
def prep(txt):
    tmp = [stemmer.stem(t) for t in word_tokenize(txt) if t not in stopwords]
    tmp = " ".join(tmp)
    return tmp
#################################################
b_soup = Bs()
model = joblib.load('my_model.pkl')
vectorize = joblib.load('vectorize.pkl')
loded_vec = CountVectorizer(decode_error="replace", vocabulary=vectorize)


class GetComment(Resource):

    def get(self, idx=1):
        sql_result = Comment.query.filter_by(news_id=idx).with_entities(Comment.user_name, Comment.comment,
                                                                        Comment.toxic_num, Comment.pub_date).all()
        res = [{'user_name': r[0],
                'comment': r[1],
                'toxic_num': r[2],
                'date': r[3]
                } for r in sql_result]
        
        return jsonify(res)

    def post(self, idx=1):
        data = api.payload
        user_name = data['user_name']
        comment = data['text']
        
        toxic_num = model.predict(loded_vec.transform([prep(comment)]))
        
        new_comment = Comment(comment=comment, news_id=idx, user_name=user_name, toxic_num=np.asscalar(toxic_num[0]))
        db.session.add(new_comment)
        db.session.commit()
        sql_result = Comment.query.filter_by(news_id=idx).with_entities(Comment.user_name, Comment.comment, Comment.toxic_num, Comment.pub_date).all()
        res = [{'user_name': r[0],
                'comment': r[1],
                'toxic_num': r[2],
                'date': r[3]
                } for r in sql_result]
        
        return jsonify(res)



class GetNews(Resource):

    def get(self):
        b_soup.crawl(db)
        sql_result = News.query.with_entities(News.news_link, News.img_link, News.date, News.title, News.descript, News.id).all()
        res = [{'news_link': r[0],
                'img_link': r[1],
                'date': r[2],
                'title': r[3],
                'descript': r[4],
                'id': r[5]
                } for r in sql_result]
        return jsonify(res)


api.add_resource(GetComment, '/api/comment/<idx>', methods=['GET', 'POST'])
api.add_resource(GetNews, '/api/news/', methods=['GET'])


############################################


@app.route('/index')
def hello_world():
    return "first page"


if __name__ == '__main__':
    app.run(debug=True)
