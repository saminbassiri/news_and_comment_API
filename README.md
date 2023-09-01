# news_and_comment_API

this app provide API to get crawled news and to tag comments on news websites 

This app has 5 part: 

- a crawler to get news from [NPR website](https://www.npr.org/sections/world/)
- a vectorizer to process comments for clarification 
- a classifier module to detect toxic comments
- a SQLite database to store data
- a flask app providing APIs : get news, get comments and post new comments 

to run this app use below command

```
flask run
```

Couples

1. SQLite

To test local API via UI and more info :

​	[toxic comment classifier](https://toxicity-classifier.netlify.app/)

*For more information about the classifier module see description tap*

