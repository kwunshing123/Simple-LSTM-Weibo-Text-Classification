const express = require('express');
const request = require('request');

var app = express();
const port = 8080

app.set('view engine', 'ejs');

app.get('/', function(req, res) {
  res.render('pages/index', {result: null});
});

app.get('/analysis', function(req, res) {
  var uri = encodeURI('http://localhost:5002/weibo?content=' + req.query.weibo_content);
  request({
    url: uri,
    method: 'GET'
  }, function (err, response){
    if(!err && response.statusCode == 200) {
      res.render('pages/index', {result: JSON.parse(response.body)});
    }
  });
});

app.listen(port);
console.log('Server listening on port ' + port);
