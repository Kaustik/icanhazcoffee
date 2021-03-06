// Import express and request modules
var express = require('express');
var request = require('request');
const dotenv = require('dotenv');
dotenv.config();

const nocache = require('nocache');

const { WebClient } = require('@slack/web-api');

// An access token (from your Slack app or custom integration - xoxp, xoxb)
const token = process.env.SLACK_TOKEN;

const web = new WebClient(token);

// This argument can be a channel ID, a DM ID, a MPDM ID, or a group ID
const conversationId = 'CN2U03EGY';

// Store our app's ID and Secret. These we got from Step 1.
// For this tutorial, we'll keep your API credentials right here. But for an actual app, you'll want to  store them securely in environment variables. 
var clientId = process.env.CLIENTID;
var clientSecret = process.env.CLIENTSECRET;

var pathToPy = process.argv.slice(2);

let runPy = new Promise(function(success, nosuccess) {
    const { spawn } = require('child_process');
    const pyprog = spawn('python3', [pathToPy]);

    pyprog.stdout.on('data', function(data) {

        success(data);
    });

    pyprog.stderr.on('data', (data) => {

        nosuccess(data);
    });
});

// Instantiates Express and assigns our app variable to it
var app = express();

app.use(nocache());
app.set('etag', false);

// Again, we define a port we want to listen to
const PORT=process.env.PORT;

// Lets start our server
app.listen(PORT, function () {
    //Callback triggered when server is successfully listening. Hurray!
    console.log("Example app listening on port " + PORT);
});


// This route handles GET requests to our root ngrok address and responds with the same "Ngrok is working message" we used before
app.get('/', function(req, res) {
    res.send('Ngrok is working! Path Hit: ' + req.url);
});

// This route handles get request to a /oauth endpoint. We'll use this endpoint for handling the logic of the Slack oAuth process behind our app.
app.get('/oauth', function(req, res) {
    // When a user authorizes an app, a code query parameter is passed on the oAuth endpoint. If that code is not there, we respond with an error message
    if (!req.query.code) {
        res.status(500);
        res.send({"Error": "Looks like we're not getting code."});
        console.log("Looks like we're not getting code.");
    } else {
        // If it's there...

        // We'll do a GET call to Slack's `oauth.access` endpoint, passing our app's client ID, client secret, and the code we just got as query parameters.
        request({
            url: 'https://slack.com/api/oauth.access', //URL to hit
            qs: {code: req.query.code, client_id: clientId, client_secret: clientSecret}, //Query string data
            method: 'GET', //Specify the method

        }, function (error, response, body) {
            if (error) {
                console.log(error);
            } else {
                res.json(body);
                res.end();
            }
        })
    }
});

// Route the endpoint that our slash command will point to and send back a simple response to indicate that ngrok is working
app.post('/icanhascoffee', function(req, res) {
    runPy.then(function(fromRunpy) {
        res.send(fromRunpy);
        res.end();
    }).catch(function (error) {
        console.log('error');
        console.error(error.toString());
    });
});

setInterval(function() {
    runPy.then(function(fromRunpy) {
        (async () => {
            // See: https://api.slack.com/methods/chat.postMessage
            const res = await web.chat.postMessage({ channel: conversationId, text: fromRunpy.toString().replace(/\r?\n|\r/g, " ") });

            // `res` contains information about the posted message
            console.log('Message sent: ', res.ts);
        })();
    });
}, 300000);
