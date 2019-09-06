# Coffee Supervisor - (icanhascoffee)

This is the backend for a slack app.

## Getting Started

### [Create a new slack app](https://api.slack.com/slack-apps)

  - From Settings -> Basic information -> App Credentials
    - Fetch ClientId and Client Secret and add to your .env:
      ```
      CLIENTID='yourclientid'
      CLIENTSECRET='yourclientsecret' 
      ```
  - Go to Features -> Slash commands
    - Create new command:
        - Command: /your-command-name
        - Request URL: http://url-to-the-server.com/icanhascoffee

### Start the server
  - run "node app.js <path-to-python>"