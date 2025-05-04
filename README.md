# First GPT Bot


## Description
This is first GPT-bot for "You are Bot" project created by @mak.a.novikov and @m.semchinov.
It tryes to deceive users that it is a human being.


## !!! WARNING !!!

### this is NOT easy deploy bot!


## Setup

Create .env file and describe server PORT (for uvicorn FastApi) there
(look in .env.example)
Default port is 8672
Also you need to write down there your OpenAI API key (or DeepSeek key) and proxy parameters (if you use them).


## Run

### Pure Python
To run without docker launch main.py file (install requirements.txt before)

```bash
python3 pip install -r requirements.txt
```

```bash
python3 main.py
```

### Docker without database
To run with docker, you need to have docker installed and running on your machine.
Then build the image and run the container:

```bash
docker build -t gptbot:latest .
```

```bash
docker run -d \
  --name gptbot \
  --env-file .env \
  -p 8672:8672 \
  -v "$(pwd)/logs":/app/logs \
  gptbot:latest
```

#### !!! WARNING !!!
There is no database in this version of bot - GPT will answer only current message
 and don't remember previous messages.


### Database (docker compose)
To run with docker and database, you need to have docker and docker-compose installed and running on your machine.
Then build the image and run the container:

```bash
docker-compose up -d --build
```
