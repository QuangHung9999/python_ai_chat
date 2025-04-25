# tạo file `setup_chatbot.sh`

paste nội dung dưới đây:

```
#!/bin/bash

# 💥 Stop on error
set -e

echo "🚀 Bắt đầu setup chatbot..."

# --- System packages ---
echo "📦 Cập nhật hệ thống & cài Python + pip + venv..."
sudo apt update -y
sudo apt install -y python3-pip python3-venv

# --- Virtual environment ---
echo "🐍 Tạo virtual environment..."
python3 -m venv chatbot-env
source chatbot-env/bin/activate

# --- requirements.txt ---
echo "📦 Tạo requirements.txt..."
cat <<EOF > requirements.txt
altair==4.2.2
annotated-types==0.7.0
anthropic==0.29.0
anyio==4.3.0
asttokens==2.4.1
attrs==23.2.0
audio-recorder-streamlit==0.0.8
blinker==1.8.2
cachetools==5.3.3
certifi==2024.2.2
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
comm==0.2.2
debugpy==1.8.1
decorator==5.1.1
distro==1.9.0
entrypoints==0.4
executing==2.0.1
filelock==3.15.4
fsspec==2024.6.0
gitdb==4.0.11
GitPython==3.1.43
google-ai-generativelanguage==0.6.4
google-api-core==2.19.0
google-api-python-client==2.133.0
google-auth==2.30.0
google-auth-httplib2==0.2.0
google-generativeai==0.6.0
googleapis-common-protos==1.63.1
grpcio==1.64.1
grpcio-status==1.62.2
h11==0.14.0
httpcore==1.0.5
httplib2==0.22.0
httpx==0.27.0
huggingface-hub==0.23.4
idna==3.7
importlib-metadata==6.11.0
IProgress==0.4
ipykernel==6.29.4
ipython==8.24.0
jedi==0.19.1
Jinja2==3.1.4
jiter==0.4.2
jsonschema==4.22.0
jsonschema-specifications==2023.12.1
jupyter_client==8.6.1
jupyter_core==5.7.2
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib-inline==0.1.7
mdurl==0.1.2
nest-asyncio==1.6.0
numpy==1.26.4
openai==1.30.1
packaging==23.2
pandas==2.2.2
parso==0.8.4
pillow==10.3.0
platformdirs==4.2.2
prompt-toolkit==3.0.43
proto-plus==1.23.0
protobuf==4.25.3
psutil==5.9.8
pure-eval==0.2.2
pyarrow==16.1.0
pyasn1==0.6.0
pyasn1_modules==0.4.0
pydantic==2.7.1
pydantic_core==2.18.2
pydeck==0.9.1
Pygments==2.18.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
pytz==2024.1
PyYAML==6.0.1
pyzmq==26.0.3
referencing==0.35.1
requests==2.32.2
rich==13.7.1
rpds-py==0.18.1
rsa==4.9
six==1.16.0
smmap==5.0.1
sniffio==1.3.1
stack-data==0.6.3
streamlit==1.34.0
tenacity==8.3.0
tokenizers==0.19.1
toml==0.10.2
toolz==0.12.1
tornado==6.4
tqdm==4.66.4
traitlets==5.14.3
typing_extensions==4.11.0
tzdata==2024.1
tzlocal==5.2
uritemplate==4.1.1
urllib3==2.2.1
validators==0.28.1
watchdog==4.0.0
wcwidth==0.2.13
zipp==3.18.2
EOF

# --- Install packages ---
echo "📦 Cài Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# --- .env file ---
echo "🔑 Tạo file .env..."
cat <<EOF > .env
GOOGLE_API_KEY= YOUR_GOOGLE_KEY
EOF

# --- Run the app ---
echo "🚀 Chạy Streamlit app..."
streamlit run app.py --server.port 8501 --server.enableCORS false

```

`scp -i chatbot_ec2_key.pem setup_chatbot.sh ubuntu@<aws.ip>:/home/ubuntu/`

`sudo reboot`

`chmod +x setup_chatbot.sh`
`./setup_chatbot.sh `

# đẩy `app.py` lên ec2

`scp -i ./aws/chatbot_ec2_key.pem app.py ubuntu@<aws.ip>:/home/ubuntu/`

# Tạo inbound rule ec2

ipv4 CustomTCP 8501 0.0.0.0/0
`streamlit run app.py --server.port 8501 --server.enableCORS false`

You can access to web using given IP port !!!

# Create domain (Optional)

Go to duckdns then create domain and paste public ipv4 of ec2

## Cài đặt Nginx trên EC2

SSH vào EC2:
`sudo apt update`
`sudo apt install -y nginx`

## Cấu hình Nginx reverse proxy

`sudo nano /etc/nginx/sites-available/streamlit`

```
server {
    listen 80;
    server_name notgpt.duckdns.org;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Kích hoạt config và restart Nginx

```
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Kích hoạt môi trường ảo (nếu bạn đã tạo bằng venv)

```
cd /home/ubuntu
source chatbot-env/bin/activate
```

## Chạy app bằng tmux để giữ nó chạy nền

`tmux`

`streamlit run app.py --server.port 8501 --server.enableCORS false`

## Thoát khỏi tmux mà không dừng app

Nhấn: Ctrl + B, sau đó D
→ App vẫn chạy nền.

http://notgpt.duckdns.org
