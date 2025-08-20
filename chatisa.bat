@echo off
cd /d C:\Users\webapp\.conda\chat_isa
C:\Users\webapp\.conda\envs\chatisa\python.exe -m streamlit run chatgpt.py --server.address chatisa.fsb.miamioh.edu --server.port 443 --server.sslCertFile C:\Users\webapp\.conda\chat_isa\ssl\chatisa.pem --server.sslKeyFile C:\Users\webapp\.conda\chat_isa\ssl\chatisapriv.key
pause