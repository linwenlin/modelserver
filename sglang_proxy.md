# 腾讯云75 sglang-proxy 与 Paratera 代理服务

本文档以腾讯云75为准：

```text
服务器: 腾讯云75
SSH: ubuntu@111.229.235.75 -p 22
```

腾讯云180 后续不再作为当前服务承载机器使用；本文件不再以腾讯云180状态为准。

## 当前服务总览

腾讯云75上的代理按“独立目录 + 独立 systemd 单元 + 独立进程 + 独立端口”方式部署，避免服务之间互相影响。

```text
8188: sglang-proxy.service                    现有服务，不改动
8189: sglang-proxy-h200.service               现有服务，不改动
8190: sglang-proxy-paratera-deepseek.service  DeepSeek-V4-Pro Paratera 代理
8191: sglang-proxy-paratera-m3.service        MiniMax-M3 Paratera 代理
```

## 现有服务：8188 / 8189

> 这两个服务是已有服务，部署 Paratera 代理时不要修改它们的目录、端口、systemd 单元或配置。

### `sglang-proxy.service`

```text
服务名: sglang-proxy.service
目录: /home/david/sglang-proxy
端口: 8188
启动: /home/david/sglang-proxy/sglang-proxy -max-concurrent 400
```

用途：现有 MiniMax-M2.7 / SGLang 链路。

### `sglang-proxy-h200.service`

```text
服务名: sglang-proxy-h200.service
目录: /home/david/sglang-proxy-h200
端口: 8189
启动: /home/david/sglang-proxy-h200/sglang-proxy-h200 -max-concurrent 1024
```

用途：现有 H200 / MiniMax-M2.5 链路。

## Paratera 代理实现方式

Paratera 服务不复用现有 Go 版 `sglang-proxy` 二进制，而是使用独立 Python 代理程序。

原因：现有 Go 版 `sglang-proxy` 的普通推理后端是 SGLang 原生 `/generate`：

```text
sglang-proxy - Anthropic Messages API to SGLang proxy
Translates Anthropic /v1/messages requests into SGLang /generate calls.
```

Paratera 是 OpenAI/Anthropic 兼容 API，不支持 SGLang 原生 `/generate`：

```text
GET  /v1/models              -> 200
POST /v1/chat/completions    -> 200
POST /v1/responses           -> 200
POST /v1/messages            -> 200
POST /generate               -> 404
```

因此不能通过复制现有 Go 版 `sglang-proxy` 并修改 `SGLANG_URL` 实现 Paratera 转发。

Python 代理能力：

```text
1. 对外兼容 OpenAI /v1/chat/completions
2. 对外兼容 Anthropic /v1/messages，包括 Claude Code 常用的 stream=true
3. 支持 Transfer-Encoding: chunked 请求体，避免 Claude Code 请求被解析成 Bad request syntax
4. 普通推理转发到 Paratera /v1/messages 或 /v1/chat/completions
5. web_search 请求继续转发到 antiapi
6. 每个模型一个独立目录、独立 systemd 服务、独立进程、独立端口
```

## DeepSeek-V4-Pro Paratera 代理

### 基本信息

```text
服务名: sglang-proxy-paratera-deepseek.service
目录: /home/david/sglang-proxy-paratera-deepseek
监听: 0.0.0.0:8190
模型上游: https://llmapi.paratera.com
默认模型: DeepSeek-V4-Pro
客户端密钥: meta-deepseek-2026
web_search 上游: https://lisa.vspeak.top
启动程序: /home/david/sglang-proxy-paratera-deepseek/sglang-proxy-paratera-deepseek
```

### 链路

普通推理：

```text
Client
  -> http://111.229.235.75:8190
  -> sglang-proxy-paratera-deepseek.service
  -> Paratera /v1/messages 或 /v1/chat/completions
  -> DeepSeek-V4-Pro
```

web_search：

```text
Client
  -> http://111.229.235.75:8190
  -> sglang-proxy-paratera-deepseek.service
  -> ANTIAPI_URL=https://lisa.vspeak.top
```

### 配置形态

```env
HOST=0.0.0.0
PORT=8190

MODEL_UPSTREAM_URL=https://llmapi.paratera.com
MODEL_UPSTREAM_MODEL=DeepSeek-V4-Pro
MODEL_UPSTREAM_API_KEY=...

ANTIAPI_URL=https://lisa.vspeak.top
ANTIAPI_KEY=...
WEB_SEARCH_MODEL=claude-sonnet-4-6

PROXY_API_KEY=meta-deepseek-2026
TIMEOUT_SECONDS=600
```

### systemd 单元

```ini
[Unit]
Description=sglang-proxy Paratera DeepSeek routing (port 8190)
After=network.target

[Service]
Type=simple
User=david
WorkingDirectory=/home/david/sglang-proxy-paratera-deepseek
ExecStart=/home/david/sglang-proxy-paratera-deepseek/sglang-proxy-paratera-deepseek
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 已验证测试

```text
GET  http://111.229.235.75:8190/                         -> 200
GET  http://111.229.235.75:8190/v1/models                 -> 200，需要 Authorization
POST http://111.229.235.75:8190/v1/chat/completions       -> 200，返回 model=DeepSeek-V4-Pro
POST http://111.229.235.75:8190/v1/messages?beta=true     -> 200，SSE 返回 model=DeepSeek-V4-Pro
```

示例：

```bash
curl -i http://111.229.235.75:8190/

curl -i http://111.229.235.75:8190/v1/models \
  -H 'Authorization: Bearer meta-deepseek-2026'

curl -i http://111.229.235.75:8190/v1/chat/completions \
  -H 'Authorization: Bearer meta-deepseek-2026' \
  -H 'Content-Type: application/json' \
  -d '{"model":"ignored","messages":[{"role":"user","content":"请只输出 OK"}],"max_tokens":16}'
```

## MiniMax-M3 Paratera 代理

M3 不复用 DeepSeek 的进程；它是独立目录、独立 systemd 服务、独立进程和独立端口。

```text
服务名: sglang-proxy-paratera-m3.service
目录: /home/david/sglang-proxy-paratera-m3
监听: 0.0.0.0:8191
模型上游: https://llmapi.paratera.com
默认模型: MiniMax-M3
客户端密钥: meta-minimax-2026
web_search 上游: https://lisa.vspeak.top
启动程序: /home/david/sglang-proxy-paratera-m3/sglang-proxy-paratera-m3
```

### 链路

```text
Client
  -> http://111.229.235.75:8191
  -> sglang-proxy-paratera-m3.service
  -> Paratera /v1/messages 或 /v1/chat/completions
  -> MiniMax-M3
```

### 配置形态

```env
HOST=0.0.0.0
PORT=8191

MODEL_UPSTREAM_URL=https://llmapi.paratera.com
MODEL_UPSTREAM_MODEL=MiniMax-M3
MODEL_UPSTREAM_API_KEY=...

ANTIAPI_URL=https://lisa.vspeak.top
ANTIAPI_KEY=...
WEB_SEARCH_MODEL=claude-sonnet-4-6

PROXY_API_KEY=meta-minimax-2026
TIMEOUT_SECONDS=600
```

### systemd 单元

```ini
[Unit]
Description=sglang-proxy Paratera MiniMax-M3 routing (port 8191)
After=network.target

[Service]
Type=simple
User=david
WorkingDirectory=/home/david/sglang-proxy-paratera-m3
ExecStart=/home/david/sglang-proxy-paratera-m3/sglang-proxy-paratera-m3
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 已验证测试

```text
GET  http://111.229.235.75:8191/                         -> 200
POST http://111.229.235.75:8191/v1/chat/completions       -> 200，返回 model=MiniMax-M3
POST http://111.229.235.75:8191/v1/messages?beta=true     -> 200，SSE 返回 model=MiniMax-M3
```

示例：

```bash
curl -i http://111.229.235.75:8191/

curl -i http://111.229.235.75:8191/v1/chat/completions \
  -H 'Authorization: Bearer meta-minimax-2026' \
  -H 'Content-Type: application/json' \
  -d '{"model":"ignored","messages":[{"role":"user","content":"请只输出 OK"}],"max_tokens":16}'
```

## Claude Code 使用注意

DeepSeek 和 M3 代理都已支持 Claude Code 常见的 Anthropic Messages 流式请求形式：

```text
POST /v1/messages?beta=true
Transfer-Encoding: chunked
stream=true
```

之前 DeepSeek 8190 出现过如下错误：

```text
Bad request syntax ('{"max_tokens":1024,...}POST /v1/messages?beta=true HTTP/1.1')
```

原因是 Python `BaseHTTPRequestHandler` 不会自动解码 `Transfer-Encoding: chunked` 请求体，导致 JSON body 残留在连接里，被当成下一条 HTTP request line。当前 8190 / 8191 脚本已加入 chunked body 读取逻辑，并已通过公网复现测试。

## 运维命令

DeepSeek：

```bash
sudo systemctl status sglang-proxy-paratera-deepseek.service --no-pager -l
sudo journalctl -u sglang-proxy-paratera-deepseek.service -n 100 --no-pager
sudo systemctl restart sglang-proxy-paratera-deepseek.service
```

M3：

```bash
sudo systemctl status sglang-proxy-paratera-m3.service --no-pager -l
sudo journalctl -u sglang-proxy-paratera-m3.service -n 100 --no-pager
sudo systemctl restart sglang-proxy-paratera-m3.service
```

端口检查：

```bash
ss -ltnp | grep -E ':(8188|8189|8190|8191)\b'
```
