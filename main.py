from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from whisperlivekit import AudioProcessor, TranscriptionEngine, parse_args
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transcription_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """在服务启动时初始化转录引擎，避免每个连接重复加载模型。"""
    global transcription_engine
    # 你可以在这里调整模型大小和语言
    # 参数示例：model="medium"（更大更准），diarization=True（开启说话人识别）
    transcription_engine = TranscriptionEngine(model="base", lan="zh")
    yield
    # 可以在这里添加清理逻辑


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_index():
    """返回自定义的前端页面。"""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


async def handle_websocket_results(websocket: WebSocket, results_generator):
    """消费转录结果并通过 WebSocket 发送给前端。"""
    try:
        async for response in results_generator:
            # 在这里，你可以对 response 进行后处理，比如添加翻译
            # response 中通常包含 "text" 字段
            await websocket.send_json(response)
    except Exception as e:
        logger.warning(f"WebSocket 结果处理出错: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    """处理音频流并返回转录结果的核心 WebSocket 端点。"""
    await websocket.accept()
    logger.info("WebSocket 连接已建立。")

    # 为每个连接创建独立的音频处理器，共享全局转录引擎
    audio_processor = AudioProcessor(transcription_engine=transcription_engine)

    # 启动一个异步任务来发送结果
    results_generator = await audio_processor.create_tasks()
    results_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            # 接收前端发来的原始音频字节数据
            message = await websocket.receive_bytes()
            # 送入处理器进行识别
            await audio_processor.process_audio(message)
    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开。")
    except Exception as e:
        logger.error(f"WebSocket 端点出错: {e}")
    finally:
        results_task.cancel()
        await audio_processor.cleanup()


if __name__ == "__main__":
    import uvicorn

    # 启动服务，监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)