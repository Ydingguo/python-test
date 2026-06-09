import os
import glob
import asyncio
import threading
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, Document
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pymupdf4llm
from cache import get_history, set_history
from database import save_message

load_dotenv()

INDEX_DIR = "index_storage"

def _build_index():
    Settings.llm = OpenAILike(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com/v1",
        max_tokens=1024,
        is_chat_model=True,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./models"
    )

    if os.path.exists(INDEX_DIR):
        print("从缓存加载索引...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    documents = []
    for pdf_path in glob.glob("data/*.pdf"):
        md_text = pymupdf4llm.to_markdown(pdf_path)
        documents.append(Document(text=md_text, metadata={"source": pdf_path}))
    non_pdf_files = [f for f in glob.glob("data/*") if not f.endswith(".pdf")]
    if non_pdf_files:
        documents += SimpleDirectoryReader(input_files=non_pdf_files).load_data()
    print(f"已加载 {len(documents)} 个文档，构建索引中...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"索引已保存至 {INDEX_DIR}/")
    return index

_index = _build_index()
_retriever = _index.as_retriever()

def _build_messages(question: str, session_id: str) -> list:
    nodes = _retriever.retrieve(question)
    context = "\n\n".join([n.get_content() for n in nodes])
    history = [
        ChatMessage(role=MessageRole(h["role"]), content=h["content"])
        for h in get_history(session_id)
    ]
    return [
        ChatMessage(role=MessageRole.SYSTEM,
                    content=f"你是客服助手，请根据以下内容回答用户问题，文档中没有的信息请如实说明：\n\n{context}")
    ] + history + [
        ChatMessage(role=MessageRole.USER, content=question)
    ]

def _append_turn(question: str, answer: str, session_id: str):
    history = get_history(session_id)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    set_history(session_id, history)

    def _write_db():
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)

    threading.Thread(target=_write_db, daemon=True).start()

def ask(question: str, session_id: str) -> str:
    messages = _build_messages(question, session_id)
    answer = Settings.llm.chat(messages).message.content
    _append_turn(question, answer, session_id)
    return answer

async def ask_stream(question: str, session_id: str):
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            messages = _build_messages(question, session_id)
            chunks = []
            for chunk in Settings.llm.stream_chat(messages):
                token = chunk.delta
                if token:
                    chunks.append(token)
                    loop.call_soon_threadsafe(q.put_nowait, token)
            _append_turn(question, "".join(chunks), session_id)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=_run, daemon=True).start()
    while True:
        token = await q.get()
        if token is None:
            break
        yield token
