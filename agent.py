import os
os.environ["OPENAI_API_KEY"]="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"
from collections import deque
from typing import Dict, List, Optional, Any

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# 임베딩 모델 정의
embedding_model = OpenAIEmbeddings()
# VectorStore를 비어 있음으로 초기화
import faiss

# 임베딩 사이즈를 1536으로
embedding_size = 1536
# faiss에 이 사이즈로 셋업
index = faiss.IndexFlatL2(embedding_size)
# 기초 작업들을 거친 후 VectorStore를 성공적으로 셋업
VectorStore = FAISS(embedding_model.embed_query,index, InMemoryDocstore({}),{})