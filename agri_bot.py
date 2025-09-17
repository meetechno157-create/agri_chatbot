import asyncio
import websockets
import json
import os
import torch
import faiss
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# -------------------------------
# TinyLlama RAG Chatbot Class
# -------------------------------
class TinyLlamaRAGChatbot:
    def __init__(self,
                 model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 max_context_length=2048,
                 chunk_size=500,
                 chunk_overlap=50):
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chat_history = []

        print("Loading TinyLlama model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        print("RAG Chatbot initialized successfully!")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def load_text_files(self, file_paths: List[str]) -> List[str]:
        documents = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append(content)
                    print(f"Loaded {file_path}: {len(content)} characters")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return documents

    def chunk_text(self, text: str, source_file: str) -> List[Dict]:
        import re
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = self.count_tokens(sentence)
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'source': source_file,
                    'token_count': current_tokens
                })
                overlap_text = ' '.join(current_chunk.split()[-self.chunk_overlap:])
                current_chunk = overlap_text + ' ' + sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += ' ' + sentence
                current_tokens += sentence_tokens
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'source': source_file,
                'token_count': current_tokens
            })
        return chunks

    def build_rag_database(self, file_paths: List[str]):
        print("Building RAG database...")
        documents = self.load_text_files(file_paths)
        all_chunks = []
        for i, doc in enumerate(documents):
            file_name = os.path.basename(file_paths[i])
            chunks = self.chunk_text(doc, file_name)
            all_chunks.extend(chunks)
            print(f"Created {len(chunks)} chunks from {file_name}")
        self.chunks = [chunk['content'] for chunk in all_chunks]
        self.chunk_metadata = all_chunks
        if self.chunks:
            print("Generating embeddings...")
            embeddings = self.embedding_model.encode(self.chunks)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            print(f"RAG database built with {len(self.chunks)} chunks")

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None:
            return []
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'content': self.chunks[idx],
                    'source': self.chunk_metadata[idx]['source'],
                    'score': float(scores[0][i]),
                    'tokens': self.chunk_metadata[idx]['token_count']
                })
        return results

    def format_chat_history(self, max_history_tokens: int = 10000) -> str:
        if not self.chat_history:
            return ""
        history_text = ""
        current_tokens = 0
        for exchange in reversed(self.chat_history):
            exchange_text = f"User: {exchange['human']}\nAssistant: {exchange['assistant']}\n"
            exchange_tokens = self.count_tokens(exchange_text)
            if current_tokens + exchange_tokens > max_history_tokens:
                break
            history_text = exchange_text + history_text
            current_tokens += exchange_tokens
        return history_text

    def generate_response(self, user_input: str) -> str:
        relevant_chunks = self.retrieve_relevant_chunks(user_input, top_k=3)
        context = ""
        total_context_tokens = 0
        max_context_tokens = self.max_context_length // 2
        for chunk in relevant_chunks:
            chunk_text = f"Source ({chunk['source']}): {chunk['content']}\n\n"
            chunk_tokens = self.count_tokens(chunk_text)
            if total_context_tokens + chunk_tokens > max_context_tokens:
                break
            context += chunk_text
            total_context_tokens += chunk_tokens
        history = self.format_chat_history(max_history_tokens=500)

        system_prompt = """
        - YOUR ROLE IS TO PROVIDE THE INFORMATION RELATED TO AGRICULTURE ONLY, EXCEPT THIS YOU DO NOT HAVE TO ANSWER ANY OTHER QUESTIONS.
        - FOLLOW THE INSTRUCTIONS BELOW STRICTLY:
        You are an expert AI assistant specialized strictly in agriculture.
        Ignore questions from other domains such as IT, politics, human sciences, etc.
        Only answer questions related to agriculture, crops, soil, irrigation, fertilizers, seeds, agri-technology, pests, livestock, or related fields.
        If a question is not about agriculture or no relevant agricultural information found, respond politely:
        "I'm sorry, but I can only answer questions about agriculture."
        Never invent or hallucinate answers.
        Do not exceed the token limit of 2048 in output.
        """

        prompt = f"""<|system|>
{system_prompt}
Agricultural knowledge base context (may be empty):
{context or '[No relevant agricultural information found in the database.]'}
Chat history (for reference only):
{history}
<|user|>
{user_input}
<|assistant|>
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_context_length-200)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_start = full_response.find("<|assistant|>")
        if assistant_start != -1:
            response = full_response[assistant_start + len("<|assistant|>"):].strip()
        else:
            response = full_response[len(prompt):].strip()
        return response

    def chat(self, user_input: str) -> str:
        response = self.generate_response(user_input)
        self.chat_history.append({'human': user_input, 'assistant': response})
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        return response


# -------------------------------
# WebSocket Server
# -------------------------------

# Initialize chatbot
chatbot = TinyLlamaRAGChatbot()
AGRI_DATA_FILES = [
    "/home/itechnolabs/Downloads/combined_text.txt",
    "/home/itechnolabs/Downloads/combined_2.txt",
]
chatbot.build_rag_database(AGRI_DATA_FILES)


connected_clients = set()


async def handler(websocket, path):
    # On connect
    connected_clients.add(websocket)
    print(f"[CONNECTED] Client connected: {websocket.remote_address}")
    await websocket.send(json.dumps({"event": "server_response", "data": {"response": "You are connected!"}}))

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                event = data.get("event")
                payload = data.get("data", {})

                # Handle chat messages
                if event == "chat_message":
                    user_input = payload.get("message", "").strip()
                    if not user_input:
                        await websocket.send(json.dumps({
                            "event": "server_response",
                            "data": {"response": "Please enter a valid message."}
                        }))
                        continue

                    response = chatbot.chat(user_input)
                    await websocket.send(json.dumps({
                        "event": "server_response",
                        "data": {"response": response}
                    }))

                # Get chat history
                elif event == "get_history":
                    await websocket.send(json.dumps({
                        "event": "chat_history",
                        "data": {"history": chatbot.chat_history}
                    }))

                # Ping event
                elif event == "ping":
                    await websocket.send(json.dumps({
                        "event": "pong",
                        "data": {"message": "pong"}
                    }))

                # Unknown event
                else:
                    await websocket.send(json.dumps({
                        "event": "error",
                        "data": {"message": f"Unknown event: {event}"}}
                    ))

            except Exception as e:
                error_msg = f"Internal server error: {str(e)}"
                print("[ERROR]", error_msg)
                await websocket.send(json.dumps({
                    "event": "error",
                    "data": {"message": error_msg}
                }))

    finally:
        # On disconnect
        connected_clients.remove(websocket)
        print(f"[DISCONNECTED] Client disconnected: {websocket.remote_address}")


# -------------------------------
# Start WebSocket Server
# -------------------------------
async def main():
    print("🚀 WebSocket server starting at ws://0.0.0.0:5000")
    async with websockets.serve(handler, "0.0.0.0", 5000):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
