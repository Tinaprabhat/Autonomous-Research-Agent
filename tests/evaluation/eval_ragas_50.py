"""
RAGAS evaluation on 50 synthetic QA pairs.

Uses non-LLM metrics only (no OpenAI key required):
  - RougeScore       (ROUGE-L lexical overlap between answer and reference)
  - BleuScore        (n-gram precision)
  - NonLLMStringSimilarity (character-level similarity)
  + our own token-overlap faithfulness and relevance (from metrics.py)

Run:
    python -m tests.evaluation.eval_ragas_50
"""

from __future__ import annotations
import asyncio
import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ragas.metrics.collections import RougeScore, BleuScore, NonLLMStringSimilarity
from tests.evaluation.metrics import faithfulness, relevance, conciseness

# ── 50 synthetic QA pairs ────────────────────────────────────────────────────
# Each entry: query, context chunks, generated answer, reference answer

QA_PAIRS = [
    # ── Transformers & Attention (1-10) ──────────────────────────────────────
    {"query": "What is the attention mechanism?",
     "context": ["Attention allows models to weigh the importance of different tokens relative to each other.",
                 "Queries, keys, and values are used to compute attention scores."],
     "answer": "The attention mechanism computes weighted combinations of values using query-key dot products.",
     "reference": "Attention allows models to focus on relevant parts of the input using queries, keys, and values."},

    {"query": "How does self-attention differ from cross-attention?",
     "context": ["Self-attention computes attention within the same sequence.",
                 "Cross-attention uses queries from one sequence and keys/values from another."],
     "answer": "Self-attention operates within a single sequence, while cross-attention attends across two sequences.",
     "reference": "Self-attention uses one sequence for Q, K, V; cross-attention uses separate sequences for Q versus K and V."},

    {"query": "What is multi-head attention?",
     "context": ["Multi-head attention runs several attention operations in parallel.",
                 "Each head can attend to different positional patterns in the input."],
     "answer": "Multi-head attention runs multiple attention operations in parallel, each focusing on different patterns.",
     "reference": "Multiple attention heads run in parallel, allowing the model to attend to different representation subspaces."},

    {"query": "What is the transformer architecture?",
     "context": ["Transformers consist of stacked encoder and decoder blocks.",
                 "Each block contains multi-head attention and feed-forward sublayers."],
     "answer": "Transformers stack encoder and decoder blocks, each with multi-head attention and feed-forward layers.",
     "reference": "The transformer uses stacked self-attention and feed-forward layers in an encoder-decoder structure."},

    {"query": "What are positional encodings?",
     "context": ["Transformers lack inherent sequential ordering.",
                 "Positional encodings inject order information into token embeddings."],
     "answer": "Positional encodings add order information to embeddings since transformers have no inherent sequence order.",
     "reference": "Positional encodings provide sequence order information that transformers otherwise lack."},

    {"query": "What is the scaled dot-product attention formula?",
     "context": ["Attention is computed as softmax(QK^T / sqrt(d_k)) * V.",
                 "Scaling by sqrt(d_k) prevents vanishing gradients in large dimensions."],
     "answer": "Scaled dot-product attention is softmax(QK^T / sqrt(d_k)) V, where d_k is the key dimension.",
     "reference": "The formula is Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V, scaled to avoid gradient issues."},

    {"query": "What is the feed-forward network in a transformer?",
     "context": ["Each transformer block includes a position-wise feed-forward network.",
                 "The FFN applies two linear transformations with a ReLU activation in between."],
     "answer": "Each transformer block includes a position-wise feed-forward network that applies two linear transformations with a ReLU activation in between.",
     "reference": "A position-wise FFN with two linear transformations and a ReLU in between follows each attention sublayer."},

    {"query": "How do transformers handle variable-length sequences?",
     "context": ["Padding tokens are added to make sequences equal length in a batch.",
                 "Attention masks prevent attending to padding positions."],
     "answer": "Padding tokens are added to make sequences equal length in a batch, and attention masks prevent attending to padding positions.",
     "reference": "Transformers use padding tokens and attention masks to handle variable-length sequences efficiently in batches."},

    {"query": "What is layer normalisation in transformers?",
     "context": ["Layer normalisation stabilises training by normalising activations.",
                 "It is applied after each sublayer in the transformer block."],
     "answer": "Layer normalisation normalises activations after each sublayer to stabilise transformer training.",
     "reference": "Layer norm is applied after each sublayer to normalise activations and improve training stability."},

    {"query": "What is the encoder-only transformer architecture?",
     "context": ["Encoder-only models like BERT use only the encoder stack.",
                 "They are suited for understanding tasks like classification and NER."],
     "answer": "Encoder-only transformers (e.g., BERT) use only the encoder and excel at understanding tasks.",
     "reference": "Encoder-only models use the encoder stack for tasks requiring deep text understanding like classification."},

    # ── BERT & Pre-training (11-20) ───────────────────────────────────────────
    {"query": "How does BERT pre-train?",
     "context": ["BERT uses masked language modelling (MLM) and next sentence prediction (NSP).",
                 "15% of tokens are masked during MLM training."],
     "answer": "BERT pre-trains with masked language modelling (masking 15% of tokens) and next sentence prediction.",
     "reference": "BERT pre-trains using MLM where 15% of tokens are masked, and NSP to learn sentence relationships."},

    {"query": "What is masked language modelling?",
     "context": ["In MLM, a fraction of input tokens are replaced with [MASK].",
                 "The model learns to predict the original masked tokens."],
     "answer": "Masked language modelling masks a fraction of tokens and trains the model to reconstruct them.",
     "reference": "MLM replaces some tokens with [MASK] and trains the model to predict the original tokens."},

    {"query": "What is next sentence prediction?",
     "context": ["NSP trains BERT to determine if one sentence follows another.",
                 "It helps BERT understand sentence-level relationships."],
     "answer": "Next sentence prediction trains the model to determine whether two sentences are consecutive.",
     "reference": "NSP is a pre-training task where BERT learns whether two sentences are naturally consecutive."},

    {"query": "What is fine-tuning in the context of BERT?",
     "context": ["Fine-tuning adds a task-specific head on top of the pre-trained BERT model.",
                 "The entire model is then trained end-to-end on the downstream task."],
     "answer": "Fine-tuning adds a task head to BERT and trains the whole model on a downstream task.",
     "reference": "BERT fine-tuning involves appending a task-specific layer and training on labeled data end-to-end."},

    {"query": "What is the WordPiece tokenisation used by BERT?",
     "context": ["WordPiece splits unknown words into common subwords.",
                 "This allows BERT to handle out-of-vocabulary words."],
     "answer": "WordPiece tokenisation splits words into subwords so BERT can handle out-of-vocabulary terms.",
     "reference": "BERT's WordPiece tokeniser breaks rare words into subword units, enabling robust vocabulary coverage."},

    {"query": "What is the CLS token in BERT?",
     "context": ["The [CLS] token is prepended to every input sequence.",
                 "Its final hidden state is used as a sentence-level representation for classification."],
     "answer": "The [CLS] token is prepended to inputs; its final state represents the whole sequence for classification.",
     "reference": "BERT's [CLS] token aggregates sentence-level information and is used for downstream classification."},

    {"query": "How many layers does BERT-base have?",
     "context": ["BERT-base has 12 transformer layers, 12 attention heads, and 768 hidden dimensions.",
                 "BERT-large has 24 layers and 1024 hidden dimensions."],
     "answer": "BERT-base has 12 layers, 12 heads, and 768 hidden units.",
     "reference": "BERT-base uses 12 transformer encoder layers with 768-dimensional hidden states and 12 attention heads."},

    {"query": "What tasks does BERT excel at?",
     "context": ["BERT achieves strong results on SQuAD, GLUE, and NER tasks.",
                 "Its bidirectional encoding captures rich contextual representations."],
     "answer": "BERT excels at QA (SQuAD), text classification (GLUE), and NER due to its bidirectional encoding.",
     "reference": "BERT performs strongly on tasks like question answering, NER, and text classification."},

    {"query": "What is bidirectional encoding?",
     "context": ["Unlike GPT, BERT reads the entire sequence left-to-right and right-to-left simultaneously.",
                 "This gives each token context from both directions."],
     "answer": "Bidirectional encoding means each token attends to both left and right context simultaneously.",
     "reference": "BERT's bidirectional encoder reads tokens in both directions at once, giving richer context."},

    {"query": "What is the difference between BERT and GPT?",
     "context": ["BERT is encoder-only and pre-trains with MLM.",
                 "GPT is decoder-only and pre-trains with causal language modelling."],
     "answer": "BERT is encoder-only and pre-trains with MLM, while GPT is decoder-only and pre-trains with causal language modelling.",
     "reference": "BERT uses a bidirectional encoder with MLM; GPT uses a unidirectional decoder with causal LM."},

    # ── RAG & Retrieval (21-30) ───────────────────────────────────────────────
    {"query": "What is retrieval-augmented generation?",
     "context": ["RAG combines a retriever that fetches documents with a generator that synthesises answers.",
                 "It improves factuality by grounding responses in retrieved evidence."],
     "answer": "RAG retrieves relevant documents then uses them as context for a generative model to produce answers.",
     "reference": "Retrieval-augmented generation fetches relevant documents and uses them as context for generation."},

    {"query": "What is dense retrieval?",
     "context": ["Dense retrieval encodes queries and documents into dense vectors.",
                 "Similarity search finds the nearest documents to the query vector."],
     "answer": "Dense retrieval embeds queries and documents as vectors and finds similar documents via nearest-neighbour search.",
     "reference": "Dense retrieval uses embedding vectors to find semantically similar documents via similarity search."},

    {"query": "How does BM25 work?",
     "context": ["BM25 ranks documents by term frequency weighted by inverse document frequency.",
                 "It also accounts for document length normalisation."],
     "answer": "BM25 ranks documents using TF-IDF with document length normalisation.",
     "reference": "BM25 is a probabilistic ranking model using term frequency, inverse document frequency, and length normalisation."},

    {"query": "What is hybrid retrieval?",
     "context": ["Hybrid retrieval combines sparse keyword search with dense vector search.",
                 "It captures both exact keyword matches and semantic similarity."],
     "answer": "Hybrid retrieval combines BM25 keyword search with dense vector search to improve recall.",
     "reference": "Hybrid retrieval merges sparse keyword matching with dense semantic search for better document recall."},

    {"query": "What is FAISS?",
     "context": ["FAISS is a library for efficient similarity search over large vector collections.",
                 "It supports both exact and approximate nearest-neighbour search."],
     "answer": "FAISS is a vector search library supporting exact and approximate nearest-neighbour retrieval at scale.",
     "reference": "FAISS enables efficient similarity search and clustering over millions of high-dimensional vectors."},

    {"query": "What is a cross-encoder reranker?",
     "context": ["A cross-encoder scores query-document pairs jointly by processing them together.",
                 "It is more accurate than bi-encoders but slower due to pairwise computation."],
     "answer": "A cross-encoder reranker scores each query-document pair together for high-accuracy reranking.",
     "reference": "Cross-encoders jointly encode query and document pairs for precise relevance scoring in reranking."},

    {"query": "What is the difference between bi-encoder and cross-encoder?",
     "context": ["Bi-encoders embed query and document independently for fast retrieval.",
                 "Cross-encoders process them jointly for higher accuracy but slower speed."],
     "answer": "Bi-encoders are fast but less accurate; cross-encoders are slow but more accurate.",
     "reference": "Bi-encoders embed independently enabling fast search; cross-encoders jointly encode for higher accuracy."},

    {"query": "What is vector indexing?",
     "context": ["Vector indexing organises embeddings for efficient nearest-neighbour queries.",
                 "FAISS IndexFlatL2 performs exact L2-distance search."],
     "answer": "Vector indexing structures embeddings so similarity searches run efficiently.",
     "reference": "Vector indexing organises embedding data structures for fast nearest-neighbour retrieval."},

    {"query": "What is semantic chunking?",
     "context": ["Semantic chunking splits documents at topically coherent boundaries.",
                 "It preserves meaning by avoiding splits mid-topic."],
     "answer": "Semantic chunking splits text at topic boundaries to preserve coherent meaning in each chunk.",
     "reference": "Semantic chunking divides documents at topic-coherent boundaries rather than fixed character counts."},

    {"query": "Why is reranking important in RAG?",
     "context": ["The initial retrieval may return many loosely relevant chunks.",
                 "Reranking re-scores those chunks to surface the most relevant ones first."],
     "answer": "Reranking re-scores retrieved chunks so the most relevant ones are used in generation.",
     "reference": "Reranking improves RAG quality by promoting the most relevant chunks above noisier retrieval results."},

    # ── LoRA & Fine-tuning (31-38) ────────────────────────────────────────────
    {"query": "What is LoRA?",
     "context": ["LoRA adds low-rank decomposition matrices to frozen pre-trained weights.",
                 "Only the adapter matrices are trained, reducing parameter count significantly."],
     "answer": "LoRA adds low-rank decomposition matrices to frozen pre-trained weights; only the adapter matrices are trained, reducing parameter count significantly.",
     "reference": "LoRA adds low-rank adapter matrices to frozen weights so only a small fraction of parameters are trained."},

    {"query": "Why is LoRA memory efficient?",
     "context": ["LoRA only trains the small adapter matrices, not the full model weights.",
                 "The base model stays frozen, reducing GPU memory requirements."],
     "answer": "LoRA trains only small adapter matrices while keeping base weights frozen, drastically reducing memory use.",
     "reference": "LoRA's efficiency comes from training tiny adapter matrices instead of the large frozen base weights."},

    {"query": "What is the rank in LoRA?",
     "context": ["The rank r controls the size of the low-rank decomposition matrices.",
                 "Lower r means fewer trainable parameters."],
     "answer": "The rank r determines the size of the low-rank matrices; lower rank means fewer trainable parameters.",
     "reference": "LoRA's rank r is the dimension of the decomposition matrices, controlling trainable parameter count."},

    {"query": "What is PEFT?",
     "context": ["PEFT stands for Parameter-Efficient Fine-Tuning.",
                 "It encompasses methods like LoRA, prefix tuning, and adapters."],
     "answer": "PEFT (Parameter-Efficient Fine-Tuning) covers methods like LoRA that fine-tune models with few parameters.",
     "reference": "PEFT methods like LoRA, prefix tuning, and adapters enable fine-tuning with minimal parameter updates."},

    {"query": "What is instruction fine-tuning?",
     "context": ["Instruction fine-tuning trains models on (instruction, response) pairs.",
                 "It teaches models to follow user instructions more accurately."],
     "answer": "Instruction fine-tuning trains on instruction-response pairs to improve instruction-following behaviour.",
     "reference": "Instruction fine-tuning uses instruction-response data to teach models to follow natural language commands."},

    {"query": "What is catastrophic forgetting?",
     "context": ["When a model is fine-tuned on new data, it can forget previously learned knowledge.",
                 "This is called catastrophic forgetting."],
     "answer": "Catastrophic forgetting is when fine-tuning on new data causes a model to lose previously learned knowledge.",
     "reference": "Catastrophic forgetting occurs when fine-tuning overwrites prior knowledge stored in model weights."},

    {"query": "How does LoRA prevent catastrophic forgetting?",
     "context": ["LoRA keeps the original model weights frozen.",
                 "Only the small adapter matrices are updated, preserving base knowledge."],
     "answer": "LoRA freezes base weights and only trains adapters, so prior knowledge is not overwritten.",
     "reference": "By freezing the base model and training only adapters, LoRA avoids overwriting pre-trained knowledge."},

    {"query": "What is QLoRA?",
     "context": ["QLoRA quantises the base model to 4-bit precision.",
                 "LoRA adapters are trained on top of the quantised model."],
     "answer": "QLoRA combines 4-bit quantisation of the base model with LoRA adapters for extreme memory efficiency.",
     "reference": "QLoRA uses 4-bit quantisation plus LoRA adapters to fine-tune large models with minimal memory."},

    # ── Evaluation metrics (39-44) ────────────────────────────────────────────
    {"query": "What is faithfulness in RAG evaluation?",
     "context": ["Faithfulness measures how much of the answer is grounded in the retrieved context.",
                 "A faithful answer does not hallucinate facts not present in the context."],
     "answer": "Faithfulness measures whether the answer is grounded in retrieved context without hallucination.",
     "reference": "A faithful RAG answer uses only information present in the retrieved context, avoiding hallucinations."},

    {"query": "What is answer relevance in RAG evaluation?",
     "context": ["Answer relevance measures how well the answer addresses the original question.",
                 "A relevant answer directly responds to the query."],
     "answer": "Answer relevance measures how directly the generated answer addresses the user's question.",
     "reference": "Answer relevance evaluates whether the answer is on-topic and directly responds to the query."},

    {"query": "What is context precision?",
     "context": ["Context precision measures what fraction of retrieved chunks are relevant.",
                 "High precision means the retriever surfaces mostly relevant chunks."],
     "answer": "Context precision is the fraction of retrieved chunks that are relevant to the query.",
     "reference": "Context precision measures how many of the retrieved chunks are actually relevant to the question."},

    {"query": "What is ROUGE score?",
     "context": ["ROUGE measures overlap between a generated text and a reference text.",
                 "ROUGE-1 measures unigram overlap; ROUGE-L measures longest common subsequence."],
     "answer": "ROUGE measures n-gram overlap between generated and reference texts.",
     "reference": "ROUGE computes word-level overlap between generated summaries or answers and reference texts."},

    {"query": "What is BLEU score?",
     "context": ["BLEU measures n-gram precision between generated text and one or more references.",
                 "It is widely used to evaluate machine translation quality."],
     "answer": "BLEU measures n-gram precision between generated and reference text, commonly used for translation.",
     "reference": "BLEU is an n-gram precision metric originally designed for machine translation evaluation."},

    {"query": "What is hallucination in language models?",
     "context": ["Hallucination occurs when a model generates facts not supported by the input context.",
                 "It is a key challenge in deploying LLMs for factual tasks."],
     "answer": "Hallucination is when a language model generates unsupported or false information.",
     "reference": "LLM hallucination refers to generating plausible-sounding but factually incorrect or unsupported content."},

    # ── Embeddings & Vector search (45-50) ────────────────────────────────────
    {"query": "What are sentence embeddings?",
     "context": ["Sentence embeddings represent entire sentences as dense vectors.",
                 "Models like sentence-transformers produce embeddings that capture semantic meaning."],
     "answer": "Sentence embeddings are dense vectors representing the semantic content of sentences.",
     "reference": "Sentence embeddings encode the meaning of full sentences as fixed-size dense vectors."},

    {"query": "What is cosine similarity?",
     "context": ["Cosine similarity measures the angle between two vectors.",
                 "Vectors pointing in the same direction have similarity close to 1."],
     "answer": "Cosine similarity measures the angle between vectors; similar vectors have values near 1.",
     "reference": "Cosine similarity measures directional similarity between vectors, ranging from -1 to 1."},

    {"query": "What is the all-MiniLM-L6-v2 model?",
     "context": ["all-MiniLM-L6-v2 is a sentence-transformer model producing 384-dimensional embeddings.",
                 "It is fast and suitable for CPU deployment."],
     "answer": "all-MiniLM-L6-v2 produces 384-dim sentence embeddings and is efficient enough for CPU use.",
     "reference": "all-MiniLM-L6-v2 is a lightweight sentence-transformer outputting 384-dim embeddings with good speed."},

    {"query": "What is approximate nearest neighbour search?",
     "context": ["ANN search finds vectors close to a query vector without exhaustive comparison.",
                 "It trades some accuracy for significant speed gains at scale."],
     "answer": "Approximate nearest-neighbour search finds close vectors quickly by trading some accuracy for speed.",
     "reference": "ANN search finds approximately nearest vectors without comparing against every item in the index."},

    {"query": "What is embedding dimensionality?",
     "context": ["Embedding dimensionality is the number of values in each vector.",
                 "Higher dimensions can capture more nuance but require more memory and compute."],
     "answer": "Embedding dimensionality is the size of the vector; higher dimensions encode more information but cost more.",
     "reference": "The dimensionality of an embedding vector determines its capacity to encode semantic information."},

    {"query": "What is the role of embeddings in RAG?",
     "context": ["Embeddings convert text chunks and queries into comparable vector representations.",
                 "Similarity search over embeddings retrieves the most relevant chunks."],
     "answer": "Embeddings convert text to vectors, enabling similarity search to retrieve relevant chunks for RAG.",
     "reference": "In RAG, embeddings represent text as vectors so similarity search can find contextually relevant chunks."},
]


async def _score_pair(rouge, bleu, sim, pair: dict) -> dict:
    r = await rouge.ascore(response=pair["answer"], reference=pair["reference"])
    b = await bleu.ascore(response=pair["answer"],  reference=pair["reference"])
    s = await sim.ascore(response=pair["answer"],   reference=pair["reference"])
    faith = faithfulness(pair["answer"], pair["context"])
    relev = relevance(pair["answer"], pair["query"])
    conci = conciseness(pair["answer"])
    return {
        "query":        pair["query"][:55],
        "rouge_l":      round(r.value, 4),
        "bleu":         round(b.value, 4),
        "str_sim":      round(s.value, 4),
        "faithfulness": round(faith, 4),
        "relevance":    round(relev, 4),
        "conciseness":  round(conci, 4),
    }


async def run_ragas_eval(pairs: list[dict]) -> dict:
    rouge = RougeScore()
    bleu  = BleuScore()
    sim   = NonLLMStringSimilarity()

    results = []
    for pair in pairs:
        res = await _score_pair(rouge, bleu, sim, pair)
        results.append(res)

    keys = ["rouge_l", "bleu", "str_sim", "faithfulness", "relevance", "conciseness"]
    aggregate = {k: round(sum(r[k] for r in results) / len(results), 4) for k in keys}
    return {"n": len(results), "per_pair": results, "aggregate": aggregate}


def main():
    print(f"\nRAGAS evaluation — {len(QA_PAIRS)} QA pairs\n" + "─" * 56)
    report = asyncio.run(run_ragas_eval(QA_PAIRS))

    print(f"\n{'Query':<55}  ROUGE   BLEU   Sim   Faith  Relev  Conci")
    print("─" * 115)
    for r in report["per_pair"]:
        print(f"{r['query']:<55}  "
              f"{r['rouge_l']:.3f}  {r['bleu']:.3f}  {r['str_sim']:.3f}  "
              f"{r['faithfulness']:.3f}  {r['relevance']:.3f}  {r['conciseness']:.3f}")

    print("\n── Aggregate ─────────────────────────────────────────────────")
    for k, v in report["aggregate"].items():
        print(f"  {k:<18}: {v:.4f}")

    out = os.path.join(os.path.dirname(__file__), "ragas_50_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report → {out}")
    return report


if __name__ == "__main__":
    main()
